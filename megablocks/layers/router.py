from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.moe_expert_choice:
            #import pdb; pdb.set_trace()
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            ###scores = self.layer(x).softmax(dim=1) # [batch_size, seq_len, dim] -> [batch_size, seq_len, num_experts]
            # Let experts choose the highest prob tokens
            # k = n * c / e (https://arxiv.org/pdf/2202.09368)
            # where n = tokens (in the paper it seems they even do tokens across batches not just within each batch
            #; could ablate this by folding together bs & sq)
            # c = capacity (1 or 2 in the paper) & e = num_experts
            # [batch_size, seq_len, num_experts] -> [batch_size, k, num_experts] (k is not top k here!!!)
            # expert_weights corresponds to matrix G (e x k) in the paper
            # expert_indices corresponds to matrix I (e x k) in the paper
            ###expert_weights, expert_indices = torch.topk(scores, (capacity * sq) // self.args.moe_num_experts, dim=1)

            # Softmax is still taken along expert dim, not token dim.
            # https://github.com/google/flaxformer/blob/399ea3a85e9807ada653fd0de1a9de627eb0acde/flaxformer/architectures/moe/routing.py#L322C7-L322C66
            scores = self.layer(x).softmax(dim=-1) # [batch_size, seq_len, num_experts]
            # [batch_size, num_experts, k]
            expert_weights, expert_indices = torch.topk(scores.transpose(1,2), (capacity * sq) // self.args.moe_num_experts, dim=-1)

        else:
            scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights, p=self.args.moe_normalize_expert_weights,dim=-1, keepdim=True)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices
