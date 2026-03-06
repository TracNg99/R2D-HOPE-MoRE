"""
Sparse per-token MoE router.
Fixes the original notebook's dense sequence-level routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from r2d_hope.config import R2DConfig


class SparseRouter(nn.Module):
    """
    Per-token Top-K sparse router with auxiliary load-balancing loss.

    Key differences from original notebook Router:
      1. Routes per token [B, S, E], not per sequence [B, E].
      2. Returns a sparse selection mask — only top_k experts execute.
      3. Computes auxiliary loss to prevent expert collapse.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.fc = nn.Linear(config.d_model, config.num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
        Returns:
            weights:   [B, S, K]  — softmax weights for top-K selected experts
            indices:   [B, S, K]  — expert indices (int64)
            aux_loss:  scalar     — load-balancing loss (add to training loss × small coeff)
        """
        logits = self.fc(x)                             # [B, S, E]
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)  # [B, S, K]
        weights = F.softmax(topk_logits, dim=-1)        # normalise among selected K

        # Auxiliary load-balancing loss (Switch Transformer style):
        # Encourages uniform token distribution across experts.
        # loss = E * sum_i(f_i * p_i) where f_i = fraction of tokens routed to i,
        #                                      p_i = mean router probability for i.
        with torch.no_grad():
            f_i = F.one_hot(topk_idx[..., 0], self.num_experts).float().mean(dim=(0, 1))
        p_i = F.softmax(logits, dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_experts * (f_i * p_i).sum()

        return weights, topk_idx, aux_loss
