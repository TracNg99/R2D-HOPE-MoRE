"""
R²D-HOPE-MoRE Shared Recursive Block and Core Engine.

Architecture:
  One shared block is applied nested_depth times (HOPE recursion).
  Each application:
    1. CrossAttn context injection (HOPE input injection)
    2. Sparse Top-K MoE routing (per-token)
    3. Selected experts execute and aggregate
    4. ReZero residual update: h = h + alpha * aggregated
    5. Persistent recurrent state updated by MemoryConsolidator (Expert 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import R2DConfig
from .routing import SparseRouter
from .experts import (
    LocalPatternExpert,
    LogicalReasoningExpert,
    MemoryConsolidator,
    ConvolutionalExpert,
)


class HOPEContextInjector(nn.Module):
    """
    Cross-attention from current working state (query) into frozen context tokens (kv).
    Implements the HOPE 'input injection' mechanism that creates the illusion of depth
    by re-grounding the state against the original context at every loop step.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        d = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_proj = nn.Linear(d, 2 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, S, D]  — working state (query)
        context: [B, C, D]  — compressed context tokens (key/value, static)
        """
        B, S, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, Hd).permute(0, 2, 1, 3)     # [B,H,S,Hd]
        kv = self.kv_proj(context).reshape(B, -1, 2, H, Hd).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)                                                # [B,H,C,Hd]

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.norm(x + self.out_proj(attn_out))


class R2D_HOPE_Block(nn.Module):
    """
    The single shared block iterated nested_depth times.

    Parameter count (d_model=384, 4 experts):
      HOPEContextInjector:  ~0.59M
      SparseRouter:          ~1.5K
      LocalPatternExpert:   ~0.59M  (SWA + RoPE QKV)
      LogicalReasoningExpert:~1.18M (SwiGLU FFN, d_ffn=1024)
      MemoryConsolidator:   ~0.30M
      ConvolutionalExpert:  ~0.15M  (depthwise-sep conv)
      LayerNorms + biases:  ~0.02M
      alpha (scalar):        negligible
    Total shared block: ~2.85M
    Applied 20× but counted once → true param cost = 2.85M
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        self.config = config
        self.injector = HOPEContextInjector(config)
        self.router = SparseRouter(config)
        self.experts = nn.ModuleList([
            LocalPatternExpert(config),       # E0
            LogicalReasoningExpert(config),   # E1
            MemoryConsolidator(config),       # E2  — updates recurrent state
            ConvolutionalExpert(config),      # E3
        ])
        # ReZero / MAGNETO-style init: start as identity, alpha grows with training.
        # Use higher LR on alpha (see optimizer setup in model.py).
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               [B, S, D]
            context:         [B, C, D]  (static — not modified)
            recurrent_state: [B, D]

        Returns:
            updated_x:       [B, S, D]
            context:         [B, C, D]  (pass-through)
            updated_state:   [B, D]
        """
        # 1. HOPE injection
        x_injected = self.injector(x, context)

        # 2. Sparse per-token routing
        weights, indices, aux_loss = self.router(x_injected)   # [B,S,K], [B,S,K], scalar
        self._last_aux_loss = aux_loss  # stored for retrieval in training loop

        B, S, D = x_injected.shape
        aggregated_x = torch.zeros_like(x_injected)
        new_state = recurrent_state

        # 3. Execute only selected experts per token (sparse dispatch)
        for expert_id, expert in enumerate(self.experts):
            # Mask: which (batch, seq) positions route to this expert?
            # indices: [B, S, K]  — check if expert_id appears in top-K for each token
            token_mask = (indices == expert_id).any(dim=-1)  # [B, S]

            if not token_mask.any():
                continue  # skip entirely if no token routed here

            # Gather weight for this expert (0 if not in top-K, softmax weight otherwise)
            k_match = (indices == expert_id).float()               # [B, S, K]
            expert_weight = (weights * k_match).sum(dim=-1)        # [B, S]

            # Run expert via gradient checkpoint for memory efficiency
            out_x, out_state = checkpoint(
                expert, x_injected, recurrent_state, use_reentrant=False
            )

            # Weighted aggregation
            aggregated_x += out_x * expert_weight.unsqueeze(-1)    # [B, S, D]

            # Only MemoryConsolidator (E2) meaningfully updates state;
            # but we let all experts return state and take the weighted update.
            if expert_id == 2:  # MemoryConsolidator
                new_state = out_state

        # 4. ReZero residual: h_{t+1} = h_t + alpha * f(h')
        updated_x = x + self.alpha * aggregated_x

        return updated_x, context, new_state


class R2D_HOPE_Core(nn.Module):
    """
    The Infinite-Depth Engine: one shared block applied nested_depth times.
    Only ONE copy of R2D_HOPE_Block's weights exists — all loops share them.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        self.config = config
        self.shared_block = R2D_HOPE_Block(config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the shared block nested_depth times.

        Returns:
            final_x:     [B, S, D]
            final_state: [B, D]
        """
        current_x = x
        current_state = recurrent_state
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for _ in range(self.config.nested_depth):
            current_x, context, current_state = self.shared_block(
                current_x, context, current_state
            )
            total_aux_loss = total_aux_loss + self.shared_block._last_aux_loss

        self._total_aux_loss = total_aux_loss / self.config.nested_depth
        return current_x, current_state
