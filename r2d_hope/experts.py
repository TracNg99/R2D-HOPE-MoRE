"""
Expert modules for R²D-HOPE-MoRE.

Four specialists:
  E0 — LocalPatternExpert:    True sliding-window self-attention (fixes mislabeled SWA)
  E1 — LogicalReasoningExpert: Gated FFN with expanded inner dim
  E2 — MemoryConsolidator:    GRU-style gate that updates persistent recurrent state
  E3 — ConvolutionalExpert:   Depthwise-separable conv (replaces full Conv1d — 1/768 params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import R2DConfig
from .embeddings import RotaryEmbedding


# ---------------------------------------------------------------------------
# E0: True Sliding Window Attention
# ---------------------------------------------------------------------------

class LocalPatternExpert(nn.Module):
    """
    Causal sliding-window self-attention.
    Each token attends only to the previous `window_size` tokens → O(n·w).
    Uses RoPE for positional encoding.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        d = config.d_model

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.rope = RotaryEmbedding(config.head_dim, rope_base=config.rope_base)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, S, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, S, Hd]

        q, k = self.rope(q, k)

        # Causal local mask: attend to at most window_size past tokens
        mask = self._local_causal_mask(S, x.device)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.norm(x + self.out_proj(out)), state  # state unchanged

    def _local_causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(S, device=device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)  # [S, S]: row=query, col=key
        # Causal: query_pos >= key_pos; local: within window_size
        mask = (dist >= 0) & (dist <= self.window_size)
        # SDPA expects True = keep, False = mask out; additive float mask
        return mask.unsqueeze(0).unsqueeze(0).float().masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))


# ---------------------------------------------------------------------------
# E1: Gated FFN (Logical Reasoning Expert)
# ---------------------------------------------------------------------------

class LogicalReasoningExpert(nn.Module):
    """
    SwiGLU-style gated FFN: uses two parallel projections × sigmoid gate.
    More expressive than a plain ReLU FFN with the same parameter count.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        d, d_ffn = config.d_model, config.d_ffn
        self.gate_proj = nn.Linear(d, d_ffn, bias=False)
        self.up_proj = nn.Linear(d, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.norm(x + out), state  # state unchanged


# ---------------------------------------------------------------------------
# E2: Memory Consolidator (recurrent state update)
# ---------------------------------------------------------------------------

class MemoryConsolidator(nn.Module):
    """
    GRU-style gate that updates the persistent recurrent state vector.
    Fixes the original notebook's Expert_MemoryConsolidator which discarded
    the returned state — here the updated state is explicitly returned and
    threaded across both loop iterations and diffusion timesteps.
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        d = config.d_model
        # Reset and update gates (GRU formulation)
        self.reset_gate = nn.Linear(d * 2, d, bias=True)
        self.update_gate = nn.Linear(d * 2, d, bias=True)
        self.candidate = nn.Linear(d * 2, d, bias=True)
        self.norm_x = nn.LayerNorm(d)
        self.norm_s = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x:     [B, S, D]
        # state: [B, D]
        pooled = self.norm_x(x).mean(dim=1)           # [B, D]
        norm_s = self.norm_s(state)                   # [B, D]
        cat = torch.cat([pooled, norm_s], dim=-1)     # [B, 2D]

        r = torch.sigmoid(self.reset_gate(cat))        # reset gate
        z = torch.sigmoid(self.update_gate(cat))       # update gate
        cat_reset = torch.cat([pooled, r * norm_s], dim=-1)
        h_cand = torch.tanh(self.candidate(cat_reset))
        new_state = (1.0 - z) * state + z * h_cand   # [B, D]

        # x is returned unchanged; only state is updated by this expert
        return x, new_state


# ---------------------------------------------------------------------------
# E3: Depthwise-Separable Convolution Expert
# ---------------------------------------------------------------------------

class ConvolutionalExpert(nn.Module):
    """
    Depthwise-separable Conv1d (kernel=7).
    Replaces the full Conv1d(d,d,k=7) from the original notebook which cost
    768×768×7 ≈ 4.1M params per expert. Depthwise+pointwise costs:
      d_model×1×7  +  d_model×d_model  =  2.7K + 0.15M  (at d=384)
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        d = config.d_model
        self.depthwise = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d, bias=False)
        self.pointwise = nn.Conv1d(d, d, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, D]
        h = x.permute(0, 2, 1)            # [B, D, S]
        h = self.act(self.depthwise(h))
        h = self.pointwise(h).permute(0, 2, 1)  # [B, S, D]
        return self.norm(x + h), state    # state unchanged
