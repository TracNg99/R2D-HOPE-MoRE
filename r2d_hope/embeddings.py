"""
Factorised Embedding + Rotary Positional Encoding (RoPE).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEmbedding(nn.Module):
    """
    ALBERT-style factorised embedding:
      token_id → d_embedding (small bottleneck) → d_model
    Saves (vocab_size - d_model) * d_embedding parameters vs direct embedding.

    With vocab=16384, d_emb=192, d_model=384:
      Direct:       16384 × 384 = 6.29M
      Factorised:   16384 × 192 + 192 × 384 = 3.22M  (-49%)
    """

    def __init__(self, vocab_size: int, d_embedding: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embedding, padding_idx=0)
        self.project = nn.Linear(d_embedding, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.project(self.embed(token_ids)))

    def get_output_embeddings(self) -> nn.Linear:
        """Returns a tied output projection for the LM head."""
        return self.project


class RotaryEmbedding(nn.Module):
    """
    RoPE (Su et al. 2021) with extended base for long-context generalisation.
    Applied inside each attention block, not as a standalone layer.
    """

    def __init__(self, head_dim: int, rope_base: float = 500_000.0, max_seq_len: int = 131_072):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = -1
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self.max_seq_len = max_seq_len

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._cached_seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
        self._cached_seq_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [B, H, S, Hd]
        Returns rotated (q, k).
        """
        S = q.shape[2]
        self._build_cache(S, q.device, q.dtype)
        cos = self._cos_cached[:S].unsqueeze(0).unsqueeze(0)  # [1,1,S,Hd]
        sin = self._sin_cached[:S].unsqueeze(0).unsqueeze(0)
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin
