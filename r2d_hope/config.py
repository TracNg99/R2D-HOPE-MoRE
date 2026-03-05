"""
R²D-HOPE-MoRE Configuration
Target: ≤18M parameters (LM core only, VLM frontend deferred)
"""
from dataclasses import dataclass, field
import torch


@dataclass
class R2DConfig:
    # Dimensions
    d_model: int = 384
    d_ffn: int = 1024        # Expert FFN inner dim (~2.67× d_model)
    d_embedding: int = 192   # Factorised embedding bottleneck
    num_heads: int = 6       # head_dim = d_model // num_heads = 64
    head_dim: int = 64

    # Vocabulary
    vocab_size: int = 16384  # Aligned to actual tokenizer output

    # Recursion / depth
    nested_depth: int = 20   # Number of times the shared block loops

    # MoE
    num_experts: int = 4
    top_k_experts: int = 2   # Sparse: only top_k run per token per step
    expert_capacity_factor: float = 1.25  # overflow buffer for load balancing

    # Local attention
    window_size: int = 512   # Sliding window half-size for local SWA experts
    rope_base: float = 500_000.0  # Extended RoPE for long-context generalisation

    # Diffusion
    num_diffusion_timesteps: int = 1000
    ddim_inference_steps: int = 50

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True  # Tie input ↔ output embeddings (saves ~6M)

    # Runtime
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.head_dim == self.d_model // self.num_heads, (
            f"head_dim={self.head_dim} != d_model/num_heads={self.d_model // self.num_heads}"
        )
        assert self.top_k_experts <= self.num_experts
