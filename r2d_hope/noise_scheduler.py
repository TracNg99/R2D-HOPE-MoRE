"""
Cosine noise scheduler with aligned train/inference schedule.
Fixes the train-inference mismatch in the original notebook.
"""
import math
import torch
import torch.nn as nn


class CosineNoiseScheduler(nn.Module):
    """
    Cosine beta schedule (Nichol & Dhariwal 2021).
    Exposes both add_noise (training) and ddim_step (inference)
    using the exact same alphas_cumprod buffer — eliminating the
    train/inference mismatch present in the original notebook.
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__()
        self.T = T

        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_bar = f / f[0]
        betas = torch.clamp(1.0 - alphas_bar[1:] / alphas_bar[:-1], max=0.999)
        alphas = 1.0 - betas

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0).float())
        # Pad with 1.0 at index -1 for t_prev = -1 case in ddim_step
        abar_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", abar_prev.float())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
        Returns (x_t, eps).
        """
        abar = self.alphas_cumprod[t].view(-1, 1, 1)  # [B,1,1]
        eps = torch.randn_like(x0)
        x_t = abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Inference (DDIM — deterministic, O(S) steps with S << T)
    # ------------------------------------------------------------------
    def ddim_step(
        self,
        x_t: torch.Tensor,
        pred_eps: torch.Tensor,
        t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM reverse step from t → t-1.
        eta=0 → fully deterministic; eta=1 → DDPM stochastic.
        """
        abar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        abar_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)

        # Predicted x_0
        x0_pred = (x_t - (1.0 - abar_t).sqrt() * pred_eps) / abar_t.sqrt()
        x0_pred = x0_pred.clamp(-10.0, 10.0)  # numerical guard

        # Direction pointing to x_t
        sigma = eta * ((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)).sqrt()
        noise = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)

        x_prev = abar_prev.sqrt() * x0_pred \
                 + (1.0 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt() * pred_eps \
                 + sigma * noise
        return x_prev

    def make_ddim_timesteps(self, inference_steps: int) -> list[int]:
        """Uniformly spaced subset of [0, T-1] for DDIM inference."""
        step = self.T // inference_steps
        return list(range(self.T - 1, -1, -step))[:inference_steps]
