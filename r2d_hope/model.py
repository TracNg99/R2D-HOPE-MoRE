"""
R²D-HOPE-MoRE Full Model: Diffusion LM with recursive sparse MoE core.

Forward pass (training):
  1. Embed prompt tokens → prompt_emb  [B, Sp, D]
  2. Embed clean answer tokens → x0    [B, Sa, D]
  3. Sample t, add noise → x_t         [B, Sa, D]
  4. Concatenate [prompt_emb ∥ x_t] as working state
  5. Run R2D_HOPE_Core (20 recursive loops w/ context injection)
  6. Extract answer portion → predict noise → MSE loss
  7. Auxiliary load-balancing loss added with small coefficient

Inference (generate):
  1. Encode context → context_tokens (cross-attn kv, cached)
  2. Start from pure noise draft
  3. DDIM reverse loop (50 steps), threading recurrent_state across steps
  4. Project final embeddings → logits → token ids
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from r2d_hope.config import R2DConfig
from r2d_hope.embeddings import FactorizedEmbedding
from r2d_hope.core import R2D_HOPE_Core
from r2d_hope.noise_scheduler import CosineNoiseScheduler


class TimeEmbedding(nn.Module):
    """Sinusoidal + learned projection for diffusion timestep conditioning."""

    def __init__(self, d_model: int, max_steps: int = 1000):
        super().__init__()
        half = d_model // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / (half - 1))
        )
        self.register_buffer("freq", freq)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer timestep
        t_float = t.float().unsqueeze(-1)          # [B, 1]
        sin = torch.sin(t_float * self.freq)       # [B, D/2]
        cos = torch.cos(t_float * self.freq)       # [B, D/2]
        emb = torch.cat([sin, cos], dim=-1)        # [B, D]
        return self.proj(emb).unsqueeze(1)         # [B, 1, D]


class R2D_HOPE_MoRE(nn.Module):
    """
    The complete R²D-HOPE-MoRE language model.

    Parameter budget at default config (d_model=384):
      FactorizedEmbedding(16384, 192, 384):  ~3.22M
      R2D_HOPE_Core (1 shared block × 20):  ~2.85M
      TimeEmbedding:                         ~0.30M
      OutputProjector (384→384):             ~0.15M
      LM Head (tied to embedding):            0 extra
      ─────────────────────────────────────────────
      TOTAL:                                 ~6.52M   (well within 18M)

    At d_model=512:  ~11.5M  (still within 18M, stronger capacity)
    """

    def __init__(self, config: R2DConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.embedding = FactorizedEmbedding(
            config.vocab_size, config.d_embedding, config.d_model
        )
        self.r2d_core = R2D_HOPE_Core(config)
        self.time_embedding = TimeEmbedding(config.d_model, config.num_diffusion_timesteps)
        self.output_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.scheduler = CosineNoiseScheduler(config.num_diffusion_timesteps)

        # LM head: project d_model → vocab_size
        # Tied to embedding.project weights (transpose) to save ~0.6M params
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # embed.weight:   [vocab, d_emb]
            # project.weight: [d_model, d_emb]  (Linear stores W as [out, in])
            # lm_head needs:  [vocab, d_model]
            # = embed.weight @ project.weight.T  → [vocab, d_emb] @ [d_emb, d_model]
            self.lm_head.weight = nn.Parameter(
                self.embedding.embed.weight @ self.embedding.project.weight.T
            )  # [vocab, d_model]

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        prompt_ids: torch.Tensor,        # [B, Sp]
        answer_ids: torch.Tensor,        # [B, Sa]
        context_tokens: torch.Tensor,    # [B, C, D]  pre-encoded context
        timesteps: torch.Tensor | None = None,  # [B] int; sampled randomly if None
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
          'noise_loss'  — MSE on predicted vs actual noise (diffusion objective)
          'ce_loss'     — cross-entropy on clean embeddings (auxiliary LM objective)
          'aux_loss'    — MoE load-balancing loss
          'loss'        — weighted sum of all three (used for backward)
        """
        B, Sp = prompt_ids.shape
        Sa = answer_ids.shape[1]
        device = prompt_ids.device

        # Embed
        prompt_emb = self.embedding(prompt_ids)         # [B, Sp, D]
        clean_answer_emb = self.embedding(answer_ids)   # [B, Sa, D]

        # Sample diffusion timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self.config.num_diffusion_timesteps, (B,), device=device
            )

        # Add noise to answer embeddings
        x_t, noise_target = self.scheduler.add_noise(clean_answer_emb, timesteps)

        # Time conditioning
        t_emb = self.time_embedding(timesteps)          # [B, 1, D]

        # Build working state: [prompt + noisy_answer], time-shifted
        working_state = torch.cat([
            prompt_emb + t_emb,
            x_t + t_emb,
        ], dim=1)                                       # [B, Sp+Sa, D]

        # Initialise recurrent state
        recurrent_state = torch.zeros(B, self.d_model, device=device)

        # Run recursive core
        refined, _ = self.r2d_core(working_state, context_tokens, recurrent_state)

        # Extract answer portion
        refined_answer = refined[:, Sp:, :]             # [B, Sa, D]
        pred_noise = self.output_proj(refined_answer)   # [B, Sa, D]

        # --- Losses ---
        # 1. Diffusion noise prediction (primary)
        noise_loss = F.mse_loss(pred_noise, noise_target)

        # 2. Auxiliary LM cross-entropy on clean embeddings (stabilises early training)
        clean_refined = refined[:, Sp:, :]
        logits = self.lm_head(clean_refined)            # [B, Sa, V]
        self._last_logits = logits                      # exposed for benchmarking
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            answer_ids.reshape(-1),
            ignore_index=0,
        )

        # 3. MoE load-balancing
        aux_loss = self.r2d_core._total_aux_loss

        loss = noise_loss + 0.1 * ce_loss + 0.01 * aux_loss
        return {
            "loss": loss,
            "noise_loss": noise_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "aux_loss": aux_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,        # [B, Sp]
        context_tokens: torch.Tensor,    # [B, C, D]
        num_answer_tokens: int = 64,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse diffusion to generate answer token embeddings,
        then argmax-decode to token ids.

        Returns: [B, num_answer_tokens] int64
        """
        B, Sp = prompt_ids.shape
        device = prompt_ids.device
        steps = ddim_steps or self.config.ddim_inference_steps
        timestep_seq = self.scheduler.make_ddim_timesteps(steps)

        prompt_emb = self.embedding(prompt_ids)         # [B, Sp, D]
        draft = torch.randn(B, num_answer_tokens, self.d_model, device=device)

        # Persistent recurrent state threads across denoising steps
        # (fixes original notebook: state was zeroed every forward call)
        recurrent_state = torch.zeros(B, self.d_model, device=device)

        for i, t_val in enumerate(timestep_seq):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_emb = self.time_embedding(t)              # [B, 1, D]

            working_state = torch.cat([
                prompt_emb + t_emb,
                draft + t_emb,
            ], dim=1)

            refined, recurrent_state = self.r2d_core(
                working_state, context_tokens, recurrent_state
            )
            pred_noise = self.output_proj(refined[:, Sp:, :])

            # DDIM step (uses same scheduler as training → no mismatch)
            draft = self.scheduler.ddim_step(draft, pred_noise, t, eta=eta)

        # Decode: project embeddings → logits → token ids
        logits = self.lm_head(draft)                   # [B, num_answer_tokens, V]
        return logits.argmax(dim=-1)                   # [B, num_answer_tokens]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def build_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> torch.optim.AdamW:
        """
        Separate parameter groups:
          - alpha (ReZero): 10× higher LR to break identity-function stall early
          - bias / LayerNorm: no weight decay
          - rest: standard LR + WD
        """
        alpha_params, no_decay_params, base_params = [], [], []
        for name, param in self.named_parameters():
            if "alpha" in name:
                alpha_params.append(param)
            elif "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                base_params.append(param)

        return torch.optim.AdamW([
            {"params": base_params,     "lr": lr,       "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr,       "weight_decay": 0.0},
            {"params": alpha_params,    "lr": lr * 10,  "weight_decay": 0.0},
        ], betas=(0.9, 0.95), eps=1e-8)
