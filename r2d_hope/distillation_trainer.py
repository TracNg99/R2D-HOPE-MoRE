"""
Distillation Trainer for R²D-HOPE-MoRE

Extends the base trainer with reasoning trace distillation from OpenRouter teachers.

Key differences from standard training:
  1. Uses reasoning traces (CoT) as training targets, not just next-token prediction
  2. Combines diffusion noise prediction with cross-entropy on teacher reasoning
  3. Optional: Direct Preference Optimization (DPO) on ranked teacher responses

Usage:
    from r2d_hope.distillation_trainer import DistillationTrainer, DistillConfig
    
    trainer = DistillationTrainer(model, distill_cfg, train_cfg)
    trainer.train_from_jsonl('distillation_data.jsonl')
"""
from __future__ import annotations

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .trainer import TrainConfig, cosine_lr_with_warmup, save_checkpoint, load_latest_checkpoint
from .model import R2D_HOPE_MoRE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DistillConfig:
    """Configuration for distillation training."""
    
    # Data
    data_path: str = "./distillation_data.jsonl"
    max_seq_len: int = 512
    
    # Loss weights
    noise_loss_weight: float = 1.0      # Standard diffusion loss
    ce_loss_weight: float = 0.5          # Cross-entropy on reasoning (vs 0.1 in pre-training)
    distill_loss_weight: float = 1.0     # Additional distillation-specific loss
    aux_loss_weight: float = 0.01       # MoE load balancing (unchanged)
    
    # Training
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 10_000             # Distillation typically shorter than pre-training
    warmup_steps: int = 500
    lr: float = 2e-4                   # Slightly lower LR for fine-tuning
    lr_min: float = 2e-5
    
    # Validation
    eval_every: int = 500
    save_every: int = 1_000
    
    # Distillation specific
    use_dpo: bool = False              # Direct Preference Optimization
    dpo_beta: float = 0.1              # DPO temperature
    reasoning_loss_mode: str = "full"   # "full" | "answer_only" | "reasoning_only"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillationDataset(Dataset):
    """
    Loads distillation data from JSONL and yields (prompt_ids, reasoning_ids) pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []
        
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} distillation samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        
        # Tokenize prompt and reasoning
        prompt_ids = s.get("prompt_ids", [])
        reasoning_ids = s.get("reasoning_ids", [])
        
        # Fallback: tokenize text if IDs not present
        if not prompt_ids:
            prompt_ids = self.tokenizer.encode(s["prompt_text"], add_special_tokens=False)
        if not reasoning_ids:
            reasoning_text = s.get("reasoning_text", s.get("reasoning", "") + " " + s.get("answer", ""))
            reasoning_ids = self.tokenizer.encode(reasoning_text, add_special_tokens=False)
        
        # Truncate to max_seq_len
        prompt_ids = prompt_ids[:self.max_seq_len]
        reasoning_ids = reasoning_ids[:self.max_seq_len]
        
        # Pad to same length for batching
        prompt_ids = prompt_ids + [0] * (self.max_seq_len - len(prompt_ids))
        reasoning_ids = reasoning_ids + [0] * (self.max_seq_len - len(reasoning_ids))
        
        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "answer_ids": torch.tensor(reasoning_ids, dtype=torch.long),  # reasoning is the target
        }


def collate_distill(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "prompt_ids": torch.stack([x["prompt_ids"] for x in batch]),
        "answer_ids": torch.stack([x["answer_ids"] for x in batch]),
    }


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Trainer for reasoning distillation from large teacher models.
    """
    
    def __init__(
        self,
        model: R2D_HOPE_MoRE,
        distill_cfg: DistillConfig,
        train_cfg: Optional[TrainConfig] = None,
    ):
        self.model = model
        self.distill_cfg = distill_cfg
        self.train_cfg = train_cfg or TrainConfig()
        
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self.optimizer = model.build_optimizer(
            lr=distill_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: cosine_lr_with_warmup(s, self._effective_train_cfg())
        )
        
        self.global_step = 0
        self.history = {"step": [], "train_loss": [], "eval_loss": [], "lr": []}
    
    def _effective_train_cfg(self) -> TrainConfig:
        """Merge distill config into train config for LR scheduling."""
        cfg = TrainConfig()
        cfg.warmup_steps = self.distill_cfg.warmup_steps
        cfg.max_steps = self.distill_cfg.max_steps
        cfg.lr = self.distill_cfg.lr
        cfg.lr_min = self.distill_cfg.lr_min
        return cfg
    
    def compute_distillation_loss(
        self,
        model_out: dict,
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute distillation loss combining diffusion and reasoning objectives.
        
        Standard R2D forward returns:
          - noise_loss: MSE on predicted noise
          - ce_loss: Cross-entropy on answer tokens
          - aux_loss: MoE load balancing
        
        We weight these differently for distillation and add reasoning-specific losses.
        """
        noise_loss = model_out["noise_loss"]
        ce_loss = model_out["ce_loss"]
        aux_loss = model_out["aux_loss"]
        
        # Get logits from the model's stored _last_logits (for reasoning)
        # Shape: [B, S, V]
        if hasattr(self.model, '_last_logits'):
            logits = self.model._last_logits
            B, S, V = logits.shape
            
            # Mask for valid tokens (non-padding)
            mask = (answer_ids != 0).float()  # [B, S]
            
            # Reasoning loss: cross-entropy on full reasoning chain
            ce_reasoning = F.cross_entropy(
                logits.reshape(-1, V),
                answer_ids.reshape(-1),
                reduction='none',
            ).reshape(B, S)
            ce_reasoning = (ce_reasoning * mask).sum() / mask.sum().clamp(min=1)
        else:
            ce_reasoning = ce_loss  # Fallback
        
        # Combine losses with distillation weights
        # Higher weight on CE because we're directly imitating teacher outputs
        total_loss = (
            self.distill_cfg.noise_loss_weight * noise_loss +
            self.distill_cfg.ce_loss_weight * ce_reasoning +
            self.distill_cfg.aux_loss_weight * aux_loss
        )
        
        return {
            "total": total_loss,
            "noise": noise_loss.detach(),
            "ce_reasoning": ce_reasoning.detach(),
            "aux": aux_loss.detach(),
        }
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step with gradient accumulation."""
        prompt_ids = batch["prompt_ids"].to(self.device)
        answer_ids = batch["answer_ids"].to(self.device)
        B, S = prompt_ids.shape
        
        # Dummy context (no VLM frontend in distillation phase)
        ctx = torch.zeros(B, 1, self.model.config.d_model, device=self.device, dtype=self.dtype)
        
        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            model_out = self.model(prompt_ids, answer_ids, ctx)
            losses = self.compute_distillation_loss(model_out, prompt_ids, answer_ids)
            loss = losses["total"] / self.distill_cfg.grad_accum_steps
        
        # Backward
        self.scaler.scale(loss).backward()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def train(self, data_path: Optional[str] = None) -> dict:
        """
        Main training loop.
        """
        data_path = data_path or self.distill_cfg.data_path
        
        # Build tokenizer from saved path
        from .data import build_or_load_tokenizer
        tokenizer = build_or_load_tokenizer(
            tokenizer_dir=self.train_cfg.tokenizer_dir,
            vocab_size=self.model.config.vocab_size,
        )
        
        # Dataset and loader
        dataset = DistillationDataset(
            data_path,
            tokenizer,
            max_seq_len=self.distill_cfg.max_seq_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.distill_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_distill,
            num_workers=0,  # Safe for Colab
            pin_memory=True,
        )
        
        # Resume if checkpoint exists
        self.global_step = load_latest_checkpoint(
            self.model, self.optimizer, self.scaler, self.train_cfg
        )
        for _ in range(self.global_step):
            self.scheduler.step()
        
        print(f"\n{'─' * 60}")
        print(f"  Distillation Training — Step {self.global_step} → {self.distill_cfg.max_steps}")
        print(f"  Data: {len(dataset)} samples | Batch: {self.distill_cfg.batch_size}×{self.distill_cfg.grad_accum_steps}")
        print(f"{'─' * 60}\n")
        
        self.model.train()
        running_loss = 0.0
        t0 = time.time()
        data_iter = iter(loader)
        
        while self.global_step < self.distill_cfg.max_steps:
            # Accumulate gradients
            accum_losses = {"total": 0.0, "noise": 0.0, "ce_reasoning": 0.0, "aux": 0.0}
            
            for _ in range(self.distill_cfg.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                
                step_losses = self.train_step(batch)
                for k, v in step_losses.items():
                    accum_losses[k] += v
            
            # Optimizer step
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            
            # Logging
            running_loss += accum_losses["total"]
            if self.global_step % self.train_cfg.log_every == 0:
                avg_loss = running_loss / self.train_cfg.log_every
                lr_now = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                
                print(f"  step {self.global_step:>6} | loss {avg_loss:.4f} | "
                      f"ce {accum_losses['ce_reasoning']:.4f} | "
                      f"lr {lr_now:.2e} | {elapsed:.1f}s")
                
                self.history["step"].append(self.global_step)
                self.history["train_loss"].append(avg_loss)
                self.history["lr"].append(lr_now)
                running_loss = 0.0
                t0 = time.time()
            
            # Checkpointing
            if self.global_step % self.distill_cfg.save_every == 0:
                save_checkpoint(
                    self.global_step,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.history,
                    self.train_cfg,
                )
        
        # Final save
        save_checkpoint(
            self.global_step,
            self.model,
            self.optimizer,
            self.scaler,
            self.history,
            self.train_cfg,
        )
        
        print(f"\n✓ Distillation complete at step {self.global_step}")
        return self.history


# ---------------------------------------------------------------------------
# Direct Preference Optimization (DPO) variant
# ---------------------------------------------------------------------------

class DPOTrainer(DistillationTrainer):
    """
    DPO variant that learns from ranked teacher responses (chosen vs rejected).
    
    Requires data in format:
      {
        "prompt_ids": [...],
        "chosen_ids": [...],      # preferred reasoning
        "rejected_ids": [...],    # less preferred reasoning
      }
    """
    
    def compute_dpo_loss(
        self,
        policy_logits_chosen: torch.Tensor,
        policy_logits_rejected: torch.Tensor,
        reference_logits_chosen: torch.Tensor,
        reference_logits_rejected: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """
        DPO loss from Rafailov et al.
        """
        # Log-probs under policy
        policy_chosen_logps = torch.log_softmax(policy_logits_chosen, dim=-1)
        policy_rejected_logps = torch.log_softmax(policy_logits_rejected, dim=-1)
        
        # Log-probs under reference (frozen teacher, or previous checkpoint)
        with torch.no_grad():
            ref_chosen_logps = torch.log_softmax(reference_logits_chosen, dim=-1)
            ref_rejected_logps = torch.log_softmax(reference_logits_rejected, dim=-1)
        
        # DPO objective
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps
        
        loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
        return loss


# ---------------------------------------------------------------------------
# Utility: Generate + Train pipeline
# ---------------------------------------------------------------------------

def generate_and_train(
    model: R2D_HOPE_MoRE,
    api_key: str,
    prompts: list[str],
    teachers: list[str],
    output_dir: str = "./distill_output",
    train_cfg: Optional[TrainConfig] = None,
) -> dict:
    """
    End-to-end: Generate traces from OpenRouter, then distill.
    """
    from .distillation_data import OpenRouterDistiller, TEACHER_MODELS
    
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "distillation_data.jsonl")
    
    # Generate
    print("=" * 60)
    print("Phase 1: Generating reasoning traces from teachers")
    print("=" * 60)
    
    distiller = OpenRouterDistiller(api_key=api_key)
    distiller.generate_dataset(
        prompts=prompts,
        teachers=[TEACHER_MODELS.get(t, t) for t in teachers],
        output_path=data_path,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Phase 2: Distillation training")
    print("=" * 60)
    
    distill_cfg = DistillConfig(
        data_path=data_path,
        max_steps=5_000,  # Shorter for fine-tuning
    )
    
    trainer = DistillationTrainer(model, distill_cfg, train_cfg)
    history = trainer.train()
    
    return history
