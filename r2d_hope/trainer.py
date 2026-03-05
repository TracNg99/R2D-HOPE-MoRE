"""
Training loop for R²D-HOPE-MoRE.

Features:
  - Mixed precision (bfloat16 on A100/H100, float16 on T4/V100)
  - Gradient accumulation (simulate larger batch on small GPU memory)
  - Gradient clipping
  - Linear warmup + cosine LR decay
  - Google Drive checkpointing (Colab-safe)
  - TensorBoard logging (works in Colab via %tensorboard magic)
  - SA++ pruner integration (optional — activated after warmup)
  - Periodic perplexity eval on a held-out slice
"""
from __future__ import annotations

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- Paths ---
    run_name:          str   = "r2d_hope_run1"
    checkpoint_dir:    str   = "./checkpoints"   # set to /content/drive/MyDrive/... in Colab
    log_dir:           str   = "./logs"
    tokenizer_dir:     str   = "./tokenizer"

    # --- Data ---
    dataset_name:      str   = "wikitext"
    dataset_config:    str   = "wikitext-103-raw-v1"
    seq_len:           int   = 512
    batch_size:        int   = 8              # per-device batch
    grad_accum_steps:  int   = 4              # effective batch = 32
    num_workers:       int   = 2

    # --- Optimiser ---
    lr:                float = 3e-4
    lr_min:            float = 3e-5
    weight_decay:      float = 0.1
    beta1:             float = 0.9
    beta2:             float = 0.95
    grad_clip:         float = 1.0

    # --- Schedule ---
    warmup_steps:      int   = 500
    max_steps:         int   = 20_000         # ~1 epoch on WikiText-103 at bs=32, seq=512
    eval_every:        int   = 500
    save_every:        int   = 1_000
    log_every:         int   = 50

    # --- SA++ pruning (optional) ---
    use_sapp:          bool  = False
    sapp_warmup_steps: int   = 5_000          # prune after this many steps
    sapp_lambda:       float = 1e-4

    # --- Hardware ---
    dtype:             str   = "bfloat16"     # "bfloat16" | "float16" | "float32"
    compile_model:     bool  = False          # torch.compile (PyTorch 2.x, faster on A100)

    # --- Vocab (must match model config) ---
    vocab_size:        int   = 16384


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min / cfg.lr + (1.0 - cfg.lr_min / cfg.lr) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    metrics: dict,
    cfg: TrainConfig,
) -> str:
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "metrics":   metrics,
        "train_cfg": asdict(cfg),
    }, path)
    # Keep only last 3 checkpoints to save Drive space
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in all_ckpts[:-3]:
        old.unlink()
    print(f"  Saved checkpoint: {path}")
    return str(path)


def load_latest_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: TrainConfig,
) -> int:
    """Returns the step to resume from (0 if no checkpoint found)."""
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    if not ckpt_dir.exists():
        return 0
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not all_ckpts:
        return 0
    latest = all_ckpts[-1]
    print(f"  Resuming from {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> dict[str, float]:
    model.eval()
    total_loss = total_nll = total_tokens = 0
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        B, S = input_ids.shape

        # build dummy context (zeros — eval without context)
        ctx = torch.zeros(B, 1, model.config.d_model, device=device, dtype=dtype)

        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(input_ids, labels, ctx)

        # CE loss = average NLL per token
        total_nll   += out["ce_loss"].item() * B * S
        total_tokens += B * S
        total_loss   += out["loss"].item()
        n += 1

    model.train()
    avg_nll = total_nll / max(1, total_tokens)
    return {
        "eval_loss": total_loss / max(1, n),
        "eval_nll":  avg_nll,
        "eval_ppl":  math.exp(min(avg_nll, 20)),   # cap at e^20 to avoid inf
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    cfg: TrainConfig,
    pruner=None,     # optional SABlockPruner instance
) -> dict[str, list]:
    """
    Full training loop. Returns history dict with loss/ppl curves.

    Usage:
        from r2d_hope.trainer import train, TrainConfig
        history = train(model, train_loader, eval_loader, TrainConfig())
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    # float16 scaler; bfloat16 doesn't need scaling (native range is sufficient)
    use_scaler = (dtype == torch.float16)

    model = model.to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        print("  torch.compile() enabled")
        model = torch.compile(model)

    optimizer = model.build_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay,
                                       betas=(cfg.beta1, cfg.beta2))
    scaler = GradScaler(enabled=use_scaler)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_lr_with_warmup(s, cfg)
    )

    # Resume if checkpoint exists
    start_step = load_latest_checkpoint(model, optimizer, scaler, cfg)
    if start_step > 0:
        # fast-forward scheduler
        for _ in range(start_step):
            scheduler.step()

    # TensorBoard
    writer = None
    if _TB_AVAILABLE:
        log_dir = Path(cfg.log_dir) / cfg.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

    history: dict[str, list] = {"step": [], "train_loss": [], "eval_ppl": [], "lr": []}
    running_loss = 0.0
    step = start_step
    t0 = time.time()

    print(f"\n{'─'*60}")
    print(f"  Training R²D-HOPE-MoRE on {device} ({cfg.dtype})")
    print(f"  Steps: {start_step} → {cfg.max_steps} | "
          f"Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"{'─'*60}\n")

    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)

    while step < cfg.max_steps:
        # --- accumulate gradients ---
        accum_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            B, S = input_ids.shape

            # Dummy context (pre-training without VLM frontend)
            ctx = torch.zeros(B, 1, model.config.d_model, device=device, dtype=dtype)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(input_ids, labels, ctx)
                loss = out["loss"]
                if pruner is not None and not any(m.pruned for _, m in pruner._masks):
                    loss = loss + pruner.regularization_loss(model)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # --- optimizer step ---
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        step += 1

        running_loss += accum_loss * cfg.grad_accum_steps

        # --- SA++ pruning trigger ---
        if (pruner is not None
                and cfg.use_sapp
                and step == cfg.sapp_warmup_steps
                and not any(m.pruned for _, m in pruner._masks)):
            print(f"\n  [SA++] Pruning at step {step}...")
            density_map = pruner.prune(model)
            print(pruner.report())

        # --- logging ---
        if step % cfg.log_every == 0:
            avg_loss = running_loss / cfg.log_every
            lr_now   = scheduler.get_last_lr()[0]
            elapsed  = time.time() - t0
            tokens_per_sec = cfg.log_every * cfg.batch_size * cfg.grad_accum_steps * S / elapsed
            print(f"  step {step:>6} | loss {avg_loss:.4f} | "
                  f"lr {lr_now:.2e} | {tokens_per_sec:,.0f} tok/s")
            if writer:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/lr",   lr_now,   step)
            history["step"].append(step)
            history["train_loss"].append(avg_loss)
            history["lr"].append(lr_now)
            running_loss = 0.0
            t0 = time.time()

        # --- eval ---
        if step % cfg.eval_every == 0:
            metrics = evaluate(model, eval_loader, device, dtype)
            ppl = metrics["eval_ppl"]
            print(f"  ── eval step {step}: loss={metrics['eval_loss']:.4f}  "
                  f"ppl={ppl:.2f}  nll={metrics['eval_nll']:.4f}")
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}", v, step)
            history["eval_ppl"].append((step, ppl))

        # --- checkpoint ---
        if step % cfg.save_every == 0:
            save_checkpoint(step, model, optimizer, scaler,
                            {"step": step, **history}, cfg)

    # Final checkpoint
    save_checkpoint(step, model, optimizer, scaler, {"step": step, **history}, cfg)
    if writer:
        writer.close()

    print(f"\n  Training complete. Final step: {step}")
    return history
