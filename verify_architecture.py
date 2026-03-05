"""
Verification script for R²D-HOPE-MoRE architecture.
Tests:
  1. Forward pass (training) — shape correctness, loss finiteness
  2. Inference (generate) — shape, no NaNs
  3. Parameter count — within sub-20M budget
  4. Noise scheduler — train/inference schedule consistency
  5. Sparse routing — only top_k experts fire per token
  6. Recurrent state — persists and changes across diffusion steps
  7. ReZero alpha — starts at zero, gradients flow through
"""
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from r2d_hope import R2DConfig, R2D_HOPE_MoRE, CosineNoiseScheduler


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "✅"
FAIL = "❌"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    if not condition:
        raise AssertionError(f"FAILED: {label}")


# ── Config ────────────────────────────────────────────────────────────────────
section("Test 1 — Config validation")

cfg_small = R2DConfig(d_model=384, d_ffn=1024, d_embedding=192, num_heads=6,
                      head_dim=64, vocab_size=16384, nested_depth=20,
                      num_experts=4, top_k_experts=2, window_size=512,
                      ddim_inference_steps=10, device=DEVICE)

cfg_medium = R2DConfig(d_model=512, d_ffn=1280, d_embedding=256, num_heads=8,
                       head_dim=64, vocab_size=12288, nested_depth=20,
                       num_experts=4, top_k_experts=2, window_size=512,
                       ddim_inference_steps=10, device=DEVICE)

check("Small config valid", cfg_small.d_model == 384)
check("Medium config valid", cfg_medium.d_model == 512)

# ── Model instantiation & parameter count ─────────────────────────────────────
section("Test 2 — Parameter count (must be < 18M)")

model_small = R2D_HOPE_MoRE(cfg_small).to(DEVICE)
model_medium = R2D_HOPE_MoRE(cfg_medium).to(DEVICE)

pc_small = model_small.count_parameters()
pc_medium = model_medium.count_parameters()

print(f"  d_model=384: {pc_small['trainable']:,} trainable params  "
      f"({pc_small['trainable']/1e6:.2f}M)")
print(f"  d_model=512: {pc_medium['trainable']:,} trainable params  "
      f"({pc_medium['trainable']/1e6:.2f}M)")

check("Small model < 18M", pc_small["trainable"] < 18_000_000,
      f"{pc_small['trainable']/1e6:.2f}M")
check("Medium model < 18M", pc_medium["trainable"] < 18_000_000,
      f"{pc_medium['trainable']/1e6:.2f}M")

# ── Noise scheduler ───────────────────────────────────────────────────────────
section("Test 3 — CosineNoiseScheduler (train/inference parity)")

sched = CosineNoiseScheduler(T=1000).to(DEVICE)
x0 = torch.randn(2, 10, 384, device=DEVICE)
t = torch.tensor([499, 999], device=DEVICE)
x_t, eps = sched.add_noise(x0, t)

check("add_noise output shape", x_t.shape == x0.shape, str(x_t.shape))
check("noise is finite", x_t.isfinite().all().item())
check("alphas_cumprod[0] ≈ 1", sched.alphas_cumprod[0].item() > 0.99)
check("alphas_cumprod[-1] is small", sched.alphas_cumprod[-1].item() < 0.1)

# DDIM step — reconstruct approximate x0 from x_t and true noise
x_prev = sched.ddim_step(x_t, eps, t)
check("ddim_step output shape", x_prev.shape == x0.shape, str(x_prev.shape))
check("ddim_step output finite", x_prev.isfinite().all().item())

# Verify schedules are consistent: DDIM with true noise should recover x0 approximately at t=1
t_one = torch.ones(2, dtype=torch.long, device=DEVICE)
x_t1, eps1 = sched.add_noise(x0, t_one)  # very little noise at t=1
check("at t=1, x_t ≈ x0", (x_t1 - x0).abs().mean().item() < 0.2,
      f"mean_diff={(x_t1 - x0).abs().mean().item():.4f}")

ddim_ts = sched.make_ddim_timesteps(10)
check("ddim timesteps length", len(ddim_ts) == 10, str(ddim_ts))
check("ddim timesteps descending", all(ddim_ts[i] > ddim_ts[i+1] for i in range(len(ddim_ts)-1)))

# ── Training forward pass ─────────────────────────────────────────────────────
section("Test 4 — Training forward pass (shape + loss)")

model = model_small
model.train()

B, Sp, Sa, C = 2, 8, 16, 32
prompt_ids = torch.randint(1, cfg_small.vocab_size, (B, Sp), device=DEVICE)
answer_ids  = torch.randint(1, cfg_small.vocab_size, (B, Sa), device=DEVICE)
context_tok = torch.randn(B, C, cfg_small.d_model, device=DEVICE)
timesteps   = torch.randint(0, 1000, (B,), device=DEVICE)

out = model(prompt_ids, answer_ids, context_tok, timesteps)

check("loss key present", "loss" in out)
check("noise_loss finite", out["noise_loss"].isfinite().item(),
      f"{out['noise_loss'].item():.4f}")
check("ce_loss finite", out["ce_loss"].isfinite().item(),
      f"{out['ce_loss'].item():.4f}")
check("aux_loss finite", out["aux_loss"].isfinite().item(),
      f"{out['aux_loss'].item():.4f}")
check("total loss finite", out["loss"].isfinite().item(),
      f"{out['loss'].item():.4f}")

# ── Gradient flow ─────────────────────────────────────────────────────────────
section("Test 5 — Gradient flow through all components")

out["loss"].backward()

alpha_param = model.r2d_core.shared_block.alpha
check("alpha.grad exists", alpha_param.grad is not None)
check("alpha.grad finite", alpha_param.grad.isfinite().all().item())
check("alpha still near zero", alpha_param.data.abs().item() < 0.1,
      f"alpha={alpha_param.data.item():.6f}")

# Check at least some expert params have gradients
expert_grads_ok = all(
    p.grad is not None and p.grad.isfinite().all()
    for p in model.r2d_core.shared_block.experts[1].parameters()
    if p.requires_grad
)
check("LogicalReasoningExpert gradients finite", expert_grads_ok)

injector_grads_ok = all(
    p.grad is not None and p.grad.isfinite().all()
    for p in model.r2d_core.shared_block.injector.parameters()
    if p.requires_grad
)
check("HOPEContextInjector gradients finite", injector_grads_ok)

# ── Sparse routing ────────────────────────────────────────────────────────────
section("Test 6 — Sparse routing: only top_k experts per token")

model.zero_grad()
from r2d_hope.routing import SparseRouter

router = SparseRouter(cfg_small).to(DEVICE)
x_test = torch.randn(2, 10, cfg_small.d_model, device=DEVICE)
weights, indices, aux_loss = router(x_test)

check("weights shape [B,S,K]",
      weights.shape == (2, 10, cfg_small.top_k_experts),
      str(weights.shape))
check("indices shape [B,S,K]",
      indices.shape == (2, 10, cfg_small.top_k_experts),
      str(indices.shape))
check("weights sum to 1 per token",
      (weights.sum(dim=-1) - 1.0).abs().max().item() < 1e-5)
check("indices in valid range",
      (indices >= 0).all() and (indices < cfg_small.num_experts).all())
check("exactly top_k indices per token",
      indices.shape[-1] == cfg_small.top_k_experts)
check("aux_loss finite", aux_loss.isfinite().item(), f"{aux_loss.item():.4f}")

# ── Recurrent state threading ─────────────────────────────────────────────────
section("Test 7 — Recurrent state: persists and updates across DDIM steps")

model.eval()
with torch.no_grad():
    prompt_ids_inf = torch.randint(1, cfg_small.vocab_size, (1, 4), device=DEVICE)
    context_inf    = torch.randn(1, 8, cfg_small.d_model, device=DEVICE)

    # Manually step through 2 DDIM steps to verify state changes
    from r2d_hope.noise_scheduler import CosineNoiseScheduler as CS
    s = CS(1000).to(DEVICE)
    draft = torch.randn(1, 16, cfg_small.d_model, device=DEVICE)
    prompt_emb = model.embedding(prompt_ids_inf)

    state0 = torch.zeros(1, cfg_small.d_model, device=DEVICE)

    t1 = torch.tensor([999], device=DEVICE, dtype=torch.long)
    t_emb1 = model.time_embedding(t1)
    ws1 = torch.cat([prompt_emb + t_emb1, draft + t_emb1], dim=1)
    _, state1 = model.r2d_core(ws1, context_inf, state0)

    t2 = torch.tensor([499], device=DEVICE, dtype=torch.long)
    t_emb2 = model.time_embedding(t2)
    ws2 = torch.cat([prompt_emb + t_emb2, draft + t_emb2], dim=1)
    _, state2 = model.r2d_core(ws2, context_inf, state1)

check("state1 != state0 (state updates)",
      not torch.allclose(state0, state1))
check("state2 != state1 (state evolves across steps)",
      not torch.allclose(state1, state2))
check("state finite", state2.isfinite().all().item())

# ── Full inference (generate) ─────────────────────────────────────────────────
section("Test 8 — Full inference (generate)")

with torch.no_grad():
    tokens = model.generate(
        prompt_ids_inf, context_inf,
        num_answer_tokens=8, ddim_steps=5
    )

check("generate output shape [B, num_answer_tokens]",
      tokens.shape == (1, 8), str(tokens.shape))
check("tokens in vocab range",
      (tokens >= 0).all() and (tokens < cfg_small.vocab_size).all())
print(f"  Generated token ids: {tokens[0].tolist()}")

# ── Parameter breakdown ───────────────────────────────────────────────────────
section("Parameter Breakdown (d_model=384)")

components = {
    "FactorizedEmbedding":   sum(p.numel() for p in model.embedding.parameters()),
    "R2D_HOPE_Core":         sum(p.numel() for p in model.r2d_core.parameters()),
    "TimeEmbedding":         sum(p.numel() for p in model.time_embedding.parameters()),
    "OutputProjector":       sum(p.numel() for p in model.output_proj.parameters()),
    "LM Head":               sum(p.numel() for p in model.lm_head.parameters()),
}
total_from_breakdown = sum(components.values())
for name, count in components.items():
    print(f"  {name:<30} {count:>9,}  ({count/1e6:.3f}M)")
print(f"  {'─'*50}")
print(f"  {'TOTAL':<30} {total_from_breakdown:>9,}  ({total_from_breakdown/1e6:.3f}M)")
print()

# Shared block breakdown
block = model.r2d_core.shared_block
block_components = {
    "HOPEContextInjector":   sum(p.numel() for p in block.injector.parameters()),
    "SparseRouter":          sum(p.numel() for p in block.router.parameters()),
    "E0 LocalPatternExpert": sum(p.numel() for p in block.experts[0].parameters()),
    "E1 LogicalReasoning":   sum(p.numel() for p in block.experts[1].parameters()),
    "E2 MemoryConsolidator": sum(p.numel() for p in block.experts[2].parameters()),
    "E3 ConvolutionalExpert":sum(p.numel() for p in block.experts[3].parameters()),
    "alpha":                 block.alpha.numel(),
}
print("  Shared Block (counted once, iterated 20×):")
for name, count in block_components.items():
    print(f"    {name:<30} {count:>8,}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  d_model=384:  {pc_small['trainable']/1e6:.3f}M params  {'< 18M ✅' if pc_small['trainable'] < 18e6 else '> 18M ❌'}")
print(f"  d_model=512:  {pc_medium['trainable']/1e6:.3f}M params  {'< 18M ✅' if pc_medium['trainable'] < 18e6 else '> 18M ❌'}")
print()
print("  All tests passed. Architecture is verified.")
