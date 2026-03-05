"""
Verification for SA++ block sparsification integration.
Tests:
  1.  wrap() identifies and replaces eligible Linear layers (not router/embed/lm_head)
  2.  BlockImportanceMask is identity when importance=1 (unit test)
  3.  Phase-1 forward pass is finite
  4.  regularization_loss() is finite, positive, and differentiable
  5.  Gradients flow through log_importance AND linear.weight
  6.  prune() applies correct block sparsity per layer type
  7.  Post-pruning forward pass is finite with correct shapes
  8.  HOPE recursion runs correctly with pruned shared block
  9.  Non-zero weight count confirms actual sparsity
 10.  HOPE amplification: pruned block savings scale with nested_depth
 11.  report() output shows sane densities
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from r2d_hope import R2DConfig, R2D_HOPE_MoRE, SABlockPruner
from r2d_hope.block_sparsification import BlockImportanceMask, SPARSITY_CONFIG

DEVICE = "cpu"
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


# ── Shared config ─────────────────────────────────────────────────────────────
cfg = R2DConfig(
    d_model=384, d_ffn=1024, d_embedding=192, num_heads=6, head_dim=64,
    vocab_size=16384, nested_depth=4,  # small depth for fast testing
    num_experts=4, top_k_experts=2, window_size=64,
    ddim_inference_steps=4, device=DEVICE,
)
B, Sp, Sa, C = 2, 6, 8, 16
prompt_ids = torch.randint(1, cfg.vocab_size, (B, Sp))
answer_ids  = torch.randint(1, cfg.vocab_size, (B, Sa))
ctx_tokens  = torch.randn(B, C, cfg.d_model)

# ── Test 1: wrap() identifies correct layers ──────────────────────────────────
section("Test 1 — wrap() replaces eligible Linear layers only")

model = R2D_HOPE_MoRE(cfg)
pruner = SABlockPruner(sparsity_lambda=1e-4, block_size=(16, 16))
pruner.wrap(model)

check("At least one mask created", len(pruner._masks) > 0, f"{len(pruner._masks)} masks")
all_bim = all(isinstance(m, BlockImportanceMask) for _, m in pruner._masks)
check("All replaced modules are BlockImportanceMask", all_bim)
check("Router fc NOT wrapped", not any("router" in n for n, _ in pruner._masks))
check("FFN layers wrapped",
      any(n.endswith(k) for n, _ in pruner._masks for k in ("gate_proj", "up_proj", "down_proj")))
check("Attention projections wrapped",
      any(n.endswith(k) for n, _ in pruner._masks for k in ("q_proj", "qkv")))
check("GRU gates wrapped",
      any(n.endswith(k) for n, _ in pruner._masks for k in ("reset_gate", "update_gate")))

print(f"\n  Wrapped layers ({len(pruner._masks)} total):")
for name, mask in pruner._masks:
    leaf = name.split(".")[-1]
    tgt = SPARSITY_CONFIG.get(leaf, "N/A")
    print(f"    {name:<60} blocks={mask.num_blocks:>4}  target_sparsity={tgt}")

# ── Test 2: BlockImportanceMask is identity when importance=1 ─────────────────
section("Test 2 — BlockImportanceMask is identity when importance=1")

lin_test = nn.Linear(64, 32, bias=False)
bim_test = BlockImportanceMask(lin_test, block_size=(16, 16))

# softplus(x) = 1 when x = log(e - 1)
softplus_inv_1 = math.log(math.e - 1)
with torch.no_grad():
    bim_test.log_importance.fill_(softplus_inv_1)

x_in = torch.randn(4, 64)
with torch.no_grad():
    out_mask = bim_test(x_in)
    out_raw  = lin_test(x_in)

diff = (out_mask - out_raw).abs().max().item()
check("Mask(importance=1) numerically identical to raw Linear",
      diff < 1e-5, f"max_diff={diff:.2e}")

# ── Test 3: Phase-1 forward is finite ─────────────────────────────────────────
section("Test 3 — Phase-1 forward pass is finite")

model.eval()
with torch.no_grad():
    ts = torch.randint(0, 1000, (B,))
    out_p1 = model(prompt_ids, answer_ids, ctx_tokens, ts)

check("Phase-1 total loss finite",  out_p1["loss"].isfinite().item(),
      f"{out_p1['loss'].item():.4f}")
check("Phase-1 noise_loss finite",  out_p1["noise_loss"].isfinite().item())
check("Phase-1 ce_loss finite",     out_p1["ce_loss"].isfinite().item())
check("Phase-1 loss > 0",           out_p1["loss"].item() > 0)

# ── Test 4: regularization_loss is finite, positive, differentiable ───────────
section("Test 4 — regularization_loss() finite, positive, differentiable")

model.train()
reg = pruner.regularization_loss(model)
check("reg_loss scalar",   reg.ndim == 0)
check("reg_loss finite",   reg.isfinite().item(),   f"{reg.item():.6f}")
check("reg_loss > 0",      reg.item() > 0,           f"{reg.item():.6f}")
check("reg_loss requires_grad", reg.requires_grad)

# ── Test 5: Gradients flow through log_importance AND linear.weight ───────────
section("Test 5 — Gradient flow through mask parameters")

model.zero_grad()
out = model(prompt_ids, answer_ids, ctx_tokens)
total_loss = out["loss"] + pruner.regularization_loss(model)
total_loss.backward()

imp_grads_ok = all(
    m.log_importance.grad is not None and m.log_importance.grad.isfinite().all()
    for _, m in pruner._masks
)
check("log_importance.grad finite for all masks", imp_grads_ok)

weight_grads_ok = all(
    m.linear.weight.grad is not None and m.linear.weight.grad.isfinite().all()
    for _, m in pruner._masks
)
check("linear.weight.grad finite through mask", weight_grads_ok)

# ── Test 6: prune() applies correct block sparsity ────────────────────────────
section("Test 6 — prune() applies correct block sparsity per layer type")

model_p = R2D_HOPE_MoRE(cfg)
pruner_p = SABlockPruner(sparsity_lambda=1e-4, block_size=(16, 16))
pruner_p.wrap(model_p)
density_map = pruner_p.prune(model_p)

for name, density in density_map.items():
    leaf = name.split(".")[-1]
    target_sparsity = SPARSITY_CONFIG.get(leaf, 0.0)
    expected_density = 1.0 - target_sparsity
    # tolerance: ±1 block rounding
    any_mask_nb = next(m.num_blocks for n, m in pruner_p._masks if n == name)
    tol = 1.0 / any_mask_nb + 0.01
    ok = abs(density - expected_density) <= tol
    check(f"  {leaf} density ≈ {100*expected_density:.0f}%",
          ok, f"actual={100*density:.1f}%  tol=±{100*tol:.1f}%")

check("All masks marked pruned", all(m.pruned for _, m in pruner_p._masks))

# ── Test 7: Post-pruning forward is finite ────────────────────────────────────
section("Test 7 — Post-pruning forward pass: finite outputs, correct shapes")

model_p.eval()
with torch.no_grad():
    out_pruned = model_p(prompt_ids, answer_ids, ctx_tokens)

check("Pruned loss finite",       out_pruned["loss"].isfinite().item(),
      f"{out_pruned['loss'].item():.4f}")
check("Pruned noise_loss finite", out_pruned["noise_loss"].isfinite().item())
check("Pruned ce_loss finite",    out_pruned["ce_loss"].isfinite().item())

# ── Test 8: HOPE recursion with pruned shared block ───────────────────────────
section("Test 8 — HOPE recursion (nested_depth=4) with pruned block")

with torch.no_grad():
    tokens = model_p.generate(
        prompt_ids[:1], ctx_tokens[:1], num_answer_tokens=4, ddim_steps=2
    )

check("Generate shape (1, 4)", tokens.shape == (1, 4), str(tokens.shape))
check("Tokens in vocab range",
      (tokens >= 0).all().item() and (tokens < cfg.vocab_size).all().item())
print(f"  Generated: {tokens[0].tolist()}")

# ── Test 9: Non-zero weight count confirms sparsity ───────────────────────────
section("Test 9 — Non-zero weight count (effective sparsity confirmed)")

def count_nonzero(m: nn.Module) -> tuple[int, int]:
    nz, tot = 0, 0
    for name, p in m.named_parameters():
        if "weight" in name and "log_importance" not in name:
            tot += p.numel()
            nz  += int(p.count_nonzero().item())
    return nz, tot

model_dense = R2D_HOPE_MoRE(cfg)
nz_d, tot_d = count_nonzero(model_dense)
nz_p, tot_p = count_nonzero(model_p)
sparsity = 1.0 - nz_p / max(1, nz_d)

print(f"  Dense:  {nz_d:>10,} / {tot_d:,} non-zero weights")
print(f"  Pruned: {nz_p:>10,} / {tot_p:,} non-zero weights")
print(f"  Weight sparsity achieved: {100*sparsity:.1f}%")
check("≥ 8% weight sparsity achieved", sparsity >= 0.08, f"{100*sparsity:.1f}%")

# ── Test 10: HOPE amplification factor ────────────────────────────────────────
section("Test 10 — HOPE amplification: pruning benefit × nested_depth")

zero_ffn, total_ffn = 0, 0
for name, mask in pruner_p._masks:
    if any(name.endswith(k) for k in ("gate_proj", "up_proj", "down_proj")):
        total_ffn += mask.num_blocks
        zero_ffn  += int((~mask.block_mask).sum().item())

if total_ffn > 0:
    ffn_sparsity = 100 * zero_ffn / total_ffn
    print(f"  FFN block sparsity:         {ffn_sparsity:.1f}%")
    print(f"  nested_depth:               {cfg.nested_depth}")
    print(f"  Effective FFN compute saved (vs dense 1-block model): "
          f"~{ffn_sparsity:.1f}% × {cfg.nested_depth} = "
          f"{ffn_sparsity * cfg.nested_depth:.0f}% total FFN units saved")
    check("FFN sparsity ≈ 50% (target)", abs(ffn_sparsity - 50.0) < 5.0,
          f"{ffn_sparsity:.1f}%")

# ── Test 11: report() ─────────────────────────────────────────────────────────
section("Test 11 — report() shows sane densities")

rpt = pruner_p.report()
check("Report has layer names", "gate_proj" in rpt or "q_proj" in rpt)
check("Report has TOTAL line", "TOTAL" in rpt)
print()
print(rpt)

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"  Wrapped layers:          {len(pruner._masks)}")
print(f"  Weight sparsity:         {100*sparsity:.1f}%")
print(f"  HOPE amplification:      {cfg.nested_depth}× benefit per pruned block iteration")
print(f"  Overhead (Phase 1):      {sum(m.log_importance.numel() for _,m in pruner._masks):,} "
      f"extra scalar params (negligible)")
print()
print("  All tests passed.")
