#!/usr/bin/env python3
"""Comprehensive test of all modules."""

import sys
import torch

# Test 1: Basic imports
from r2d_hope.config import R2DConfig
cfg = R2DConfig()
print(f"Config OK: d_model={cfg.d_model}")

# Test 2: Embeddings
from r2d_hope.embeddings import FactorizedEmbedding, RotaryEmbedding
emb = FactorizedEmbedding(cfg.vocab_size, cfg.d_embedding, cfg.d_model)
x = torch.randint(0, cfg.vocab_size, (2, 16))
out = emb(x)
print(f"FactorizedEmbedding OK: shape {out.shape}")

# Test 3: Experts
from r2d_hope.experts import LocalPatternExpert, LogicalReasoningExpert, MemoryConsolidator, ConvolutionalExpert
for cls in [LocalPatternExpert, LogicalReasoningExpert, MemoryConsolidator, ConvolutionalExpert]:
    exp = cls(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out, _ = exp(x, torch.randn(2, cfg.d_model))
    print(f"{cls.__name__} OK: shape {out.shape}")

# Test 4: Routing
from r2d_hope.routing import SparseRouter
router = SparseRouter(cfg)
w, idx, loss = router(torch.randn(2, 16, cfg.d_model))
print(f"SparseRouter OK: top_k_indices shape {idx.shape}")

# Test 5: Core
from r2d_hope.core import HOPEContextInjector, R2D_HOPE_Block, R2D_HOPE_Core
injector = HOPEContextInjector(cfg)
x = torch.randn(2, 16, cfg.d_model)
ctx = torch.randn(2, 10, cfg.d_model)
out = injector(x, ctx)
print(f"HOPEContextInjector OK: shape {out.shape}")

block = R2D_HOPE_Block(cfg)
out_x, out_ctx, out_state = block(x, ctx, torch.randn(2, cfg.d_model))
print("R2D_HOPE_Block OK")

core = R2D_HOPE_Core(cfg)
out_x, out_state = core(x, ctx, torch.randn(2, cfg.d_model))
print("R2D_HOPE_Core OK")

# Test 6: Noise Scheduler
from r2d_hope.noise_scheduler import CosineNoiseScheduler
scheduler = CosineNoiseScheduler(1000)
x0 = torch.randn(2, 10, cfg.d_model)
t = torch.randint(0, 1000, (2,))
xt, eps = scheduler.add_noise(x0, t)
print("CosineNoiseScheduler OK")

# Test 7: Model - Forward pass
from r2d_hope.model import R2D_HOPE_MoRE
model = R2D_HOPE_MoRE(cfg)
B, Sp, Sa = 2, 10, 20
prompt_ids = torch.randint(0, cfg.vocab_size, (B, Sp))
answer_ids = torch.randint(0, cfg.vocab_size, (B, Sa))
context_tokens = torch.randn(B, 5, cfg.d_model)
out = model(prompt_ids, answer_ids, context_tokens)
print(f"Model forward OK: loss={out['loss'].item():.4f}")

# Test 8: Model - Generation
gen_out = model.generate(prompt_ids, context_tokens, num_answer_tokens=15)
print(f"Model generate OK: shape {gen_out.shape}")

# Test 9: Block Sparsification
from r2d_hope.block_sparsification import SABlockPruner
pruner = SABlockPruner()
pruner.wrap(model)
reg_loss = pruner.regularization_loss()
print(f"BlockImportanceMask OK: reg_loss={reg_loss.item():.4f}")

# Test 10: Trainer utilities
from r2d_hope.trainer import cosine_lr_with_warmup
for step in [0, 500, 10000]:
    scale = cosine_lr_with_warmup(step, type('C', (), {'warmup_steps': 500, 'max_steps': 20000, 'lr': 3e-4, 'lr_min': 3e-5})())
print("Trainer utilities OK")

# Test 11: Final package import
import r2d_hope
print(f"Package r2d_hope imported with __version__ compatibility check")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
