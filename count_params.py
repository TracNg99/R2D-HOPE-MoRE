import torch, sys
sys.path.insert(0, ".")
from r2d_hope import R2DConfig, R2D_HOPE_MoRE

cfg = R2DConfig(
    d_model=384, d_ffn=1024, d_embedding=192, num_heads=6, head_dim=64,
    vocab_size=16384, nested_depth=20, num_experts=4, top_k_experts=2,
    window_size=512, ddim_inference_steps=10, device="cpu"
)
model = R2D_HOPE_MoRE(cfg)
block = model.r2d_core.shared_block

components = {
    "embedding.embed":        sum(p.numel() for p in model.embedding.embed.parameters()),
    "embedding.project":      sum(p.numel() for p in model.embedding.project.parameters()),
    "embedding.norm":         sum(p.numel() for p in model.embedding.norm.parameters()),
    "injector":               sum(p.numel() for p in block.injector.parameters()),
    "router":                 sum(p.numel() for p in block.router.parameters()),
    "E0 LocalPatternExpert":  sum(p.numel() for n,p in block.experts[0].named_parameters()),
    "E1 LogicalReasoning":    sum(p.numel() for p in block.experts[1].parameters()),
    "E2 MemoryConsolidator":  sum(p.numel() for p in block.experts[2].parameters()),
    "E3 ConvExpert":          sum(p.numel() for p in block.experts[3].parameters()),
    "time_embedding":         sum(p.numel() for p in model.time_embedding.parameters()),
    "output_proj":            sum(p.numel() for p in model.output_proj.parameters()),
    "lm_head":                sum(p.numel() for p in model.lm_head.parameters()),
}
for k, v in components.items():
    print(f"  {k:<45} {v:>9,}  ({v/1e6:.3f}M)")
print()
total = model.count_parameters()
print(f"  TOTAL trainable: {total['trainable']:,}  ({total['trainable']/1e6:.2f}M)")
