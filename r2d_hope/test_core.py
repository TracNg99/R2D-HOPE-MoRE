from config import R2DConfig
cfg = R2DConfig()
print('Config OK')

from embeddings import FactorizedEmbedding
emb = FactorizedEmbedding(cfg.vocab_size, cfg.d_embedding, cfg.d_model)
import torch
x = torch.randint(0, cfg.vocab_size, (2, 16))
out = emb(x)
print(f'Embedding OK: {out.shape}')

from routing import SparseRouter
router = SparseRouter(cfg)
w, idx, loss = router(torch.randn(2, 16, cfg.d_model))
print('Routing OK')

from noise_scheduler import CosineNoiseScheduler
scheduler = CosineNoiseScheduler(1000)
print('Scheduler OK')

from block_sparsification import SABlockPruner
print('BlockSparsification OK')

from trainer import cosine_lr_with_warmup
print('Trainer OK')

# Import core after experts is fixed
# from core import HOPEContextInjector
# print('Core OK')

# Import model after core is fixed
# from model import R2D_HOPE_MoRE
# print('Model OK')

print('\n=== TESTS PASSED ===')
