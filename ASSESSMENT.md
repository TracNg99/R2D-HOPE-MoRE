# R²D-HOPE-VLM: Architecture Assessment & Enhancement Proposal

**Date:** March 2026  
**Scope:** LM-only core (sub-20M target); VLM frontend deferred.

---

## 1. Idea Summary

The notebook proposes a language model that combines four research ideas:

| Component | Source Paper | Claimed Benefit |
|---|---|---|
| **HOPE / Nested Learning** | Behrouz et al., Google Research 2024 | Parameter efficiency via weight-shared recursive depth |
| **MoRE (Mixture of Recursive Experts)** | Novel composition | Specialised computation without replicating weights |
| **Diffusion-of-Thought (DoT)** | Ye et al., NeurIPS 2024 | Parallel, non-autoregressive reasoning |
| **Glyph Frontend** | `zai-org/Glyph` | 1M+ token compression via rendered images |

The ambition — achieve high performance, long-context, and reasoning in a sub-20M parameter model — is valid and tractable. The *combination* of ideas is creative. However, the implementation contains critical flaws that prevent it from being a viable research artifact.

---

## 2. What Is Correct

### 2.1 Weight-Shared Recursion (HOPE Core)
**Sound foundation.** The Relaxed Recursive Transformer paper (Google DeepMind, arXiv:2410.20672) confirms that shared-weight recursion with depth=16–32 can match independently-parameterised stacks of layers at a fraction of the parameter count. This is the strongest idea in the notebook.

### 2.2 Diffusion-of-Thought Loss Framing
The use of Gaussian-diffusion noise over token embeddings, with a recursive denoiser, is consistent with the DoT paper (arXiv:2402.07754). Training to predict added noise (MSE on embeddings) is architecturally correct.

### 2.3 Factorised Embedding
`FactorizedEmbedding(vocab_size, d_embedding=256, d_model=768)` is a well-known parameter-saving technique (used in ALBERT) that is appropriate here.

### 2.4 Gradient Checkpointing
`torch.utils.checkpoint` is applied inside the recursive core. Correct usage given the O(depth) memory footprint.

### 2.5 NoiseScheduler Concept
DDPM-style linear beta schedule for embedding diffusion is a valid starting point.

---

## 3. Critical Bugs & Flaws

### 3.1 **[FATAL] Parameter Count Massively Exceeds Sub-20M Target**

The configured model (`d_model=768, num_heads=12, num_experts=4, nested_depth=16`) is far too large. Estimates:

```
Token Embedding (FactorizedEmbedding):
  vocab_size=8192, d_emb=256, d_model=768
  = 8192×256 + 256×768 = 2.1M + 0.2M ≈ 2.3M

HOPE_Input_Injection (cross-attention per loop step, SHARED):
  MHA(d=768, heads=4): Q,K,V,O projections = 4 × 768×768 = 2.36M
  LayerNorm: ~1.5K
  → ~2.36M

Router:
  768 → 4 = 3K (negligible)

Expert_PatternMatcher (SWA-style MLP + conv):
  Linear(768→1536) + Linear(1536→768) = 1536×768×2 ≈ 2.36M
  Conv1d(768,768,k=7): 768×768×7 ≈ 4.1M
  → ~6.5M per expert

Expert_LogicalReasoner: ~6.5M
Expert_MemoryConsolidator: ~0.77M
MultiBranchBlock (if similar scale): ~6.5M

MoRE_Recursive_Core total (all shared, applied 16×):
  Injector + Router + 4 Experts ≈ 2.36 + 0.003 + 6.5 + 6.5 + 0.77 + 6.5 = ~22.6M

Time embedding: Embed(1000,768) + Linear(768,768) ≈ 0.77M + 0.59M ≈ 1.36M
Output projector: 768×768 ≈ 0.59M

TOTAL (rough): ~27M
```

**The sub-20M target is violated by at least 35%.** The Conv1d branches alone in 3 of 4 experts are the primary offender (`768×768×7 ≈ 4.1M` per expert).

**Root cause:** `d_model=768` with full-width Conv1d branches cannot fit sub-20M. The correct strategy is either (a) reduce `d_model` to 384 or (b) use depthwise convolution.

### 3.2 **[FATAL] MoRE Routing Is Dense (Not Sparse) — Defeats the Efficiency Purpose**

```python
# Cell 45 — MoRE_Recursive_Core
for j, expert in enumerate(self.experts):
    out_x, out_state = checkpoint(expert, x_injected, ...)
    aggregated_x += out_x * expert_weights[:, j].unsqueeze(-1).unsqueeze(-1)
```

All four experts run on every token at every loop step. This is a **weighted dense mixture**, not sparse MoE. It provides no compute savings and quadruples the FLOPs compared to running a single expert.

**Correct approach:** Top-1 or Top-2 sparse routing (select K experts per token, skip the rest). The router should produce a hard selection (straight-through estimator or token-choice with capacity).

### 3.3 **[FATAL] Diffusion Inference Denoising Schedule Is Numerically Unstable**

```python
# generate_with_hope (cell 0)
alpha = 1.0 - (i + 1) / steps
draft = (draft - (1.0 - alpha) * pred_noise) / (alpha**0.5)
```

When `i=0` (last step), `alpha = 1/steps = 0.05` (for steps=20), so `alpha**0.5 ≈ 0.22`. The denominator does not go to zero here, but the schedule is ad hoc and not derived from the forward process. It does not match the `NoiseScheduler.add_noise` used during training, creating a **train/inference mismatch** that will cause the model to generate noise at test time.

**Correct approach:** The `NoiseScheduler` must expose `alphas_cumprod`; inference must use the matching reverse DDPM or DDIM update rule using the same schedule parameters.

### 3.4 **[FATAL] Recurrent State Is Zero-Initialised Per Forward Pass**

```python
# Cell 47 — Combined_R2D_HOPE_MoRE_VLM_Diffusion.forward
initial_recurrent_state = torch.zeros(batch_size, self.config.d_model, device=...)
```

The `Expert_MemoryConsolidator` is supposed to build a persistent memory across the recursive loop, but its state is reset to zero at every call to `forward()`. This means the recurrent state is useful within one diffusion timestep but **cannot propagate information across timesteps**, eliminating its purpose as a memory mechanism.

**Correct approach:** Pass `recurrent_state` as an argument to `generate_with_hope` and thread it across denoising steps, or maintain it as a buffer on the model.

### 3.5 **[MAJOR] HOPE Stabilising Alpha Is Initialized to Zero**

```python
self.alpha = nn.Parameter(torch.zeros(1))
```

At init, `alpha=0` means `h_{t+1} = h_t + 0 * f(h') = h_t` — the model is an identity function. This is intentional as a "stable start" technique (similar to ReZero/MAGNETO), but it requires a **warm-up period** before gradients can flow to the experts. Without an explicit learning rate warm-up, the experts receive no gradient signal for many steps. The notebook has no warm-up schedule.

### 3.6 **[MAJOR] Router Operates on Sequence Mean, Not Per-Token**

```python
# Cell 43 — Router
def forward(self, x):
    pooled_x = x.mean(dim=1)
    return F.softmax(self.fc(pooled_x), dim=-1)  # [B, num_experts]
```

A single routing decision is made per sequence, not per token. All tokens in a sequence are routed to the same expert combination. This is not MoE — it is sequence-level mixture, which cannot learn position- or content-dependent expert specialisation within a sequence.

**Correct approach:** Route per-token: `self.fc(x)` → `[B, S, num_experts]`, then apply top-K masking and renormalize.

### 3.7 **[MAJOR] SWA Expert Is Actually a Standard MLP (Mislabeled)**

```python
# Expert_PatternMatcher / R2D_MoRE_Expert
self.swa = nn.Sequential(
    nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
)
```

This is a standard FFN, not Sliding Window Attention (SWA). SWA is an attention mechanism where each token attends only to a local window. The mislabeling suggests a copy-paste error. The architecture description says "SWA Branch" but the code implements a dense FFN.

### 3.8 **[MAJOR] Glyph Frontend Misuses the Vision Tower**

```python
visual_outputs = self.glyph_vlm.vision_tower(inputs.pixel_values)
image_features = visual_outputs[0]  # (Num_Images, Patches_Per_Img, Vis_Dim)
```

`zai-org/Glyph` is built on GLM-4V. Calling `.vision_tower` directly bypasses the language model backbone, the multimodal projector, and any instruction following. The resulting features are raw ViT patch embeddings — not semantically compressed document representations. The claimed "1M+ token compression" is not achieved by this code path.

### 3.9 **[MINOR] Tokenizer Vocabulary Mismatch**

The tokenizer is configured for `vocab_size=8192` but trains to produce `15215` tokens (observed in output). The `config_combined_more.vocab_size = 8192` is never corrected, creating a silent mismatch between embedding table size and actual tokenizer vocabulary that would cause index-out-of-bounds errors or truncation at runtime.

### 3.10 **[MINOR] Hardcoded HuggingFace API Key in Notebook**

```python
huggingface_api_key='YOUR_HUGGINGFACE_API_KEY_HERE'
```

Credentials are committed in plaintext. The key should be revoked immediately if this notebook is ever shared.

---

## 4. Architectural Coherence Assessment

| Design Goal | Current Status | Verdict |
|---|---|---|
| Sub-20M parameters | ~27M estimated | ❌ Fails |
| Long-context via recursion | Correct principle, wrong scale | ⚠️ Partial |
| Efficient MoE routing | Dense routing, negates benefit | ❌ Fails |
| Diffusion reasoning | Train/inference schedule mismatch | ❌ Broken |
| Persistent recurrent memory | Zeroed per forward call | ❌ Broken |
| Parameter sharing (HOPE) | Correctly implemented | ✅ Works |
| Factorised embedding | Correctly implemented | ✅ Works |

---

## 5. Proposed Corrected Architecture

### 5.1 Target Constraints
- **Parameters:** ≤ 18M (trainable LM core)
- **Context:** Effective context via ROPE + windowed attention (no hard limit)
- **Depth illusion:** HOPE-style shared block, 16–24 recursive loops
- **Sparsity:** True Top-2 sparse MoE, 4 experts, 2 active per token

### 5.2 Recommended Hyperparameters

```python
d_model       = 384       # Was 768; this halves cross-params quadratically
d_ffn         = 1024      # ~2.67× d_model (vs 2×, saves ~15% FFN params)
num_heads     = 6         # Matches d_model=384 (64 dims/head)
head_dim      = 64
num_experts   = 4
top_k_experts = 2         # Sparse: only 2 of 4 run per token
nested_depth  = 20        # More loops, same param count
vocab_size    = 16384     # Aligned to actual tokenizer output
d_embedding   = 192       # Factorised embedding bottleneck
window_size   = 512       # Local attention window (per RAttention paper)
rope_base     = 500000    # Extended RoPE for long-context generalisation
```

**Estimated parameter count at d_model=384:**

```
FactorizedEmbedding(16384, 192, 384):  16384×192 + 192×384 ≈ 3.15M + 0.07M = 3.22M
Shared HOPE Block:
  - CrossAttn injector (MHA, 4 heads, 384):  4×384×384 = 0.59M
  - SlidingWindowAttn (true SWA, heads=6):   4×384×384 = 0.59M  [Q,K,V,O]
  - Sparse Router: 384→4 = 1.5K (negligible)
  - Expert FFN (× 4 experts, each 384→1024→384): 2×384×1024×4 = 3.15M (all 4 defined, 2 run)
  - MemoryConsolidator gate: 2×384 → 384 = 0.3M
  - LayerNorms + biases: ~0.03M
  Total shared block: ≈ 4.67M

Time embedding (diffusion):  Embed(1000, 384) + Linear(384,384) ≈ 0.38M + 0.15M = 0.53M
Output projector: 384→384 = 0.15M
LM head (tied to embedding): 0 additional (weight-tied to FactorizedEmbedding.output)

TOTAL: 3.22 + 4.67 + 0.53 + 0.15 ≈ 8.57M  [well within 20M]
```

This leaves headroom to scale `d_model` to 512 if needed (~13.5M).

### 5.3 Key Architectural Corrections

#### Correction 1: True Sparse Per-Token Routing
```python
class SparseRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.fc = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, S, D]
        logits = self.fc(x)  # [B, S, E]
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)  # [B, S, K]
        weights = F.softmax(topk_logits, dim=-1)  # normalise among selected
        return weights, topk_idx  # sparse selection
```

#### Correction 2: Aligned Noise Schedule (Train/Inference Parity)
```python
class CosineNoiseScheduler(nn.Module):
    def __init__(self, T: int = 1000):
        super().__init__()
        t = torch.arange(T + 1, dtype=torch.float64)
        alphas_bar = torch.cos(((t / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = torch.clamp(1 - alphas_bar[1:] / alphas_bar[:-1], max=0.999)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0).float())
        self.register_buffer("betas", betas.float())

    def add_noise(self, x0, t):
        abar = self.alphas_cumprod[t].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        return x0 * abar.sqrt() + noise * (1 - abar).sqrt(), noise

    def ddim_step(self, x_t, pred_noise, t, t_prev):
        abar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        abar_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1) if t_prev >= 0 \
                    else torch.ones_like(abar_t)
        x0_pred = (x_t - (1 - abar_t).sqrt() * pred_noise) / abar_t.sqrt()
        x0_pred = x0_pred.clamp(-5, 5)
        return abar_prev.sqrt() * x0_pred + (1 - abar_prev).sqrt() * pred_noise
```

#### Correction 3: Persistent Recurrent State Across Diffusion Steps
```python
def generate_with_hope(model, context_tokens, prompt_emb, num_thoughts=64, steps=20):
    B, _, D = prompt_emb.shape
    device = prompt_emb.device
    draft = torch.randn(B, num_thoughts, D, device=device)
    # Persistent across denoising steps
    recurrent_state = torch.zeros(B, D, device=device)

    timesteps = list(range(steps - 1, -1, -1))
    for i, t_val in enumerate(timesteps):
        t = torch.tensor([t_val] * B, device=device).long()
        t_prev = t_val - 1

        with torch.no_grad():
            pred_noise, recurrent_state = model(
                draft, t, context_tokens, prompt_emb, recurrent_state
            )
        draft = model.scheduler.ddim_step(draft, pred_noise, t, t_prev)

    return draft
```

#### Correction 4: True Sliding Window Attention (replace mislabeled SWA MLP)
```python
class LocalSlidingWindowAttention(nn.Module):
    """O(n·w) attention — each token attends to ±window_size/2 neighbours."""
    def __init__(self, d_model: int, num_heads: int, window_size: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, S, H, Hd]
        # Use torch.nn.functional.scaled_dot_product_attention with a local mask
        # For a clean implementation, use the sliding window mask
        q = q.permute(0, 2, 1, 3)  # [B, H, S, Hd]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # Build causal + local attention mask
        mask = self._make_local_causal_mask(S, x.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.norm(x + self.out(out))

    def _make_local_causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(S, device=device)
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window_size // 2
        causal = idx.unsqueeze(0) >= idx.unsqueeze(1)
        return (mask & causal).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
```

#### Correction 5: Warm-up Schedule for Alpha (ReZero)
The `alpha` parameter should start at zero but have a **scaled learning rate**:
```python
# In optimizer setup
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "alpha" not in n], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if "alpha" in n], "lr": 1e-3},
], weight_decay=0.01)
# Plus cosine LR schedule with linear warmup (1000 steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, ...)
```

---

## 6. Corrected Architecture Diagram

```
INPUT TOKENS  ──► FactorizedEmbedding(16384→192→384)
                         │
                         ▼
              ┌── RoPE positional encoding ──┐
              │                              │
              │    R²D-HOPE SHARED BLOCK     │  ← 20 recursive iterations
              │  ┌────────────────────────┐  │
              │  │  CrossAttn(q=state,    │  │  ← HOPE injection
              │  │    kv=context_tokens)  │  │
              │  │         ↓              │  │
              │  │  SparseRouter(top_k=2) │  │  ← per-token routing
              │  │         ↓              │  │
              │  │  ┌──┬──┬──┬──┐       │  │
              │  │  │E0│E1│E2│E3│       │  │  ← 4 experts (2 active)
              │  │  └──┴──┴──┴──┘       │  │    each: SWA-attn OR FFN
              │  │   weighted sum →      │  │
              │  │  MemoryGate(x, state) │  │  ← persistent recurrent state
              │  │   state ← gate(x, s) │  │
              │  │  h = h + α·output    │  │  ← ReZero residual
              │  └────────────────────────┘  │
              └──────── × 20 loops ──────────┘
                         │
                         ▼
              CosineScheduler.ddim_step(·)      ← diffusion denoising
                         │
                         ▼
              OutputProjector(384→384)
                         │
                         ▼
               LM Head (weight-tied to emb)
```

---

## 7. Data & Training Corrections

| Issue | Current | Recommended |
|---|---|---|
| Vocab size mismatch | Config=8192, actual=15215 | Align config to tokenizer output |
| BPE pre-tokeniser | `Whitespace()` only | Use `ByteLevel` pre-tokeniser (handles all Unicode) |
| Synthetic data pipeline | `filter([line for line in preload_dataset])` — **bug**: `filter` requires a callable, this will raise `TypeError` | `[line for line in preload_dataset if line.strip()]` |
| Training objective | Noise prediction (MSE) only | Add auxiliary next-token CE loss on clean embeddings |
| No LR warm-up | Flat LR from step 0 | 1000-step warm-up + cosine decay |
| Diffusion steps | 1000 training / 20 inference | Use DDIM with 10–50 inference steps |

---

## 8. Implementation Priority (Ordered)

1. **Fix vocab size alignment** — blocks any training run
2. **Reduce `d_model` to 384** — brings params into target range
3. **Replace dense routing with `SparseRouter`** — core efficiency claim
4. **Align noise schedule** — fixes train/inference mismatch
5. **Thread `recurrent_state` across diffusion steps** — fixes memory module
6. **Replace mislabeled SWA MLP with true local attention** — fixes expert diversity
7. **Add LR warm-up + alpha LR scaling** — fixes gradient starvation
8. **Implement `ddim_step` on `NoiseScheduler`** — enables efficient inference

---

## 9. Summary Verdict

The idea is sound and the combination of HOPE recursion + sparse MoE + diffusion reasoning is a legitimate research direction for sub-20M long-context models. The paper foundations (Nested Learning, DoT, Relaxed Recursive Transformers) are real and validated. However, the current notebook has **4 fatal bugs** that prevent the model from training correctly or meeting its parameter budget, and **3 major structural errors** that invalidate core architectural claims. None of these are fundamental blockers to the idea — they are implementation errors that are correctable.

The proposed corrections above resolve all critical issues while keeping the parameter count at ~8.6M (d_model=384) or ~13.5M (d_model=512), well within the sub-20M target.
