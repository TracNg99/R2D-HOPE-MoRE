# R²D-HOPE-MoRE — LLMOps Master Plan
**Version 1.0 | Architecture-grounded, cost-first, production-ready**

---

## Table of Contents

1. [Executive Assessment](#1-executive-assessment)
2. [Architecture Constraints Analysis](#2-architecture-constraints-analysis)
3. [Pillar 1 — Reasoning RL Training](#3-pillar-1--reasoning-rl-training)
4. [Pillar 2 — SOTA Fine-tuning On Demand](#4-pillar-2--sota-fine-tuning-on-demand)
5. [Pillar 3 — Automated Benchmarking & Auto-Adjust](#5-pillar-3--automated-benchmarking--auto-adjust)
6. [Pillar 4 — CI/CD & Cloud Deployment](#6-pillar-4--cicd--cloud-deployment)
7. [Pillar 5 — Backend API Server](#7-pillar-5--backend-api-server)
8. [Vercel Deployment Assessment](#8-vercel-deployment-assessment)
9. [Cost Matrix](#9-cost-matrix)
10. [Verification & LLMOps Standards](#10-verification--llmops-standards)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Consolidated Guideline](#12-consolidated-guideline)

---

## 1. Executive Assessment

### Model Profile (from full codebase audit)

| Property | Value |
|---|---|
| Architecture | Diffusion LM + Sparse MoE + Recursive core |
| Parameters | 6.52M (small) / 13.65M (medium) |
| Weights in float16 | ~13MB / ~27MB |
| Vocab | 16 384 (custom BPE) |
| Forward type | **Non-autoregressive**: DDIM reverse diffusion (50 steps × 20 recursive loops = 1 000 core passes per generation) |
| Training losses | MSE noise prediction + 0.1× CE + 0.01× MoE aux |
| Existing RL support | `DistillConfig.use_dpo = True` (stub, unverified) |
| Existing fine-tune | `DistillationTrainer` + `DPOTrainer` classes |
| Tests | Component-level only; no integration suite |
| CI/CD | None |
| API | None |
| Deployment | None |

### Feasibility Summary

| Pillar | Feasibility | Confidence | Key Risk |
|---|---|---|---|
| 1. RL reasoning | High | 90% | GRPO needs diffusion-specific adaptation |
| 2. On-demand fine-tune | Very High | 98% | Model small enough for full FT in < 5 min |
| 3. Auto-benchmark | High | 95% | Custom eval harness needed (non-standard model) |
| 4. CI/CD + Cloud | Very High | 99% | Straightforward GitHub Actions + HF Spaces |
| 5. Backend API | Very High | 99% | FastAPI, well-understood domain |

---

## 2. Architecture Constraints Analysis

Understanding these constraints is what separates a working plan from a broken one.

### 2.1 Diffusion ≠ Autoregressive — The Most Important Constraint

Standard RL methods (PPO, GRPO, REINFORCE) assume an **autoregressive policy**:

```
policy π(token_t | context, token_0..t-1)  →  per-token log-probs  →  KL divergence
```

R²D-HOPE generates via **reverse diffusion**:

```
draft[B, Sa, D] ~ N(0,1)
for t in [999 → 0]:   # 50 DDIM steps
    pred_noise = model(prompt_emb, draft, t)
    draft = ddim_step(draft, pred_noise, t)
tokens = argmax(lm_head(draft))
```

**Implication**: There is no `log_prob(token_t | prefix)`. The "policy" is the entire denoising trajectory.

**Solutions that work:**
- **REINFORCE / GRPO** applied at the **sequence level** (treat full denoised output as one action)
- **DPO** with preferred/rejected full sequences (already stubbed in codebase)
- **DDPO** (Denoising Diffusion Policy Optimization) — the canonical RL method for diffusion models

**Solutions that do NOT work without major surgery:**
- Standard PPO with per-token value estimates
- RLHF with token-level KL penalty

### 2.2 The Shared Block — Gradient Flow Consideration

The core applies one shared block 20 times. Gradients flow through all 20 iterations:

```
Loss → ∂L/∂output_20 → ∂L/∂output_19 → ... → ∂L/∂output_1 → ∂L/∂shared_params
```

This means:
- **Gradient checkpointing is mandatory** for RL training (already implemented)
- RL reward signal will affect all 20 loop iterations simultaneously
- The `alpha` (ReZero) parameter is the critical bottleneck — start RL with a frozen alpha schedule

### 2.3 Inference Speed Baseline

On a T4 GPU (Colab free):
- Single forward: ~2ms per loop iteration × 20 loops = ~40ms
- Full DDIM generation (50 steps × 20 loops): ~2 seconds
- On CPU: ~20× slower → ~40 seconds

**This determines deployment target** (see Section 8 on Vercel).

---

## 3. Pillar 1 — Reasoning RL Training

### 3.1 Technique Selection

#### Tier 1: GRPO (Group Relative Policy Optimization) — **Recommended**
DeepSeek's technique used to train R1. Adapted for diffusion:

```
For each prompt p:
    Generate G=8 completions {y_1, ..., y_8} via DDIM
    Compute reward r_i for each (rule-based or verifier)
    Baseline = mean(r_1..r_8)
    Advantage_i = r_i - baseline
    Loss = -mean(advantage_i * log_p_approx(y_i | p))
```

**Diffusion adaptation**: `log_p_approx(y | p)` is approximated as the **negative MSE** between predicted and actual noise at the final denoising step. This is the ELBO lower bound on `log p(y|p)`.

#### Tier 2: DPO (Direct Preference Optimization) — **Already stubbed**
```
Given: preferred sequence y_w, rejected sequence y_l
Loss = -log σ(β × (log_p(y_w|p) - log_p(y_l|p) - log_p_ref(y_w|p) + log_p_ref(y_l|p)))
```
For diffusion: use noise-prediction MSE as `log_p` surrogate. `DPOTrainer` class exists in `distillation_trainer.py`, needs the `log_p_diffusion()` helper implemented.

#### Tier 3: REINFORCE-Leave-One-Out (RLOO) — **Simplest, start here**
```
Loss = -reward(y) × MSE_noise(y)  # MSE used as proxy for -log_p
```

### 3.2 Reward Functions

No expensive reward model needed for the first pass:

```python
# Rule-based rewards — zero cost
def reward_math(completion: str, answer: str) -> float:
    """Extract final number and compare."""
    predicted = extract_final_number(completion)
    return 1.0 if predicted == answer else 0.0

def reward_format(completion: str) -> float:
    """Penalize malformed outputs."""
    has_reasoning = bool(re.search(r'(step|because|therefore)', completion, re.I))
    has_answer = bool(re.search(r'answer|result|=', completion, re.I))
    return 0.5 * has_reasoning + 0.5 * has_answer

def reward_length(completion: str, target_len: int = 100) -> float:
    """Soft length reward to prevent degenerate short outputs."""
    return min(len(completion.split()) / target_len, 1.0)

def combined_reward(completion, answer):
    return 0.7 * reward_math(completion, answer) \
         + 0.2 * reward_format(completion) \
         + 0.1 * reward_length(completion)
```

For math reasoning (GSM8K), rule-based rewards achieve parity with reward models at this model scale.

### 3.3 Training Dataset

Use free reasoning datasets:
- **GSM8K** (grade school math, 7 500 train) — primary
- **MATH** (harder math, 12 500 problems) — secondary
- **OpenHermes** (reasoning CoT, 1M) — auxiliary

All available via HuggingFace datasets, no cost.

### 3.4 Implementation Plan

**New file**: `r2d_hope/rl_trainer.py`

```python
@dataclass
class RLConfig:
    algorithm: str = "grpo"        # "grpo" | "dpo" | "rloo"
    group_size: int = 8            # G: completions per prompt (GRPO)
    reward_fn: str = "math"        # "math" | "format" | "combined"
    kl_coeff: float = 0.01         # KL penalty against reference model
    lr: float = 5e-6               # Much lower than SFT
    max_steps: int = 5_000
    ddim_steps_rl: int = 10        # Fewer DDIM steps during RL (speed)
    rollout_batch_size: int = 4    # Prompts per rollout batch
    reward_clip: float = 5.0       # Clip extreme rewards
```

**Training loop**:
1. Load frozen reference model (copy of checkpoint)
2. For each batch of prompts:
   - Generate G completions using student (few DDIM steps for speed)
   - Score all completions with reward function
   - Compute advantages (GRPO) or preference pairs (DPO)
   - Compute RL loss using noise-prediction MSE as log-prob surrogate
   - Backprop with gradient clipping
3. Evaluate on GSM8K every 500 steps

**Cost**: Free (runs on Colab T4)

---

## 4. Pillar 2 — SOTA Fine-tuning On Demand

### 4.1 Why Full Fine-tuning is the Right Choice Here

At 6–13M parameters:
- Full fine-tuning takes < 5 minutes on T4
- Memory cost: ~26MB weights + ~100MB activations = trivially fits on any GPU
- LoRA overhead (adapter management, merge/unmerge) is not worth it at this scale

**Conclusion**: Full fine-tuning is the SOTA choice for sub-20M parameter models.

### 4.2 When LoRA Makes Sense

Still implement LoRA for:
- **Multiple task-specific adapters** (keep base frozen, swap adapters per task)
- **CPU fine-tuning** (reduce memory when no GPU available)
- **Federated fine-tuning** (send only small deltas)

```python
@dataclass
class LoRAConfig:
    r: int = 8                          # Rank (8–16 is sufficient at this scale)
    alpha: float = 16.0                 # Scaling: effective_lr = alpha/r × lr
    target_modules: list = field(       # Apply to attention projections only
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    dropout: float = 0.05
    merge_before_export: bool = True    # Merge weights before saving
```

### 4.3 Fine-tuning Modes Matrix

| Mode | When to use | Time (T4) | Memory |
|---|---|---|---|
| Full FT | New domain, <5min available | ~3 min | 2GB |
| LoRA (r=8) | Task adapter, keep base frozen | ~2 min | 1.2GB |
| Prefix Tuning | Inference-only hardware | ~1 min | 0.8GB |
| Distillation FT | New reasoning tasks | ~10 min | 2GB |
| RL post-training | Alignment/reward shaping | ~30 min | 2.5GB |

### 4.4 On-Demand API Trigger

Fine-tuning is triggered via API (Section 7):

```
POST /finetune
{
  "mode": "full" | "lora" | "distillation" | "rl",
  "dataset_url": "hf://dataset/name or drive://path/to/data.jsonl",
  "config_overrides": {"lr": 1e-4, "max_steps": 2000},
  "base_checkpoint": "latest" | "step_5000" | "url"
}
```

Returns a `job_id`. Progress streamed via `GET /finetune/{job_id}/status`.

### 4.5 Data Format Standardization

All fine-tuning modes accept one universal format:

```jsonl
{"prompt": "What is 27 × 43?", "response": "Step 1: 27 × 40 = 1080. Step 2: 27 × 3 = 81. Total: 1161.", "task": "math"}
{"prompt": "Summarize this text: ...", "response": "The text discusses...", "task": "summarization"}
```

The data pipeline auto-converts to `(prompt_ids, answer_ids)` regardless of fine-tuning mode.

---

## 5. Pillar 3 — Automated Benchmarking & Auto-Adjust

### 5.1 Benchmark Suite

Custom evaluation harness required because:
- Standard `lm-evaluation-harness` assumes autoregressive models with `log_prob(token | prefix)`
- R²D-HOPE uses DDIM generation → must evaluate by **generating and comparing outputs**

#### Benchmark Targets

| Benchmark | Task | Split Size | Target (small 6M) | Target (medium 13M) |
|---|---|---|---|---|
| GSM8K | Grade math | 1 319 test | ≥ 15% | ≥ 25% |
| ARC-Easy | Science QA | 2 376 test | ≥ 40% | ≥ 55% |
| BoolQ | Yes/No | 3 270 val | ≥ 55% | ≥ 65% |
| HellaSwag | Commonsense | 10 042 val | ≥ 30% | ≥ 40% |
| Perplexity (WikiText-103 val) | Language model | 245 docs | ≤ 60 | ≤ 45 |

**Note**: These are realistic targets for a 6–13M parameter diffusion LM. GPT-2 (117M) achieves ~45% on ARC-Easy for reference.

#### Evaluation Method (Diffusion-Specific)

For multiple-choice (ARC, HellaSwag, BoolQ): **Scoring via noise loss comparison**
```python
def score_choice(model, prompt, choices):
    """Score each choice by its diffusion noise loss (lower = more likely)."""
    scores = []
    for choice in choices:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        choice_ids = tokenizer(choice, return_tensors="pt")["input_ids"]
        ctx = torch.zeros(1, 1, model.config.d_model)
        with torch.no_grad():
            out = model(prompt_ids, choice_ids, ctx)
        scores.append(-out["noise_loss"].item())   # Higher = more likely
    return choices[scores.index(max(scores))]
```

For open-ended (GSM8K): **Generate and extract answer**
```python
def score_gsm8k(model, problem, gold_answer):
    generated = model.generate(tokenizer(problem)["input_ids"], ctx, num_answer_tokens=128)
    text = tokenizer.decode(generated[0])
    predicted = extract_last_number(text)
    return predicted == gold_answer
```

### 5.2 Auto-Adjustment System

**Triggered after every eval run.** Adjusts hyperparameters based on gap from target:

```python
class AutoAdjuster:
    def __init__(self, targets: dict, patience: int = 3):
        self.targets = targets       # {"gsm8k": 0.15, "ppl": 60, ...}
        self.history = []
        self.patience = patience     # Adjustments before escalating

    def adjust(self, metrics: dict, current_cfg: TrainConfig) -> TrainConfig:
        adjustments = {}

        # Rule 1: PPL too high → increase LR or extend training
        if metrics["ppl"] > self.targets["ppl"] * 1.2:
            adjustments["lr"] = current_cfg.lr * 1.5
            adjustments["max_steps"] = int(current_cfg.max_steps * 1.3)

        # Rule 2: Accuracy stagnating → reduce LR (plateau)
        recent = [h.get("gsm8k", 0) for h in self.history[-self.patience:]]
        if len(recent) >= self.patience and max(recent) - min(recent) < 0.01:
            adjustments["lr"] = current_cfg.lr * 0.5

        # Rule 3: Loss spiking → halve grad_clip
        if metrics.get("loss_spike", False):
            adjustments["grad_clip"] = current_cfg.grad_clip * 0.5

        # Apply adjustments
        for k, v in adjustments.items():
            setattr(current_cfg, k, v)

        self.history.append(metrics)
        return current_cfg
```

### 5.3 Benchmark Pipeline

**New file**: `r2d_hope/benchmark.py`

```
BenchmarkRunner
├── run_all(model, tokenizer) → BenchmarkReport
├── run_gsm8k(n_samples=200) → float (accuracy)
├── run_arc_easy(n_samples=200) → float (accuracy)
├── run_boolq(n_samples=200) → float (accuracy)
├── run_hellaswag(n_samples=200) → float (accuracy)
├── run_perplexity(eval_loader) → float
└── generate_report() → JSON + Markdown
```

**Automated trigger points**:
1. End of every training run (via `trainer.py` callback)
2. After every fine-tuning job (via API)
3. Nightly CI schedule (GitHub Actions cron)
4. Pre-deployment gate (must pass minimum thresholds)

### 5.4 Benchmark Output Format

```json
{
  "run_id": "r2d_hope_small_step20000",
  "timestamp": "2026-03-07T10:30:00Z",
  "model_size": "small",
  "checkpoint": "step_20000",
  "metrics": {
    "gsm8k_acc": 0.187,
    "arc_easy_acc": 0.423,
    "boolq_acc": 0.581,
    "hellaswag_acc": 0.312,
    "wikitext_ppl": 52.3
  },
  "targets_met": {
    "gsm8k": true,
    "arc_easy": true,
    "boolq": true,
    "hellaswag": true,
    "ppl": true
  },
  "all_targets_met": true,
  "adjustments_applied": []
}
```

---

## 6. Pillar 4 — CI/CD & Cloud Deployment

### 6.1 Infrastructure Stack

```
GitHub Repository
       │
       ├── Push to any branch → Run Tests (pytest)
       ├── PR to main → Tests + Lint + Type check
       └── Push to main → Tests → Docker Build → Deploy
                                         │
                               ┌─────────┴──────────┐
                               │                    │
                    HuggingFace Spaces         Docker Registry
                    (Inference API)            (GitHub Container)
                    Free A10G GPU              ghcr.io/tracng99/r2d-hope
```

### 6.2 GitHub Actions Workflows

#### Workflow 1: `test.yml` — Every push/PR
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12'}
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v --cov=r2d_hope --cov-report=xml
      - run: python -c "from r2d_hope import R2DConfig, R2D_HOPE_MoRE; print('Import OK')"
```

#### Workflow 2: `deploy.yml` — Push to main
```yaml
name: Deploy
on:
  push:
    branches: [main]
    paths: ['r2d_hope/**', 'backend/**', 'requirements.txt']
jobs:
  deploy-api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t ghcr.io/tracng99/r2d-hope-api:latest ./backend
      - name: Push to GitHub Container Registry
        run: |
          echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker push ghcr.io/tracng99/r2d-hope-api:latest
      - name: Deploy to HuggingFace Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install huggingface_hub
          python scripts/deploy_to_hf_spaces.py
```

#### Workflow 3: `benchmark.yml` — Nightly schedule
```yaml
name: Nightly Benchmark
on:
  schedule:
    - cron: '0 2 * * *'    # 2 AM UTC daily
  workflow_dispatch:         # Manual trigger
jobs:
  benchmark:
    runs-on: ubuntu-latest   # CPU-only for perplexity; GPU via HF Spaces for accuracy
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python scripts/run_benchmarks.py --output benchmark_results.json
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-${{ github.run_number }}
          path: benchmark_results.json
      - name: Comment on latest commit
        run: python scripts/post_benchmark_comment.py
```

### 6.3 Docker Container

**`backend/Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Model package
COPY r2d_hope/ ./r2d_hope/
COPY backend/ ./backend/

# Pre-download tokenizer (bakes it into image — eliminates cold start)
RUN python -c "
from r2d_hope import build_or_load_tokenizer
build_or_load_tokenizer('./tokenizer', vocab_size=16384)
"

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.4 Deployment Targets

#### Primary: HuggingFace Spaces (Free)
- Free T4 GPU (16GB VRAM)
- Persistent storage for checkpoints
- Gradio or FastAPI app
- 0$ cost for inference serving

**`spaces/app.py`**: Thin wrapper calling the FastAPI backend.

#### Secondary: Modal.com (Pay-per-use, $0 idle)
```python
import modal

app = modal.App("r2d-hope")
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@app.function(gpu="T4", image=image, timeout=120)
def generate(prompt: str, num_tokens: int = 64) -> str:
    from r2d_hope import R2D_HOPE_MoRE, R2DConfig
    # Load model, run DDIM, return text
    ...

@app.local_entrypoint()
def main():
    print(generate.remote("What is 5 + 3?"))
```

Cost: ~$0.0006/request (T4, ~2s inference). 1 000 requests/day = $0.60/day.

#### Tertiary: Render.com (CPU, $7/month)
For lightweight deployment without GPU. 40-second inference acceptable for async use cases.

---

## 7. Pillar 5 — Backend API Server

### 7.1 API Architecture

```
client (any UI / curl / SDK)
         │
         ▼
FastAPI (backend/main.py)
         │
         ├── /generate      → InferenceEngine
         ├── /finetune       → TrainingJobManager
         ├── /benchmark      → BenchmarkRunner
         ├── /checkpoint     → CheckpointManager
         ├── /config         → ConfigManager
         └── /health         → HealthCheck
```

### 7.2 API Specification (OpenAI-Compatible Where Possible)

#### `POST /v1/completions` — Generate text (OpenAI-compatible)
```json
Request:
{
  "prompt": "What is 27 × 43?",
  "max_tokens": 128,
  "temperature": 0.0,
  "model": "r2d-hope-small"
}

Response:
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "model": "r2d-hope-small",
  "choices": [{
    "text": "Step 1: 27 × 40 = 1080. Step 2: 27 × 3 = 81. Answer: 1161.",
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 8, "completion_tokens": 32, "total_tokens": 40},
  "latency_ms": 1843
}
```

#### `POST /v1/finetune` — Start fine-tuning job
```json
Request:
{
  "mode": "full",
  "dataset": {"type": "hf", "name": "gsm8k", "split": "train"},
  "config": {"lr": 1e-4, "max_steps": 2000, "batch_size": 8},
  "base_model": "latest_checkpoint"
}

Response:
{
  "job_id": "ft-xyz789",
  "status": "queued",
  "estimated_duration_minutes": 4,
  "monitor_url": "/v1/finetune/ft-xyz789/status"
}
```

#### `GET /v1/finetune/{job_id}/status` — Poll job
```json
{
  "job_id": "ft-xyz789",
  "status": "running",
  "progress": {"step": 450, "max_steps": 2000, "loss": 2.34},
  "elapsed_seconds": 54,
  "logs_url": "/v1/finetune/ft-xyz789/logs"
}
```

#### `POST /v1/benchmark` — Trigger benchmark run
```json
Request:
{
  "benchmarks": ["gsm8k", "arc_easy", "ppl"],
  "checkpoint": "latest",
  "n_samples": 200
}

Response: {"run_id": "bench-001", "status": "queued"}
```

#### `GET /v1/benchmark/{run_id}` — Get results
```json
{
  "run_id": "bench-001",
  "status": "complete",
  "results": {"gsm8k_acc": 0.187, "arc_easy_acc": 0.423, "ppl": 52.3},
  "targets_met": {"gsm8k": true, "arc_easy": true, "ppl": true},
  "report_url": "/v1/benchmark/bench-001/report.md"
}
```

#### `GET/PUT /v1/config` — Read/write training config
```json
GET response:
{
  "model_size": "small",
  "train_cfg": {"lr": 3e-4, "batch_size": 8, "max_steps": 20000},
  "rl_cfg": {"algorithm": "grpo", "group_size": 8}
}

PUT request: {"train_cfg": {"lr": 1e-4}}   # Partial update OK
```

#### `GET /v1/checkpoints` — List available checkpoints
```json
{
  "checkpoints": [
    {"name": "step_20000", "path": "...", "ppl": 52.3, "timestamp": "..."},
    {"name": "step_15000", "path": "...", "ppl": 58.1, "timestamp": "..."}
  ],
  "latest": "step_20000"
}
```

#### `GET /v1/health` — Health check
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0",
  "gpu_memory_used_gb": 0.5,
  "uptime_seconds": 3600
}
```

### 7.3 FastAPI Implementation Skeleton

**`backend/main.py`**:
```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch, asyncio, uuid

app = FastAPI(title="R²D-HOPE API", version="1.0.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton model loaded at startup
@app.on_event("startup")
async def load_model():
    app.state.engine = InferenceEngine.from_checkpoint("latest")
    app.state.job_manager = TrainingJobManager()

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    result = await app.state.engine.generate_async(req.prompt, req.max_tokens)
    return format_openai_response(result)

@app.post("/v1/finetune")
async def finetune(req: FinetuneRequest, bg: BackgroundTasks):
    job_id = f"ft-{uuid.uuid4().hex[:8]}"
    bg.add_task(app.state.job_manager.run, job_id, req)
    return {"job_id": job_id, "status": "queued"}
```

### 7.4 Streaming Support

Long operations (fine-tuning logs, benchmark progress) stream via **Server-Sent Events**:

```python
@app.get("/v1/finetune/{job_id}/logs")
async def stream_logs(job_id: str):
    from fastapi.responses import StreamingResponse
    async def event_generator():
        async for log_line in job_manager.tail_logs(job_id):
            yield f"data: {log_line}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 8. Vercel Deployment Assessment

### 8.1 The Hard Limits

| Constraint | Vercel Free | Vercel Pro | R²D-HOPE Need |
|---|---|---|---|
| Function timeout | 10s | 60s | **~2s GPU / ~40s CPU** |
| Function memory | 1 024 MB | 3 008 MB | ~500 MB weights + runtime |
| GPU support | ❌ None | ❌ None | Required for reasonable latency |
| Persistent disk | ❌ None | ❌ None | Checkpoints need storage |
| Python process lifetime | Ephemeral | Ephemeral | Model load = cold start ~10s |

### 8.2 Verdict

> **Vercel cannot serve the model inference endpoint.**
>
> Cold start (load 13MB weights + initialize DDIM scheduler): ~5–8s
> DDIM inference on CPU (50 steps × 20 loops): ~40s
> Total: ~45–48s → **exceeds Pro tier 60s timeout on first request, hard fails on free tier**

### 8.3 What Vercel CAN Do (and does it well)

| Component | Vercel Suitable? | Notes |
|---|---|---|
| Frontend dashboard (React/Next.js) | ✅ Perfect | Zero cost on free tier |
| API gateway / proxy | ✅ Good | Route requests to HF Spaces backend |
| Webhook receiver | ✅ Good | GitHub Actions → Vercel → trigger training |
| Static docs / benchmark reports | ✅ Perfect | Serve pre-generated HTML reports |
| Auth middleware | ✅ Good | JWT validation in Edge Functions |
| Model inference | ❌ Never | Timeout + no GPU |

### 8.4 Recommended Split Architecture

```
Vercel (free)                          HuggingFace Spaces (free T4)
┌─────────────────────────┐            ┌──────────────────────────┐
│  Next.js Dashboard      │  HTTPS     │  FastAPI Backend          │
│  - Training UI          │ ─────────► │  - Model inference        │
│  - Benchmark charts     │            │  - Fine-tuning jobs       │
│  - Config editor        │            │  - Checkpoint management  │
│  - Job monitor          │ ◄───────── │  - Benchmark runner       │
└─────────────────────────┘  SSE/JSON  └──────────────────────────┘
         │                                        │
         │ Deploy via GitHub Actions              │ Deploy via HF Hub push
         ▼                                        ▼
    vercel.com                          huggingface.co/spaces/user/r2d-hope
```

This architecture costs **$0/month** at low traffic.

---

## 9. Cost Matrix

### Monthly Cost Estimates

| Component | Free Tier | Paid Option | Recommendation |
|---|---|---|---|
| Model inference | HF Spaces T4 (free, sleeps after 48h idle) | Modal.com ~$0.60/1K reqs | Start free |
| Frontend | Vercel free | Vercel Pro $20/mo | Free tier sufficient |
| CI/CD | GitHub Actions 2 000 min/mo free | GitHub Actions $4/mo | Free tier |
| Storage (checkpoints) | Google Drive 15GB free | GDrive $2.99/mo | Free tier |
| Benchmark compute | GitHub Actions CPU | — | Free |
| RL training | Colab T4 free | Colab Pro $10/mo | Free for prototyping |
| **Total** | **$0/month** | ~$33/month | **Start at $0** |

### When to Pay

Scale to paid when:
- HF Space T4 sleep time is unacceptable (first-request latency > 30s)
- → Move to Modal.com: ~$0.60 per 1 000 requests
- CI/CD minutes exhausted
- → GitHub Pro: $4/month (3 000 minutes)

---

## 10. Verification & LLMOps Standards

### 10.1 Test Pyramid

```
                    ┌─────────────────┐
                    │  E2E / System   │  2 tests: full train+infer, full finetune+benchmark
                    └────────┬────────┘
               ┌─────────────┴──────────────┐
               │    Integration Tests        │  10 tests: trainer, API, benchmark runner
               └─────────────┬──────────────┘
          ┌───────────────────┴──────────────────────┐
          │              Unit Tests                   │  50+ tests: every module, edge cases
          └──────────────────────────────────────────┘
```

### 10.2 Unit Test Targets

**`tests/test_model.py`**:
```python
def test_forward_pass_returns_all_keys():
def test_loss_decreases_with_backward():
def test_generate_returns_correct_shape():
def test_build_optimizer_accepts_betas():      # regression: fixed bug
def test_noise_loss_is_nonnegative():
def test_ce_loss_is_finite():
def test_aux_loss_is_finite():
```

**`tests/test_data.py`**:
```python
def test_make_pretrain_loader_num_workers_zero():   # regression: fixed bug
def test_make_pretrain_loader_split_validation():    # regression: fixed bug
def test_streaming_dataset_yields_correct_shape():
def test_collate_fn_stacks_correctly():
```

**`tests/test_distillation.py`**:
```python
def test_regex_pattern_2_compiles():    # regression: fixed bug
def test_parse_reasoning_returns_tuple():
def test_distillation_dataset_loads():
def test_dpo_trainer_loss_shape():
```

**`tests/test_api.py`**:
```python
def test_completions_endpoint_returns_200():
def test_completions_timeout_within_5s():    # CPU inference must be < 5s for short outputs
def test_finetune_job_creates_and_polls():
def test_benchmark_endpoint_returns_metrics():
def test_health_endpoint():
```

### 10.3 Model Quality Gates (Pre-deployment)

All must pass before deployment artifact is created:

```yaml
# .github/workflows/quality_gate.yml
quality-gate:
  - name: Perplexity check
    run: python scripts/check_ppl.py --max-ppl 80   # Fail if PPL > 80 (catastrophic failure)
  - name: Output coherence
    run: python scripts/check_coherence.py           # Output must contain ≥1 English word
  - name: No crash on edge inputs
    run: python scripts/stress_test.py               # Empty prompt, very long prompt, special tokens
  - name: Latency check
    run: python scripts/check_latency.py --max-ms 5000  # CPU inference < 5s for 64 tokens
```

### 10.4 Regression Testing

After every bug fix, add a regression test immediately:

```python
# tests/regression/test_issue_fixes.py

def test_regex_pattern_2_compiles():
    """Regression: distillation_data.py:194 unclosed parenthesis."""
    from r2d_hope.distillation_data import REASONING_PATTERNS
    assert len(REASONING_PATTERNS) == 3

def test_build_optimizer_accepts_betas():
    """Regression: model.py build_optimizer missing betas parameter."""
    cfg = R2DConfig(); model = R2D_HOPE_MoRE(cfg)
    opt = model.build_optimizer(lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    assert opt is not None

def test_pretrain_loader_has_split_param():
    """Regression: make_pretrain_loader missing split parameter."""
    import inspect
    sig = inspect.signature(make_pretrain_loader)
    assert "split" in sig.parameters

def test_pretrain_loader_num_workers_zero():
    """Regression: IterableDataset deadlock with num_workers>0."""
    import inspect
    src = inspect.getsource(make_pretrain_loader)
    assert "num_workers=0" in src    # Hardcoded in body
```

### 10.5 Monitoring in Production

**Metrics to track**:

| Metric | Tool | Alert Threshold |
|---|---|---|
| Request latency p95 | HF Spaces logs | > 10s |
| Error rate | FastAPI middleware | > 1% |
| GPU memory | `torch.cuda.memory_allocated` | > 90% |
| Benchmark drift | Nightly CI | Any metric drops > 5% |
| Training loss | TensorBoard | NaN or > 10 |
| Fine-tune job failure rate | Job manager | > 10% |

**Drift detection**: Compare nightly benchmark results against baseline. If GSM8K drops > 5% absolute, create GitHub Issue automatically.

```python
# scripts/check_drift.py
baseline = load_json("benchmarks/baseline.json")
current = load_json("benchmarks/latest.json")
for metric, target in baseline.items():
    if current[metric] < target * 0.95:   # 5% relative drop
        create_github_issue(f"Metric drift: {metric} dropped from {target} to {current[metric]}")
```

---

## 11. Implementation Roadmap

### Phase 1 — Foundation (Week 1–2)
**Goal**: Tests, API skeleton, CI pipeline

- [ ] `tests/` directory with unit + regression tests (covers all 3 fixed bugs)
- [ ] `backend/main.py` — FastAPI skeleton with `/health`, `/v1/completions`
- [ ] `backend/Dockerfile` — containerized API
- [ ] `.github/workflows/test.yml` — CI on every push
- [ ] `.github/workflows/deploy.yml` — deploy on main push
- [ ] Verify all existing Colab notebooks pass end-to-end (clean runtime)

**Deliverables**: Green CI, working `/v1/completions` endpoint locally

### Phase 2 — Benchmark System (Week 3)
**Goal**: Automated, reliable evaluation

- [ ] `r2d_hope/benchmark.py` — BenchmarkRunner with 5 benchmarks
- [ ] `scripts/run_benchmarks.py` — CLI runner
- [ ] `scripts/check_drift.py` — Drift detection
- [ ] `.github/workflows/benchmark.yml` — Nightly schedule
- [ ] Establish baseline numbers and commit `benchmarks/baseline.json`
- [ ] `r2d_hope/auto_adjuster.py` — AutoAdjuster class

**Deliverables**: Nightly benchmark report visible on GitHub Actions artifacts

### Phase 3 — Fine-tuning API (Week 4)
**Goal**: On-demand fine-tuning via REST

- [ ] `backend/training.py` — TrainingJobManager with async job queue
- [ ] `POST /v1/finetune` + `GET /v1/finetune/{id}/status`
- [ ] `GET /v1/finetune/{id}/logs` — SSE streaming
- [ ] `r2d_hope/lora.py` — LoRA adapter implementation (optional mode)
- [ ] Data ingestion from HuggingFace Hub and Drive URLs

**Deliverables**: Can trigger a fine-tuning job from curl and monitor it

### Phase 4 — RL Training (Week 5–6)
**Goal**: GRPO-based reasoning improvement

- [ ] `r2d_hope/rl_trainer.py` — RLConfig + RLOO trainer (start simple)
- [ ] `r2d_hope/rewards.py` — Rule-based reward functions
- [ ] Integrate RLOO into fine-tuning API as `mode: "rl"`
- [ ] Validate GSM8K improvement vs SFT-only baseline
- [ ] Upgrade to GRPO once RLOO validated

**Deliverables**: GSM8K accuracy improves ≥ 5% absolute after RL post-training

### Phase 5 — Cloud Deployment (Week 7)
**Goal**: Production serving on HuggingFace Spaces

- [ ] `spaces/app.py` — HF Spaces wrapper
- [ ] `spaces/README.md` — Spaces configuration (hardware: t4-small)
- [ ] `scripts/deploy_to_hf_spaces.py` — Automated deploy script
- [ ] Vercel dashboard (Next.js) connecting to HF Spaces API
- [ ] End-to-end smoke test: UI → API → model → response

**Deliverables**: Live API endpoint accessible from browser

### Phase 6 — LLMOps Hardening (Week 8)
**Goal**: Production-grade reliability

- [ ] Quality gates in CI (PPL < 80, latency < 5s)
- [ ] Drift detection alerts → auto GitHub Issues
- [ ] Checkpoint versioning with metadata (PPL, benchmark scores, timestamp)
- [ ] API authentication (API key header)
- [ ] Rate limiting middleware
- [ ] `/v1/config` endpoint with validation
- [ ] Load testing (100 concurrent requests, measure p99 latency)

**Deliverables**: System meets LLMOps production standards

---

## 12. Consolidated Guideline

### 12.1 Directory Structure (Final)

```
R2D-HOPE-VLM/
├── r2d_hope/                    # Core model package
│   ├── config.py
│   ├── model.py
│   ├── core.py
│   ├── experts.py
│   ├── embeddings.py
│   ├── routing.py
│   ├── noise_scheduler.py
│   ├── block_sparsification.py
│   ├── data.py
│   ├── trainer.py
│   ├── distillation_data.py
│   ├── distillation_trainer.py
│   ├── rl_trainer.py            # NEW: Phase 4
│   ├── rewards.py               # NEW: Phase 4
│   ├── benchmark.py             # NEW: Phase 2
│   ├── auto_adjuster.py         # NEW: Phase 2
│   └── lora.py                  # NEW: Phase 3
│
├── backend/                     # FastAPI server
│   ├── main.py
│   ├── inference.py
│   ├── training.py
│   ├── schemas.py               # Pydantic models
│   ├── Dockerfile
│   └── requirements-api.txt
│
├── spaces/                      # HuggingFace Spaces
│   ├── app.py
│   └── README.md
│
├── frontend/                    # Vercel dashboard (Next.js)
│   ├── pages/
│   ├── components/
│   └── package.json
│
├── tests/                       # Test suite
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── conftest.py
│
├── scripts/                     # Automation
│   ├── run_benchmarks.py
│   ├── check_drift.py
│   ├── deploy_to_hf_spaces.py
│   └── post_benchmark_comment.py
│
├── benchmarks/
│   └── baseline.json            # Committed baseline scores
│
├── .github/
│   └── workflows/
│       ├── test.yml
│       ├── deploy.yml
│       └── benchmark.yml
│
├── TRAIN_COLAB.ipynb
├── DISTILL_COLAB.ipynb
├── BENCHMARK_COLAB.ipynb
├── requirements.txt
└── LLMOPS_MASTER_PLAN.md        # This document
```

### 12.2 Decision Reference Card

| Question | Answer |
|---|---|
| Can I deploy the model on Vercel? | No for inference. Yes for UI/gateway. |
| What RL method to use? | Start with RLOO → upgrade to GRPO |
| LoRA or full fine-tuning? | Full FT (model is only 13M params) |
| How many DDIM steps during RL rollout? | 10 (speed) vs 50 (quality) |
| Free inference hosting? | HuggingFace Spaces (T4, free) |
| Zero-cost CI/CD? | GitHub Actions (2 000 min/month free) |
| How to evaluate a diffusion LM? | Noise-loss scoring for MC, generate+parse for open-ended |
| When to trigger auto-adjustment? | Every eval run; escalate if no improvement after 3× patience |
| Minimum benchmark to pass deployment? | PPL < 80, any accuracy > random baseline |

### 12.3 Critical Invariants (Never Break These)

1. **`num_workers=0`** for all `IterableDataset` + HuggingFace streaming loaders
2. **Restart Colab runtime** after every `git pull` — Python import cache does not clear automatically
3. **Push before testing in Colab** — Colab clones from GitHub, not from local disk
4. **`betas` must be passed through** `build_optimizer` — do not re-hardcode
5. **`split="validation"`** for eval loaders — never evaluate on training data
6. **Regex patterns must compile at import time** — test with `python -c "import r2d_hope"` before pushing
7. **ReZero `alpha` needs 10× LR** — do not group it with regular params in optimizer
8. **DDIM + recurrent state must persist** across denoising steps — do not zero state between steps

---

*Document generated: 2026-03-07*
*Based on full codebase audit of R2D-HOPE-VLM @ commit 641cb83*
