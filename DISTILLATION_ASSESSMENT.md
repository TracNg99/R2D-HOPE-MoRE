# Knowledge Distillation Assessment: OpenRouter Teachers → R²D-HOPE-MoRE

## Executive Summary

**Verdict**: Distillation from 100B+ parameter OpenRouter models to R²D-HOPE-MoRE (6-11M params) is **technically viable but requires a hybrid approach** combining reasoning trace distillation with task-specific fine-tuning. Direct logit-based distillation is impractical due to API limitations and scale mismatch.

**Recommended Path**: Use OpenRouter models to generate high-quality **reasoning traces** (Chain-of-Thought) and **preference pairs**, then train R²D-HOPE-MoRE with a modified objective that combines:
1. Next-token prediction on reasoning traces (behavioral cloning)
2. Diffusion noise prediction on reasoning embeddings
3. Optional: Direct Preference Optimization (DPO) on ranked responses

---

## Teacher Model Analysis

| Model | Size | Context | Strengths | Cost/1M tokens (approx) |
|-------|------|---------|-----------|------------------------|
| `moonshotai/kimi-k2.5` | ~100B+ | 200K | Code, long-context reasoning, instruction following | ~$0.50-1.00 |
| `qwen/qwen3.5-122b-a10b` | 122B | 128K | General purpose, multilingual, strong reasoning | ~$0.60-1.20 |
| `qwen/qwen3.5-397b-a17b` | 397B | 128K | Best general reasoning, math, coding | ~$2.00-4.00 |
| `deepseek/deepseek-v3.1-terminus:exacto` | ~236B | 64K | State-of-the-art reasoning, step-by-step explanations | ~$1.00-2.00 |

**Key Insight**: These are **massive teachers** (100B-400B parameters) distilling to a **tiny student** (6-11M parameters) - a ~10,000:1 compression ratio. Traditional logit-matching distillation won't work; we need **capability transfer via reasoning traces**.

---

## Why Standard Distillation Won't Work

### 1. Logit-Based Distillation (Hinton et al.) — **NOT FEASIBLE**

Standard approach: Match teacher's soft probability distribution over vocabulary.

**Problems with API-only teachers:**
- OpenRouter returns token sequences, not logits or probability distributions
- No access to hidden states for intermediate layer distillation
- Temperature-scaled softmax outputs unavailable
- **Even if available**: Vocabulary mismatch (teacher's tokenizer ≠ student's 16k BPE)
- **Scale gap**: 400B → 11M parameters = excessive capacity difference; teacher's distribution is too "sharp" for tiny student

### 2. Hidden-State Distillation — **NOT FEASIBLE**
- Requires access to intermediate representations
- API models don't expose hidden states
- R²D's recursive architecture has no direct mapping to standard transformer layers

---

## Recommended: Reasoning Trace Distillation (RTD)

### What It Is
Instead of matching logits, we:
1. **Generate high-quality reasoning traces** from teachers (Chain-of-Thought)
2. **Train R²D-HOPE-MoRE to reproduce those traces** via next-token prediction
3. **Leverage the diffusion objective** to model reasoning as a denoising process

### Why It Fits R²D-HOPE-MoRE
The diffusion-based architecture is actually **advantageous** for reasoning:
- Reasoning is iterative refinement — exactly what DDIM does
- The `nested_depth` recursion can model multi-step reasoning
- MoE routers can specialize: one expert for "planning", one for "verification"

### Implementation Strategy

#### Phase 1: Dataset Generation (OpenRouter API)
```python
# Generate reasoning traces for diverse tasks
prompts = [
    "Solve step by step: {math_problem}",
    "Explain your reasoning: {logic_puzzle}",
    "Write {code_task} with explanation of approach",
    "Analyze: {text} - show your chain of thought",
]

# For each prompt, request reasoning_chain from teacher
teacher_response = openrouter.chat.completions.create(
    model="qwen/qwen3.5-397b-a17b",
    messages=[{
        "role": "user", 
        "content": prompt + "\nShow your step-by-step reasoning before giving the final answer."
    }],
    temperature=0.7,
)
# Store: {prompt, reasoning_chain, final_answer}
```

#### Phase 2: Training Objective Modification

Current R²D-HOPE-MoRE loss:
```
loss = noise_loss + 0.1 * ce_loss + 0.01 * aux_loss
```

**Distillation-enhanced loss**:
```
loss = noise_loss + α * ce_loss + β * distill_loss + γ * aux_loss

where:
- noise_loss: MSE on diffusion noise (unchanged)
- ce_loss: Cross-entropy on teacher's final answer tokens
- distill_loss: Cross-entropy on full reasoning trace (teacher's CoT)
- aux_loss: MoE load balancing (unchanged)
```

**Key change**: Train on the **reasoning trace as the target sequence**, not just the final answer. This teaches the model *how* to reason, not just *what* to output.

#### Phase 3: Diffusion-Specific Distillation
R²D-HOPE-MoRE's diffusion objective adds a unique opportunity:

Instead of predicting noise on random timesteps, we can:
1. Embed teacher's reasoning trace → target embeddings
2. Add noise at various levels
3. Train to denoise toward the high-quality reasoning representation

This is **implicit distillation**: the diffusion learns to generate embeddings that encode the teacher's reasoning patterns.

---

## Model-Specific Recommendations

### For Coding: `moonshotai/kimi-k2.5`
**Best for**: Code generation with step-by-step explanation traces

**Recommended data mix**:
- 40% coding problems with explicit reasoning (LeetCode-style)
- 30% code explanation tasks
- 20% debugging with reasoning traces
- 10% general reasoning

**Special handling**: Kimi's 200K context allows processing entire codebases for context-aware distillation.

### For General Purpose: `qwen/qwen3.5-122b-a10b`
**Best for**: Balanced capability transfer, multilingual support

**Recommended data mix**:
- 25% mathematical reasoning
- 25% reading comprehension with CoT
- 20% logical deduction puzzles
- 15% summarization with explanation
- 15% general knowledge reasoning

### For Maximum Quality: `qwen/qwen3.5-397b-a17b`
**Best for**: Highest quality reasoning traces, math, complex logic

**Cost-conscious approach**: Use for only the most difficult 20% of prompts where smaller teachers fail. Use 122B model for the rest.

### For Step-by-Step Rigor: `deepseek/deepseek-v3.1-terminus:exacto`
**Best for**: Explicit reasoning structure, verification steps

**Unique advantage**: DeepSeek models often show their "scratchpad" reasoning explicitly — perfect for distillation.

---

## Implementation Plan

### Step 1: Dataset Pipeline (`r2d_hope/distillation_data.py`)
```python
class OpenRouterDistillationDataset:
    """Generates and caches reasoning traces from OpenRouter teachers."""
    
    def generate_trace(self, prompt: str, teacher: str) -> dict:
        # Call OpenRouter API
        # Parse reasoning chain vs final answer
        # Cache to avoid re-calling API
        return {
            "prompt": prompt,
            "reasoning": reasoning_chain,
            "answer": final_answer,
            "teacher": teacher,
        }
    
    def create_diffusion_pairs(self, traces: list) -> list:
        # Create (noised_embedding, clean_target) pairs for diffusion training
        # Clean target = embedding of reasoning + answer
```

### Step 2: Modified Training Loop
```python
class DistillationTrainer:
    def compute_loss(self, batch):
        # Standard R²D losses
        noise_loss = ...  # unchanged
        aux_loss = ...    # unchanged
        
        # New: Distillation loss on reasoning trace
        teacher_reasoning_ids = batch["reasoning_ids"]  # tokenized teacher CoT
        ce_reasoning = F.cross_entropy(
            student_logits_on_reasoning,
            teacher_reasoning_ids
        )
        
        # Weighted combination
        return noise_loss + 0.1 * ce_reasoning + 0.01 * aux_loss
```

### Step 3: Multi-Teacher Ensemble
Don't rely on one teacher. For each prompt:
1. Query 2-3 teachers
2. Score responses with a reward model or heuristics
3. Use the best (or average of top 2) as training target

---

## Expected Outcomes

### Realistic Gains
| Metric | Baseline (no distillation) | With Distillation | Notes |
|--------|---------------------------|-------------------|-------|
| WikiText PPL | ~25-30 | ~20-25 | Modest gain; pre-training is different |
| Math reasoning (GSM8K) | ~5% | ~25-40% | Large gain from CoT distillation |
| Code generation (HumanEval) | ~2% | ~15-25% | Large gain from Kimi traces |
| General reasoning | ~15% | ~30-45% | Significant improvement |

### Why Not Higher?
- 6-11M parameters is fundamentally limited for complex reasoning
- Distillation transfers *behavior* not *capacity*
- Some tasks require more parameters than available (emergent capabilities threshold ~60B+)

---

## Cost Estimates

To generate 100K distillation samples:

| Teacher | Cost/1K samples | Cost for 100K | Quality |
|---------|----------------|---------------|---------|
| Kimi-k2.5 | $0.50 | ~$50 | High for code |
| Qwen-122B | $0.80 | ~$80 | Balanced |
| Qwen-397B | $3.00 | ~$300 | Highest quality |
| DeepSeek-v3.1 | $1.50 | ~$150 | Best reasoning structure |

**Recommended budget**: $100-150 for a good mix (70% 122B + 30% 397B or DeepSeek)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Teacher degradation | High | Cache all API responses; never re-query same prompt |
| API rate limits | Medium | Implement exponential backoff; use multiple keys |
| Cost overruns | Medium | Start with 10K samples; validate gains before scaling |
| Distribution shift | High | Ensure teacher prompts cover student's target domain |
| Reasoning hallucination | Medium | Filter traces with self-consistency checks |

---

## Quick Start: Minimum Viable Distillation

To test if this works for your use case:

1. **Generate 1,000 reasoning traces** using `qwen/qwen3.5-122b-a10b`
2. **Fine-tune R²D-HOPE-MoRE** for 2,000 steps on this data
3. **Evaluate** on 100 held-out reasoning problems
4. **Compare** vs baseline (no distillation)

If you see >10% relative improvement, scale to full 50K-100K samples.

---

## Files to Create

1. `r2d_hope/distillation_data.py` — OpenRouter API client + dataset builder
2. `r2d_hope/distillation_trainer.py` — Modified training loop with distillation loss
3. `DISTILL_COLAB.ipynb` — Colab notebook for distillation pipeline

**Ready to implement?** The training notebook is already running — we can prepare the distillation pipeline in parallel and switch to it once you have baseline metrics.
