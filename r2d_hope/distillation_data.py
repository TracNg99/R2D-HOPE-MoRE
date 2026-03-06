"""
OpenRouter Distillation Data Pipeline for R²D-HOPE-MoRE

Generates high-quality reasoning traces from large teacher models via OpenRouter API,
then formats them for diffusion-based training on R²D-HOPE-MoRE.

Features:
  - Multi-teacher support (Kimi, Qwen, DeepSeek)
  - Response caching to avoid API re-calls
  - Self-consistency filtering (majority vote across multiple samples)
  - Tokenization-aware chunking for the student's 16k BPE tokenizer
  - Export to JSONL for streaming training

Usage:
    from r2d_hope.distillation_data import OpenRouterDistiller
    
    distiller = OpenRouterDistiller(api_key='sk-or-...')
    dataset = distiller.generate_dataset(
        prompts=[...],
        teachers=['qwen/qwen3.5-122b-a10b'],
        n_samples_per_prompt=3,  # for self-consistency voting
        output_path='distillation_data.jsonl'
    )
"""
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from collections import Counter

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

TEACHER_MODELS = {
    # Free DeepSeek R1 distill models - BEST VALUE
    "deepseek-r1-qwen-32b-free": "deepseek/deepseek-r1-distill-qwen-32b:free",
    "deepseek-r1-llama-70b-free": "deepseek/deepseek-r1-distill-llama-70b:free",
    
    # Very cheap DeepSeek R1 distill models
    "deepseek-r1-qwen-14b": "deepseek/deepseek-r1-distill-qwen-14b",  # $0.15/M
    "deepseek-r1-llama-8b": "deepseek/deepseek-r1-distill-llama-8b",    # $0.04/M
    "deepseek-r1-qwen-1.5b": "deepseek/deepseek-r1-distill-qwen-1.5b",  # $0.18/M
    
    # Original premium teachers (expensive)
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "qwen-122b": "qwen/qwen3.5-122b-a10b",
    "qwen-397b": "qwen/qwen3.5-397b-a17b",
    "deepseek-v3": "deepseek/deepseek-v3.1-terminus:exacto",
}

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """Single reasoning trace from a teacher model."""
    prompt: str
    reasoning: str          # Chain-of-Thought explanation
    answer: str           # Final answer
    teacher: str          # Model identifier
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    hash: str             # Content hash for deduplication
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> ReasoningTrace:
        return cls(**d)


@dataclass
class DistillationSample:
    """Final training sample after aggregation and tokenization."""
    prompt_ids: list[int]
    reasoning_ids: list[int]   # Full CoT + answer, what student learns to generate
    prompt_text: str
    reasoning_text: str
    teacher_votes: dict      # {teacher_name: count} for ensemble samples
    
    def to_dict(self) -> dict:
        return {
            "prompt_ids": self.prompt_ids,
            "reasoning_ids": self.reasoning_ids,
            "prompt_text": self.prompt_text,
            "reasoning_text": self.reasoning_text,
            "teacher_votes": self.teacher_votes,
        }


# ---------------------------------------------------------------------------
# OpenRouter Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Thin wrapper around OpenRouter API with retry logic."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass to constructor.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
    
    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retries: int = 3,
    ) -> dict:
        """
        Returns: {
            "content": str,
            "tokens_prompt": int,
            "tokens_completion": int,
            "latency_ms": float,
        }
        """
        url = f"{OPENROUTER_BASE}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enforce_distillable_text": True,  # Required for compliance with model licenses
        }
        
        for attempt in range(retries):
            try:
                t0 = time.perf_counter()
                resp = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                latency = (time.perf_counter() - t0) * 1000
                
                if resp.status_code == 429:  # Rate limit
                    wait = 2 ** attempt
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                    
                resp.raise_for_status()
                data = resp.json()
                
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                return {
                    "content": choice["message"]["content"],
                    "tokens_prompt": usage.get("prompt_tokens", 0),
                    "tokens_completion": usage.get("completion_tokens", 0),
                    "latency_ms": latency,
                }
                
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1)
        
        raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# Reasoning Parser
# ---------------------------------------------------------------------------

REASONING_PATTERNS = [
    # Pattern 1: "Let me think... <thinking>reasoning</thinking> Answer: X"
    re.compile(r"(?:<thinking>|Let me think|Thinking:|Reasoning:)(.+?)(?:</thinking>|Answer:|The answer is|\*\*Answer\*\*:)", re.DOTALL | re.IGNORECASE),
    # Pattern 2: Step-by-step markers
    re.compile(r"(?:Step \d+[:.])?\s*(.+?)(?=(?:Step \d+[:.]|Therefore,|Thus,|So,|Answer:|The answer is)|$", re.DOTALL),
    # Pattern 3: Generic split on "Answer:" or similar
    re.compile(r"(.+?)(?:\n\n(?:Answer|Result|Final|Conclusion)[:\s]+)(.+)", re.DOTALL),
]

def parse_reasoning(response: str) -> tuple[str, str]:
    """
    Split response into reasoning chain and final answer.
    Returns: (reasoning, answer)
    """
    response = response.strip()
    
    # Try structured patterns first
    for pattern in REASONING_PATTERNS:
        match = pattern.search(response)
        if match:
            if len(match.groups()) == 2:
                reasoning = match.group(1).strip()
                answer = match.group(2).strip()
            else:
                reasoning = match.group(1).strip()
                # Answer is the remainder after the match
                answer = response[match.end():].strip() or reasoning.split('\n')[-1]
            return reasoning, answer
    
    # Fallback: no clear separator — treat everything as reasoning, last line as answer
    lines = response.split('\n')
    if len(lines) > 1:
        reasoning = '\n'.join(lines[:-1]).strip()
        answer = lines[-1].strip()
    else:
        reasoning = response
        answer = response
    
    return reasoning, answer


# ---------------------------------------------------------------------------
# Main Distiller
# ---------------------------------------------------------------------------

class OpenRouterDistiller:
    """
    Generates reasoning traces from OpenRouter teachers for distillation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./distill_cache",
    ):
        self.client = OpenRouterClient(api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, prompt: str, teacher: str) -> Path:
        """Deterministic cache file path."""
        key = hashlib.sha256(f"{prompt}:{teacher}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{teacher.replace('/', '_')}_{key}.json"
    
    def _load_cached(self, prompt: str, teacher: str) -> Optional[ReasoningTrace]:
        path = self._cache_path(prompt, teacher)
        if path.exists():
            with open(path, 'r') as f:
                return ReasoningTrace.from_dict(json.load(f))
        return None
    
    def _save_cached(self, trace: ReasoningTrace):
        path = self._cache_path(trace.prompt, trace.teacher)
        with open(path, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2)
    
    def generate_trace(
        self,
        prompt: str,
        teacher: str,
        use_cache: bool = True,
        system_prompt: Optional[str] = None,
    ) -> ReasoningTrace:
        """
        Generate a single reasoning trace from a teacher model.
        """
        if use_cache:
            cached = self._load_cached(prompt, teacher)
            if cached:
                return cached
        
        # Default system prompt to encourage reasoning
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that thinks step by step. "
                "Explain your reasoning clearly before giving the final answer. "
                "Use the format: 'Reasoning: [your step-by-step thinking] Answer: [final answer]'"
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            result = self.client.chat_completion(
                model=teacher,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
            )
        except Exception as e:
            print(f"  API error for {teacher}: {e}")
            raise
        
        reasoning, answer = parse_reasoning(result["content"])
        
        trace = ReasoningTrace(
            prompt=prompt,
            reasoning=reasoning,
            answer=answer,
            teacher=teacher,
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            latency_ms=result["latency_ms"],
            hash=hashlib.sha256(result["content"].encode()).hexdigest()[:16],
        )
        
        if use_cache:
            self._save_cached(trace)
        
        return trace
    
    def generate_with_consistency(
        self,
        prompt: str,
        teacher: str,
        n_samples: int = 3,
        temperature: float = 0.7,
    ) -> ReasoningTrace:
        """
        Generate multiple samples and pick the most consistent answer (self-consistency).
        Returns the trace with the majority-voted answer.
        """
        traces = []
        answers = []
        
        for i in range(n_samples):
            # Vary temperature slightly for diversity
            temp = temperature + (0.1 * (i - n_samples // 2))
            trace = self.generate_trace(prompt, teacher, use_cache=False)  # fresh samples
            traces.append(trace)
            # Normalize answer for voting (lowercase, strip punctuation)
            norm = re.sub(r'[^\w\s]', '', trace.answer.lower()).strip()
            answers.append(norm)
        
        # Majority vote
        vote_counts = Counter(answers)
        winner = vote_counts.most_common(1)[0][0]
        
        # Find best trace matching winner (first match)
        for t, a in zip(traces, answers):
            if a == winner:
                return t
        
        return traces[0]  # fallback
    
    def generate_dataset(
        self,
        prompts: list[str],
        teachers: list[str],
        output_path: str,
        n_samples_per_prompt: int = 1,
        use_consistency: bool = False,
        tokenizer = None,  # PreTrainedTokenizerFast
        max_seq_len: int = 512,
    ) -> list[DistillationSample]:
        """
        Generate full distillation dataset with tokenization.
        """
        samples: list[DistillationSample] = []
        
        print(f"Generating {len(prompts)} prompts × {len(teachers)} teachers = {len(prompts)*len(teachers)} traces")
        
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
            
            teacher_votes = {}
            best_trace = None
            
            for teacher in teachers:
                try:
                    if use_consistency and n_samples_per_prompt > 1:
                        trace = self.generate_with_consistency(prompt, teacher, n_samples_per_prompt)
                    else:
                        trace = self.generate_trace(prompt, teacher)
                    
                    teacher_votes[teacher] = 1
                    
                    # Use first teacher's response as the target
                    # (Could ensemble across teachers with more complex logic)
                    if best_trace is None:
                        best_trace = trace
                        
                except Exception as e:
                    print(f"  Failed for {teacher}: {e}")
                    continue
            
            if best_trace is None:
                print(f"  ⚠ No valid traces for this prompt")
                continue
            
            # Tokenize for student training
            if tokenizer:
                prompt_ids = tokenizer.encode(best_trace.prompt, add_special_tokens=False)
                reasoning_ids = tokenizer.encode(
                    best_trace.reasoning + " " + best_trace.answer,
                    add_special_tokens=False,
                    max_length=max_seq_len,
                    truncation=True,
                )
            else:
                prompt_ids = []
                reasoning_ids = []
            
            sample = DistillationSample(
                prompt_ids=prompt_ids,
                reasoning_ids=reasoning_ids,
                prompt_text=best_trace.prompt,
                reasoning_text=best_trace.reasoning + "\nAnswer: " + best_trace.answer,
                teacher_votes=teacher_votes,
            )
            samples.append(sample)
            
            # Periodic save
            if (i + 1) % 10 == 0:
                self._save_jsonl(samples, output_path)
                print(f"  Saved {len(samples)} samples to {output_path}")
        
        # Final save
        self._save_jsonl(samples, output_path)
        print(f"\n✓ Dataset complete: {len(samples)} samples → {output_path}")
        return samples
    
    def _save_jsonl(self, samples: list[DistillationSample], path: str):
        """Append samples to JSONL file."""
        with open(path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + '\n')


# ---------------------------------------------------------------------------
# Prompt templates for various tasks
# ---------------------------------------------------------------------------

REASONING_PROMPT_TEMPLATES = {
    "math": [
        "Solve step by step: {problem}",
        "Show your work and solve: {problem}",
        "Calculate with explanation: {problem}",
    ],
    "code": [
        "Write Python code for: {task}\nExplain your approach first.",
        "Solve this coding problem with reasoning: {task}",
        "Design and implement: {task}\nWalk through your logic.",
    ],
    "logic": [
        "Think through this logic puzzle: {puzzle}",
        "Analyze step by step: {puzzle}",
        "Explain your reasoning for: {puzzle}",
    ],
    "reading": [
        "Read this text and explain your understanding:\n{text}",
        "Summarize and explain the key points:\n{text}",
        "Analyze this passage with reasoning:\n{text}",
    ],
    "qa": [
        "Question: {question}\nThink step by step before answering.",
        "Answer with detailed reasoning: {question}",
        "Explain your thought process for: {question}",
    ],
}

def get_reasoning_prompts(domain: str, items: list[str]) -> list[str]:
    """Fill templates with actual content."""
    templates = REASONING_PROMPT_TEMPLATES.get(domain, REASONING_PROMPT_TEMPLATES["qa"])
    prompts = []
    for item in items:
        # Cycle through templates
        tmpl = templates[len(prompts) % len(templates)]
        prompts.append(tmpl.format(**{domain: item} if domain in tmpl else {"question": item, "problem": item, "task": item, "puzzle": item, "text": item}))
    return prompts


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: Generate small dataset
    distiller = OpenRouterDistiller()
    
    prompts = [
        "What is 27 × 43? Show your calculation.",
        "Explain why the sky is blue.",
        "If a train travels 60 km/h for 2.5 hours, how far does it go?",
    ]
    
    samples = distiller.generate_dataset(
        prompts=prompts,
        teachers=[TEACHER_MODELS["deepseek-r1-qwen-32b-free"]],  # FREE - best value
        output_path="demo_distill.jsonl",
        n_samples_per_prompt=1,
    )
    
    print(f"\nGenerated {len(samples)} distillation samples")
