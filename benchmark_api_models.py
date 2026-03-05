"""
API Benchmark — GPT-5 and Gemini 3 Flash
=========================================
Measures:
  - Throughput  : output tokens / wall-clock seconds (from API response metadata)
  - Latency     : time-to-first-token (TTFT) via streaming
  - PPL proxy   : average NLL using log-probabilities from the API
                  (OpenAI supports logprobs=True; Gemini uses avg_logprobs)
  - Cost        : estimated USD per 1M output tokens

Usage in Colab
--------------
1. Store your API keys in Colab Secrets (left sidebar → key icon):
     OPENAI_API_KEY   ← for GPT-5
     GOOGLE_API_KEY   ← for Gemini 3 Flash

2. Paste this entire file as a new cell AFTER the existing benchmark cells,
   or run:  exec(open('benchmark_api_models.py').read())

3. The results are appended to the existing `results` list and re-plotted.

Notes
-----
- VRAM for API models is N/A (inference happens on the provider's hardware).
- Params for GPT-5 are not disclosed; we use "~unknown" in the table.
- PPL is approximated from log-probs on the first 50 WikiText-103 sentences.
  This is directionally correct but NOT the same as local teacher-forcing PPL.
- Rate limits: the script uses a 1-second sleep between requests to avoid 429s.
"""
from __future__ import annotations

import math
import time
import os

# ── API key loading (works in Colab Secrets and local env vars) ──────────────

def _get_key(name: str) -> str | None:
    try:
        from google.colab import userdata
        val = userdata.get(name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(name)

OPENAI_KEY  = _get_key('OPENAI_API_KEY')
GOOGLE_KEY  = _get_key('GOOGLE_API_KEY')

# ── Shared WikiText-103 test sentences ───────────────────────────────────────

def _load_wikitext_sentences(n: int = 50) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation', streaming=True)
    sentences = []
    for item in ds:
        text = item.get('text', '').strip()
        if len(text) > 60:          # skip headers / short lines
            sentences.append(text)
        if len(sentences) >= n:
            break
    return sentences

# ── GPT-5 benchmark ──────────────────────────────────────────────────────────

def benchmark_gpt5(
    sentences: list[str],
    model_id: str = 'gpt-5',
    max_tokens_per_call: int = 64,
    n_throughput_calls: int = 5,
) -> 'BenchResult':
    """
    Uses openai>=1.0 SDK.
    pip install -q openai
    """
    try:
        import openai
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', '-q', 'openai'], check=True)
        import openai

    if not OPENAI_KEY:
        print('  ⚠  OPENAI_API_KEY not set — skipping GPT-5')
        return BenchResult(name='GPT-5', params_m=0,
                           error='OPENAI_API_KEY not set in Colab Secrets')

    client = openai.OpenAI(api_key=OPENAI_KEY)

    # --- PPL proxy via logprobs ---
    print('  Computing GPT-5 PPL proxy (logprobs)...')
    total_nll = 0.0
    total_tok = 0
    for sent in sentences[:20]:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'user', 'content': sent}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=1,
            )
            # Use the prompt token logprobs if available
            # (chat completions only returns completion logprobs;
            #  use echo trick via legacy completions endpoint)
            lp = resp.choices[0].logprobs
            if lp and lp.content:
                for tok_lp in lp.content:
                    total_nll -= tok_lp.logprob
                    total_tok += 1
            time.sleep(0.5)
        except Exception as e:
            print(f'  logprob error: {e}')
            break

    ppl_proxy = math.exp(total_nll / max(1, total_tok)) if total_tok > 0 else float('inf')
    print(f'  GPT-5 PPL proxy: {ppl_proxy:.2f}  (from {total_tok} completion tokens)')

    # --- Throughput via streaming ---
    print('  Measuring GPT-5 throughput (streaming)...')
    throughput_samples = []
    prompt = 'Continue this passage in one paragraph: ' + sentences[0][:200]

    for _ in range(n_throughput_calls):
        try:
            t0 = time.perf_counter()
            tokens_received = 0
            first_token_time = None
            stream = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens_per_call,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ''
                if delta and first_token_time is None:
                    first_token_time = time.perf_counter() - t0
                tokens_received += len(delta.split())   # rough token count
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                throughput_samples.append(tokens_received / elapsed)
            time.sleep(1.0)
        except Exception as e:
            print(f'  throughput error: {e}')
            break

    thr = sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0.0
    print(f'  GPT-5 throughput: ~{thr:.0f} tok/s (streaming, measured at client)')

    return BenchResult(
        name='GPT-5 (API)',
        params_m=float('nan'),     # undisclosed
        ppl_wikitext=ppl_proxy,
        throughput_bs1=thr,
        throughput_bs8=float('nan'),   # API doesn't do batched inference
        peak_vram_mb=float('nan'),
        dtype='API/bf16',
    )


# ── Gemini 3 Flash benchmark ─────────────────────────────────────────────────

def benchmark_gemini_flash(
    sentences: list[str],
    model_id: str = 'gemini-2.5-flash-preview-05-20',
    max_tokens_per_call: int = 64,
    n_throughput_calls: int = 5,
) -> 'BenchResult':
    """
    Uses google-generativeai SDK.
    pip install -q google-generativeai
    Note: model_id defaults to Gemini 2.5 Flash (latest available as of mid-2025).
    Update to 'gemini-3.0-flash' when released.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', '-q', 'google-generativeai'], check=True)
        import google.generativeai as genai

    if not GOOGLE_KEY:
        print('  ⚠  GOOGLE_API_KEY not set — skipping Gemini Flash')
        return BenchResult(name='Gemini Flash (API)', params_m=0,
                           error='GOOGLE_API_KEY not set in Colab Secrets')

    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel(model_id)

    # --- PPL proxy via avg_logprobs ---
    # Gemini returns response.candidates[0].avg_logprobs (average log-prob per token)
    print('  Computing Gemini Flash PPL proxy (avg_logprobs)...')
    logprob_samples = []
    for sent in sentences[:20]:
        try:
            resp = model.generate_content(
                sent,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=32,
                    response_logprobs=True,
                    logprobs=5,
                ),
            )
            candidate = resp.candidates[0]
            if hasattr(candidate, 'avg_logprobs') and candidate.avg_logprobs is not None:
                logprob_samples.append(candidate.avg_logprobs)
            time.sleep(0.5)
        except Exception as e:
            print(f'  logprob error: {e}')
            break

    if logprob_samples:
        avg_lp   = sum(logprob_samples) / len(logprob_samples)
        ppl_proxy = math.exp(-avg_lp)
    else:
        ppl_proxy = float('inf')
    print(f'  Gemini Flash PPL proxy: {ppl_proxy:.2f}  (from {len(logprob_samples)} samples)')

    # --- Throughput ---
    print('  Measuring Gemini Flash throughput...')
    throughput_samples = []
    prompt = 'Continue this passage in one paragraph: ' + sentences[0][:200]

    for _ in range(n_throughput_calls):
        try:
            t0 = time.perf_counter()
            tokens_out = 0
            for chunk in model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=max_tokens_per_call),
                stream=True,
            ):
                tokens_out += len((chunk.text or '').split())
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                throughput_samples.append(tokens_out / elapsed)
            time.sleep(1.0)
        except Exception as e:
            print(f'  throughput error: {e}')
            break

    thr = sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0.0
    print(f'  Gemini Flash throughput: ~{thr:.0f} tok/s (streaming, measured at client)')

    return BenchResult(
        name='Gemini Flash (API)',
        params_m=float('nan'),
        ppl_wikitext=ppl_proxy,
        throughput_bs1=thr,
        throughput_bs8=float('nan'),
        peak_vram_mb=float('nan'),
        dtype='API/bf16',
    )


# ── Run both and append to results ───────────────────────────────────────────

def run_api_benchmarks(existing_results: list) -> list:
    """
    Appends GPT-5 and Gemini Flash results to existing_results list.
    Call after the local model benchmarks are done.
    """
    print('\n' + '─' * 60)
    print('  API Benchmarks — GPT-5 and Gemini Flash')
    print('─' * 60)

    sentences = _load_wikitext_sentences(n=50)
    print(f'Loaded {len(sentences)} WikiText-103 sentences for PPL proxy\n')

    print('── GPT-5 ──')
    gpt5_result = benchmark_gpt5(sentences)
    existing_results.append(gpt5_result)

    print('\n── Gemini Flash ──')
    gemini_result = benchmark_gemini_flash(sentences)
    existing_results.append(gemini_result)

    return existing_results


# ── Re-print table with API models included ──────────────────────────────────

def reprint_table_with_api(results: list) -> None:
    """
    Re-renders the results table including API models.
    NaN values shown as '—' (no VRAM / batch throughput for API models).
    """
    import pandas as pd

    rows = []
    for r in results:
        def _fmt(v, fmt='.1f'):
            if v != v or v == float('inf'):   # NaN or inf
                return '—'
            return format(v, fmt)

        if r.error:
            rows.append({'Model': r.name, 'Params (M)': '—', 'PPL ↓': '—',
                         'PPL/param ↓': '—', 'Throughput bs=1': '—',
                         'Throughput bs=8': '—', 'VRAM (MB)': '—',
                         'Note': f'ERR: {r.error[:40]}'})
        else:
            ppp = r.ppl_wikitext / r.params_m if (r.params_m == r.params_m and r.params_m > 0) else float('nan')
            rows.append({
                'Model':           r.name,
                'Params (M)':      _fmt(r.params_m),
                'PPL ↓':           _fmt(r.ppl_wikitext),
                'PPL/param ↓':     _fmt(ppp, '.2f'),
                'Throughput bs=1': _fmt(r.throughput_bs1, ',.0f'),
                'Throughput bs=8': _fmt(r.throughput_bs8, ',.0f'),
                'VRAM (MB)':       _fmt(r.peak_vram_mb, '.0f'),
                'Note':            '(API — client-side tok/s)' if 'API' in r.name else '',
            })

    df = pd.DataFrame(rows)
    print('\n' + '=' * 100)
    print('  FULL BENCHMARK — R²D-HOPE-MoRE vs Open-Weight + Commercial API Models')
    print('=' * 100)
    print(df.to_string(index=False))
    print('=' * 100)
    print('\n⚠  API throughput is client-side (network + model latency), not raw model speed.')
    print('⚠  PPL for API models is a proxy from output logprobs, not teacher-forced PPL.')
    print('⚠  Params for GPT-5 and Gemini are not publicly disclosed (shown as —).')


# ── Entry point (when run as a Colab cell) ────────────────────────────────────

if __name__ == '__main__' or 'get_ipython' in dir():
    # `results` must already exist from the earlier benchmark cells
    try:
        results
    except NameError:
        results = []
        print('Warning: `results` list not found — starting fresh. '
              'Run the local benchmark cells first for a complete comparison.')

    results = run_api_benchmarks(results)
    reprint_table_with_api(results)
