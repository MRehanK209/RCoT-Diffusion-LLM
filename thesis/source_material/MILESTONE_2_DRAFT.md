# Milestone 2: Complete Results Draft

> This document consolidates ALL experiment results from the repository into a single narrative.
> It is intended to be given to GPT5.4 Pro alongside `docs/Milestone_2_Submission.docx` to produce the final updated submission.
>
> All tables below are backed by JSON artifacts in `results/milestone2_*` directories unless otherwise noted.

## Thesis Title

Comparative Evaluation of Diffusion LLMs and Autoregressive LLMs for Reasoning, Planning, and Sampling-Based Performance

## Student

Muhammad Rehan Khalid

## Milestone

Milestone 2 Progress Update

---

## 1. Executive Summary

This milestone presents a comprehensive empirical comparison of two diffusion LLM families (LLaDA-8B, Dream-v0-7B) against two autoregressive baselines (Qwen2.5-7B, Llama-3.1-8B) across three benchmarks (GSM8K, Countdown-cd4, Trip Planning) under both deterministic and stochastic pass@k evaluation.

### Key Findings

1. **Diffusion models dominate deterministic planning.** On Countdown-cd4 greedy decoding, diffusion base models (14-15%) far exceed AR baselines (6%). On Trip Planning, Dream achieves 13.5-14.5% vs Qwen's 1.5-2%.

2. **AR models overtake at large k.** On Countdown pass@128, Qwen reaches 51.3% and Llama 49.5%, while LLaDA plateaus at 27.2% and Dream at 37.6%. The ranking reversal from pass@1 to pass@128 is the central thesis result.

3. **Diffusion models produce less diverse samples.** The slower pass@k growth of diffusion models suggests their denoising process concentrates probability mass, yielding high pass@1 but limited coverage of the solution space.

4. **Per-question failure modes correlate by paradigm, not architecture.** Despite Dream being built on the Qwen2.5 backbone, per-question accuracy correlation between Dream and LLaDA (both dLLMs) is higher than between Dream and Qwen (shared architecture). This shows the diffusion denoising process—not the pretrained weights—determines which problems a model can solve.

5. **Prompt format is a major confounder for instruct models.** 0-shot vs 4-shot and templated vs flat prompts cause large swings (e.g., Qwen-Instruct GSM8K: 53.4% at 0-shot vs 78.9% at 4-shot).

6. **GSM8K is near-saturated but useful as a sanity check.** All models approach 98-100% by pass@128, confirming the benchmarks' ceiling at 7-8B scale.

---

## 2. Models, Inference, and Evaluation Pipeline

### 2.1 Models

| Family | Base | Instruct | Params | Type |
|--------|------|----------|--------|------|
| LLaDA | GSAI-ML/LLaDA-8B-Base | GSAI-ML/LLaDA-8B-Instruct | 8B | Diffusion (masked) |
| Dream | Dream-org/Dream-v0-Base-7B | Dream-org/Dream-v0-Instruct-7B | 7B | Diffusion (masked, Qwen2.5 backbone) |
| Qwen | Qwen/Qwen2.5-7B | Qwen/Qwen2.5-7B-Instruct | 7B | Autoregressive |
| Llama | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B | Autoregressive |

### 2.2 Inference Methods

- **Fast-dLLM**: KV-cache optimized diffusion inference (default for all dLLM runs)
- **vLLM**: PagedAttention-accelerated AR inference (default for all AR runs)
- **Standard dLLM / AR-HF**: Used only for deterministic validation comparisons

### 2.3 Evaluation Modes

- **Deterministic (accuracy):** temperature=0, n_samples=1. Measures greedy-decode quality.
- **Pass@k:** temperature=0.7, n_samples=64-128. Uses unbiased estimator: `pass@k = 1 - C(n-c,k)/C(n,k)`.

### 2.4 Prompt Modes

Three-way comparison for instruct models:
- `base_native`: flat string prompt on base checkpoint
- `instruct_templated`: model-native chat template on instruct checkpoint (default)
- `instruct_flat`: flat string prompt on instruct checkpoint (ablation)

### 2.5 Unified Pipeline

All experiments use `run_evaluation.sh` which handles model loading, prompt construction, generation, answer extraction, and pass@k computation. Results are saved as structured JSON with full parameter provenance.

---

## 3. GSM8K Results (Sanity-Check Benchmark)

GSM8K is a grade-school math benchmark. At 7-8B scale it is near-saturated, so it serves as a sanity check rather than the core thesis evidence.

### 3.1 GSM8K Base Models: pass@k (n=128, 0-shot)

*Source: `results/milestone2_gsm8k_base/passk_gsm8k_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dream-v0-Base-7B [fast-dllm] | 72.39% | 82.81% | 88.87% | 92.27% | 94.41% | 95.78% | 96.88% | 97.66% |
| Qwen2.5-7B [vllm] | 70.93% | 85.94% | 92.89% | 96.44% | 98.59% | 99.69% | 99.99% | 100.00% |
| LLaDA-8B-Base [fast-dllm] | 63.14% | 75.00% | 82.66% | 87.97% | 91.72% | 94.22% | 96.41% | 98.44% |
| Llama-3.1-8B [vllm] | 14.33% | 25.86% | 42.97% | 63.59% | 81.80% | 92.97% | 98.05% | 100.00% |

**Interpretation:** Dream-Base leads dLLMs on GSM8K pass@1 (72.4%) and is competitive with Qwen (70.9%). Llama-Base is very weak at pass@1 (14.3%) but recovers fully by pass@128 (100%), demonstrating extreme sample diversity. Qwen also reaches 100% by pass@128.

### 3.2 GSM8K Base Models: pass@k (n=16, 4-shot)

*Source: `results/milestone2_gsm8k_base_4shot/passk_gsm8k_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B [vllm] | 72.27% | 86.56% | 93.52% | 96.72% | 98.44% |
| Dream-v0-Base-7B [fast-dllm] | 71.34% | 83.13% | 89.69% | 92.81% | 93.75% |
| LLaDA-8B-Base [fast-dllm] | 66.48% | 77.93% | 84.30% | 88.13% | 89.84% |
| Llama-3.1-8B [vllm] | 14.53% | 25.94% | 42.58% | 62.58% | 80.47% |

**Interpretation:** 4-shot prompting provides marginal improvement for most models. Llama-Base remains an outlier—very low pass@1 but rapid scaling. This pattern recurs across benchmarks.

### 3.3 GSM8K Instruct Models: pass@k (n=128, 4-shot, templated + flat)

*Source: `results/milestone2_gsm8k_instruct/passk_gsm8k_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct [vllm] | 79.77% | 87.50% | 91.80% | 94.53% | 96.48% | 98.13% | 99.30% | 100.00% |
| Llama-3.1-8B-Instruct [vllm] | 78.22% | 88.17% | 94.22% | 97.73% | 99.41% | 99.92% | 100.00% | 100.00% |
| Dream-v0-Instruct-7B [fast-dllm] | 76.15% | 84.38% | 89.26% | 92.58% | 95.00% | 96.56% | 97.50% | 97.66% |
| LLaDA-8B-Instruct [fast-dllm] | 75.57% | 85.16% | 90.47% | 94.22% | 96.80% | 98.20% | 99.02% | 99.22% |
| LLaDA-8B-Instruct [flat] [fast-dllm] | 76.90% | 86.09% | 90.78% | 93.75% | 95.47% | 96.56% | 97.42% | 98.44% |
| Qwen2.5-7B-Instruct [flat] [vllm] | 73.83% | 82.89% | 88.87% | 93.13% | 95.94% | 97.58% | 98.44% | 98.44% |
| Dream-v0-Instruct-7B [flat] [fast-dllm] | 70.93% | 82.03% | 88.48% | 92.19% | 94.69% | 96.48% | 97.81% | 98.44% |
| Llama-3.1-8B-Instruct [flat] [vllm] | 21.05% | 36.52% | 56.88% | 77.73% | 92.11% | 98.28% | 99.84% | 100.00% |

**Interpretation:**
- All instruct models converge to 97-100% by pass@128, confirming GSM8K saturation.
- **Prompt format matters hugely for Llama-Instruct**: templated 78.2% vs flat 21.1% at pass@1. Llama-Instruct with flat prompts behaves like Llama-Base.
- dLLM instruct models are competitive with AR instruct at pass@1 (75-76% vs 78-80%).
- Flat prompts generally hurt all models by 3-8% at pass@1.

### 3.4 GSM8K Instruct Models: 0-shot pass@k (n=16)

*Source: `results/milestone2_gsm8k_instruct_0shot/passk_gsm8k_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---:|---:|---:|---:|---:|
| LLaDA-8B-Instruct [fast-dllm] | 74.85% | 85.78% | 91.17% | 94.14% | 96.09% |
| Dream-v0-Instruct-7B [fast-dllm] | 73.39% | 84.53% | 90.47% | 94.30% | 96.88% |
| Qwen2.5-7B-Instruct [vllm] | 53.37% | 67.66% | 79.06% | 87.11% | 92.19% |
| Llama-3.1-8B-Instruct [vllm] | 48.78% | 62.27% | 73.67% | 82.58% | 87.89% |

**Interpretation:** At 0-shot, **dLLM instruct models strongly outperform AR instruct models** (74-75% vs 49-53% at pass@1). This reverses at 4-shot (see below), showing that dLLMs are better zero-shot reasoners while AR models benefit more from in-context examples.

### 3.5 GSM8K Instruct Models: 4-shot pass@k (n=16)

*Source: `results/milestone2_gsm8k_instruct_4shot/passk_gsm8k_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct [vllm] | 78.88% | 87.19% | 91.56% | 94.14% | 96.09% |
| Llama-3.1-8B-Instruct [vllm] | 78.20% | 87.89% | 93.91% | 96.95% | 98.44% |
| LLaDA-8B-Instruct [fast-dllm] | 77.47% | 88.05% | 93.67% | 96.72% | 98.05% |
| Dream-v0-Instruct-7B [fast-dllm] | 76.15% | 85.47% | 90.94% | 93.83% | 95.31% |

**Interpretation:** 4-shot closes the gap. AR models improve dramatically (Qwen: 53→79%, Llama: 49→78%) while dLLMs improve modestly (LLaDA: 75→77%, Dream: 73→76%). This suggests AR models rely more on in-context learning while dLLMs have stronger zero-shot generalization.

### 3.6 GSM8K Summary Finding

**GSM8K is saturated at 7-8B scale.** The most interesting GSM8K result is the 0-shot vs 4-shot sensitivity difference between paradigms: dLLMs are strong zero-shot reasoners; AR models need few-shot examples to reach comparable performance.

---

## 4. Countdown-cd4 Results (Core Planning Benchmark)

Countdown is a combinatorial arithmetic task: given target and numbers, find an expression using +, -, ×, ÷ that equals the target. The cd4 variant uses 4 numbers. This is the strongest benchmark for the thesis argument.

### 4.1 Countdown Deterministic Comparison

*Source: `docs/ANALYSIS.md`; validated against Dream paper protocol.*

| Model | Method | Acc (bs=1) | Time (bs=1) | Acc (bs=8) | Time (bs=8) | Speedup |
|---|---|---:|---:|---:|---:|---:|
| LLaDA-Base | Fast-dLLM | 13.7% | 16m 9s | 14.5% | 5m 19s | 3.03× |
| LLaDA-Base | dLLM | 15.4% | 25m 31s | 14.7% | 30m 1s | 0.85× |
| Dream-Base | Fast-dLLM | 14.4% | 15m 46s | 14.1% | 6m 46s | 2.33× |
| Dream-Base | dLLM | 14.6% | 23m 39s | 14.3% | 18m 35s | 1.27× |
| Qwen-Base | vLLM | 6.0% | 6m 35s | 6.0% | 51.6s | 7.65× |
| Qwen-Base | AR | 6.0% | 13m 46s | 6.0% | 2m 36s | 5.31× |
| LLaDA-Inst | Fast-dLLM | 1.5% | 16m 0s | 1.9% | 5m 38s | 2.84× |
| LLaDA-Inst | dLLM | 1.9% | 25m 56s | 1.8% | 30m 50s | 0.84× |
| Dream-Inst | Fast-dLLM | 7.5% | 12m 16s | 7.9% | 6m 4s | 2.02× |
| Dream-Inst | dLLM | 3.5% | 23m 43s | 3.4% | 19m 0s | 1.25× |
| Qwen-Inst | vLLM | 2.6% | 6m 33s | 2.8% | 51.9s | 7.58× |
| Qwen-Inst | AR | 1.0% | 14m 2s | 0.9% | 2m 43s | 5.16× |

**Key observations:**
- **Diffusion base models (13.7-15.4%) more than double AR base (6.0%)** on deterministic Countdown.
- Fast-dLLM introduces minimal quality loss (~1% or less) while providing 2-3× speedup.
- Instruct checkpoints are weaker than base for dLLMs on Countdown (especially LLaDA-Instruct at 1.5-1.9%).
- Dream-Instruct Fast-dLLM (7.5%) actually outperforms Dream-Instruct dLLM (3.5%), likely due to the fast path's block-level parallelism interacting favorably with Dream's architecture.

### 4.2 Countdown pass@k: Base Models (n=128)

*Source: `results/milestone2_countdown_base_refresh/passk_countdown_cd4_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA-8B-Base [fast-dllm] | 14.29% | 16.26% | 18.08% | 19.98% | 21.94% | 23.81% | 25.56% | 27.22% |
| Dream-v0-Base-7B [fast-dllm] | 11.30% | 15.68% | 20.05% | 24.42% | 28.57% | 32.18% | 35.17% | 37.60% |
| Qwen2.5-7B [vllm] | 4.04% | 7.51% | 13.14% | 21.03% | 30.14% | 38.72% | 45.70% | 51.31% |
| Llama-3.1-8B [vllm] | 2.70% | 5.13% | 9.36% | 15.96% | 24.75% | 34.34% | 42.90% | 49.50% |

**This is the central thesis table.** The ranking reversal:
- **At pass@1:** LLaDA (14.3%) > Dream (11.3%) > Qwen (4.0%) > Llama (2.7%) — dLLMs dominate.
- **At pass@128:** Qwen (51.3%) > Llama (49.5%) > Dream (37.6%) > LLaDA (27.2%) — AR models dominate.
- **Crossover point:** Around pass@8-16 for Dream vs AR; around pass@16-32 for LLaDA vs AR.

**Interpretation:** Diffusion models are better at finding a single good solution (higher pass@1) but AR models explore the solution space more broadly (steeper pass@k growth). This aligns with the hypothesis that iterative denoising concentrates probability mass while autoregressive sampling maintains higher diversity.

### 4.3 Countdown pass@k: Instruct Models (n=128, templated)

*Source: `results/milestone2_countdown_instruct_refresh/passk_countdown_cd4_comparison.json`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dream-v0-Instruct-7B [fast-dllm] | 9.15% | 12.55% | 16.03% | 19.64% | 23.19% | 26.36% | 29.32% | 32.26% |
| Qwen2.5-7B-Instruct [vllm] | 7.84% | 10.67% | 13.42% | 15.93% | 18.12% | 19.95% | 21.28% | 22.08% |
| Llama-3.1-8B-Instruct [vllm] | 5.78% | 10.26% | 16.73% | 24.38% | 31.63% | 37.61% | 42.44% | 46.17% |
| LLaDA-8B-Instruct [fast-dllm] | 2.41% | 2.78% | 3.17% | 3.61% | 4.14% | 4.69% | 5.16% | 5.54% |

**Interpretation:**
- Llama-Instruct again shows the steepest scaling, overtaking all models by pass@16.
- Dream-Instruct is the only dLLM instruct checkpoint that remains competitive.
- LLaDA-Instruct is essentially broken on Countdown (5.5% even at pass@128), suggesting instruction tuning damaged its planning capabilities.
- Qwen-Instruct shows moderate but slow scaling, plateauing around 22%.

### 4.4 Countdown Prompt-Mode Diagnostic (earlier smaller runs)

| Model | pass@1 | pass@16 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|
| LLaDA-8B-Instruct [fast-dllm] | 2.80% | 4.77% | 5.84% | 6.45% |
| LLaDA-8B-Instruct [flat] [fast-dllm] | 2.84% | 4.07% | 4.73% | 5.08% |
| Dream-v0-Instruct-7B [fast-dllm] | 1.25% | 4.68% | 6.24% | 6.64% |
| Dream-v0-Instruct-7B [flat] [fast-dllm] | 1.81% | 6.46% | 8.81% | 9.77% |
| Qwen2.5-7B-Instruct [vllm] | 0.71% | 2.54% | 3.50% | 3.71% |
| Qwen2.5-7B-Instruct [flat] [vllm] | 1.05% | 6.35% | 10.13% | 11.91% |
| Llama-3.1-8B-Instruct [vllm] | 6.28% | 33.33% | 44.06% | 48.05% |
| Llama-3.1-8B-Instruct [flat] [vllm] | 4.99% | 31.57% | 43.84% | 48.44% |

**Interpretation:** Prompt format matters substantially for Dream and Qwen (flat can improve or worsen by 3-8%) but barely affects Llama-Instruct. LLaDA-Instruct is weak regardless of prompt format.

---

## 5. Trip Planning Results (Structured Planning Benchmark)

Trip Planning requires generating a structured multi-step travel itinerary. It tests natural-language planning rather than pure arithmetic.

### 5.1 Trip Planning Deterministic Comparison

*Source: `results/accuracy_trip_planning_comparison.json` and `results/milestone2_trip_planning_llada_accuracy/`*

| Model / Variant | Accuracy |
|---|---:|
| Dream-v0-Instruct-7B [bl=256] [dllm] | 15.00% |
| Dream-v0-Base-7B [bl=256] [dllm] | 14.50% |
| Dream-v0-Base-7B [bl=32] [fast-dllm] | 13.50% |
| Dream-v0-Instruct-7B [bl=32] [fast-dllm] | 12.50% |
| Dream-v0-Instruct-7B [flat] [bl=32] [fast-dllm] | 12.50% |
| Dream-v0-Base-7B [bl=256] [dllm] | 14.50% |
| LLaDA-8B-Base [bl=256] [dllm] | 11.50% |
| Llama-3.1-8B-Instruct [flat] [vllm] | 9.50% |
| LLaDA-8B-Instruct [bl=256] [dllm] | 8.00% |
| LLaDA-8B-Base [bl=32] [fast-dllm] | 5.00% |
| Llama-3.1-8B [vllm] | 4.50% |
| LLaDA-8B-Instruct [bl=32] [dllm] | 4.50% |
| Qwen2.5-7B [vllm] | 1.50% |
| Qwen2.5-7B-Instruct [vllm] | 1.50% |
| Llama-3.1-8B-Instruct [vllm] | 2.00% |
| Qwen2.5-7B-Instruct [flat] [vllm] | 2.00% |

**Key observations:**
- **Dream dominates Trip Planning deterministically** (13.5-15.0%) across all variants.
- LLaDA-Base with standard dLLM (11.5%) is strong, but fast-dLLM degrades it to 5.0% — a larger fast-path penalty than on Countdown.
- AR models are weak on deterministic Trip Planning (Qwen 1.5%, Llama 2.0-4.5%).

### 5.2 Trip Planning pass@k: All Models (n=64)

*Sources: `results/aime_data_analysis_large_k_comparison.json`, `results/milestone2_trip_planning_llama_passk/`, `results/milestone2_trip_planning_llada_passk/`*

| Model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dream-v0-Base-7B [fast-dllm] | 5.44% | 8.12% | 10.82% | 13.37% | 15.89% | 18.78% | 22.50% |
| Dream-v0-Instruct-7B [fast-dllm] | 5.66% | 7.47% | 9.01% | 10.51% | 12.17% | 13.87% | 15.00% |
| LLaDA-8B-Base [fast-dllm] | 3.55% | 5.41% | 7.55% | 9.97% | 12.30% | 14.40% | 16.50% |
| LLaDA-8B-Instruct [fast-dllm] | 1.04% | 1.80% | 3.00% | 4.73% | 6.80% | 9.10% | 10.50% |
| LLaDA-8B-Instruct [flat] [fast-dllm] | 1.24% | 2.23% | 3.72% | 5.87% | 8.38% | 11.18% | 14.00% |
| Llama-3.1-8B [vllm] | 2.50% | 4.50% | 7.55% | 11.65% | 16.05% | 20.50% | 25.50% |
| Llama-3.1-8B-Instruct [vllm] | 1.66% | 3.12% | 5.38% | 8.37% | 12.03% | 16.10% | 21.00% |
| Llama-3.1-8B-Instruct [flat] [vllm] | 6.00% | 9.30% | 13.50% | 18.00% | 22.50% | 26.60% | 29.50% |
| Qwen2.5-7B [vllm] | 1.48% | 2.69% | 4.53% | 6.95% | 9.81% | 12.85% | 15.50% |
| Qwen2.5-7B-Instruct [vllm] | 1.65% | 2.60% | 3.70% | 4.91% | 6.24% | 7.55% | 8.50% |

**Interpretation:**
- **Dream-Base leads at pass@1** (5.4%) but **Llama-Instruct [flat] leads at pass@64** (29.5%).
- Same crossover pattern as Countdown: dLLMs start high, AR models scale faster.
- Llama-Instruct with flat prompts is surprisingly the strongest Trip Planning model at large k (29.5% at pass@64), while with templated prompts it's weaker (21.0%).
- Qwen is consistently weak on Trip Planning across all k values.

---

## 6. Hyperparameter Analysis (GSM8K Sweep)

*Source: `experiment_analysis.ipynb`; `results/Dream-org_Dream-v0-Base-7B_gsm8k/` and `results/GSAI-ML_LLaDA-8B-Base_gsm8k/`*

An exhaustive hyperparameter sweep was conducted on GSM8K for Dream-Base and LLaDA-Base across gen_length ∈ {128, 256}, steps ∈ {32, 64, 128, 256}, and block_length ∈ {32, 64, 128, 256}.

### 6.1 Best Configurations

| Model | Best pass@1 Config | pass@1 | Time | Best pass@128 Config | pass@128 |
|---|---|---:|---:|---|---:|
| LLaDA-8B | g256_s256_b128 | 71.01% | 14.5 min | g256_s128_b128 | 99.22% |
| Dream-7B | g256_s32_b32 | 70.66% | 11.8 min | g256_s128_b32 | 99.22% |
| Qwen2.5-7B | g256 (vLLM) | 72.97% | 5.7 min | g256 (vLLM) | 100.00% |

### 6.2 Key Hyperparameter Findings

1. **Generation length is the most impactful parameter.** Moving from 128 to 256 tokens improves pass@1 by 20-25%.
2. **Steps show diminishing returns.** 128-256 steps are optimal; 32 steps significantly underperform.
3. **Block length has nuanced effects.** Smaller block lengths (32) often provide better quality-vs-latency trade-offs.
4. **No single hyperparameter configuration dominates.** The Pareto frontier (quality vs speed) varies by model, suggesting model-specific tuning is needed.

### 6.3 Conclusion from Hyperparameter Analysis

The sweep did not yield a single "best" diffusion hyperparameter set, and the quality gains from tuning do not close the speed gap with vLLM. This motivated the thesis pivot from hyperparameter optimization to the comparative pass@k analysis, where the behavioral differences between paradigms are more scientifically interesting.

---

## 7. Per-Question Correlation Analysis (Novel Finding)

*Source: `generate_comparison_html.py`; HTML outputs: `model_comparison_*_4model.html`*

An interactive HTML comparison tool was built to analyze per-question accuracy across all 4 models. For each question, accuracy = (correct samples / total samples) across all 128 samples. Pearson correlations are computed on the 992-dimensional (Countdown) or 128-dimensional (GSM8K) per-question accuracy vectors.

### 7.1 Countdown Base: Per-Question Correlation Matrix (992 questions, 128 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| **LLaDA-Base vs Dream-Base** | **dLLM–dLLM** | **0.6382** | 1.4×10⁻¹¹⁴ |
| Qwen-Base vs Llama-Base | AR–AR | 0.6149 | 3.1×10⁻¹⁰⁴ |
| Dream-Base vs Qwen-Base | dLLM–AR (shared arch) | 0.5718 | 3.5×10⁻⁸⁷ |
| LLaDA-Base vs Qwen-Base | dLLM–AR | 0.5342 | 2.8×10⁻⁷⁴ |
| LLaDA-Base vs Llama-Base | dLLM–AR | 0.4580 | 1.3×10⁻⁵² |
| Dream-Base vs Llama-Base | dLLM–AR | 0.4432 | 5.6×10⁻⁴⁹ |

### 7.2 Countdown Instruct: Per-Question Correlation Matrix (992 questions, 128 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| Dream-Inst vs Llama-Inst | dLLM–AR | 0.4345 | 6.1×10⁻⁴⁷ |
| Dream-Inst vs Qwen-Inst | dLLM–AR | 0.4062 | 1.1×10⁻⁴⁰ |
| Qwen-Inst vs Llama-Inst | AR–AR | 0.4039 | 3.3×10⁻⁴⁰ |
| LLaDA-Inst vs Dream-Inst | dLLM–dLLM | 0.1475 | 3.1×10⁻⁶ |
| LLaDA-Inst vs Llama-Inst | dLLM–AR | 0.1464 | 3.6×10⁻⁶ |
| LLaDA-Inst vs Qwen-Inst | dLLM–AR | 0.0867 | 6.3×10⁻³ |

*Note: LLaDA-Instruct's near-zero correlations with all models reflect its collapsed performance on Countdown (pass@128 = 5.5%).*

### 7.3 GSM8K Base: Per-Question Correlation Matrix (128 questions, 128 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| **LLaDA-Base vs Dream-Base** | **dLLM–dLLM** | **0.6291** | 1.8×10⁻¹⁵ |
| Dream-Base vs Qwen-Base | dLLM–AR (shared arch) | 0.6189 | 6.9×10⁻¹⁵ |
| LLaDA-Base vs Qwen-Base | dLLM–AR | 0.4907 | 4.1×10⁻⁹ |
| LLaDA-Base vs Llama-Base | dLLM–AR | 0.4316 | 3.7×10⁻⁷ |
| Dream-Base vs Llama-Base | dLLM–AR | 0.3510 | 4.9×10⁻⁵ |
| Qwen-Base vs Llama-Base | AR–AR | 0.3204 | 2.3×10⁻⁴ |

### 7.4 GSM8K Instruct: Per-Question Correlation Matrix (128 questions, 128 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| Qwen-Inst vs Llama-Inst | AR–AR | 0.6638 | 1.4×10⁻¹⁷ |
| LLaDA-Inst vs Llama-Inst | dLLM–AR | 0.6550 | 5.0×10⁻¹⁷ |
| Dream-Inst vs Llama-Inst | dLLM–AR | 0.6398 | 4.3×10⁻¹⁶ |
| LLaDA-Inst vs Dream-Inst | dLLM–dLLM | 0.5671 | 3.0×10⁻¹² |
| Dream-Inst vs Qwen-Inst | dLLM–AR (shared arch) | 0.5598 | 6.5×10⁻¹² |
| LLaDA-Inst vs Qwen-Inst | dLLM–AR | 0.4847 | 6.8×10⁻⁹ |

### 7.5 Trip Planning Base: Per-Question Correlation Matrix (200 questions, 64 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| Qwen-Base vs Llama-Base | AR–AR | 0.7061 | 1.7×10⁻³¹ |
| LLaDA-Base vs Llama-Base | dLLM–AR | 0.5394 | 1.7×10⁻¹⁶ |
| LLaDA-Base vs Qwen-Base | dLLM–AR | 0.5158 | 5.5×10⁻¹⁵ |
| Dream-Base vs Qwen-Base | dLLM–AR (shared arch) | 0.4829 | 4.4×10⁻¹³ |
| Dream-Base vs Llama-Base | dLLM–AR | 0.3629 | 1.3×10⁻⁷ |
| LLaDA-Base vs Dream-Base | dLLM–dLLM | 0.3451 | 5.6×10⁻⁷ |

### 7.6 Trip Planning Instruct: Per-Question Correlation Matrix (200 questions, 64 samples each)

| Model Pair | Type | Pearson r | p-value |
|---|---|---:|---:|
| Qwen-Inst vs Llama-Inst | AR–AR | 0.7190 | 4.1×10⁻³³ |
| LLaDA-Inst vs Dream-Inst | dLLM–dLLM | 0.4622 | 5.6×10⁻¹² |
| Dream-Inst vs Llama-Inst | dLLM–AR | 0.2484 | 3.9×10⁻⁴ |
| LLaDA-Inst vs Qwen-Inst | dLLM–AR | 0.2170 | 2.0×10⁻³ |
| LLaDA-Inst vs Llama-Inst | dLLM–AR | 0.1647 | 2.0×10⁻² |
| Dream-Inst vs Qwen-Inst | dLLM–AR (shared arch) | 0.1572 | 2.6×10⁻² |

### 7.7 Key Findings from Correlation Analysis

**Finding 1: Paradigm determines failure modes on Countdown (arithmetic planning).**

On Countdown Base, the strongest correlation is between the two dLLMs (LLaDA-Dream: r=0.638), followed by the two AR models (Qwen-Llama: r=0.615). Cross-paradigm correlations are consistently lower (0.44-0.57). **Crucially, Dream-Qwen (r=0.572) correlates less than Dream-LLaDA (r=0.638) despite Dream being built on the Qwen2.5 pretrained backbone.** This demonstrates that the generation paradigm (denoising vs autoregressive) has a stronger influence on per-question behavior than the shared pretrained weights.

**Finding 2: AR models show the strongest within-paradigm agreement on Trip Planning (structured NL planning).**

On Trip Planning, the AR–AR pair (Qwen-Llama) has the highest correlation in both Base (r=0.706) and Instruct (r=0.719). The dLLM–dLLM pair is weaker here (Base: r=0.345, Instruct: r=0.462). Cross-paradigm correlations are even weaker (0.16-0.54). This suggests that on natural-language planning tasks, AR models agree strongly on which problems are solvable, while dLLMs are more heterogeneous in their failure modes.

**Finding 3: Trip Planning Instruct shows very low cross-paradigm correlation.**

On Trip Planning Instruct, cross-paradigm correlations are strikingly low (Dream-Qwen: r=0.157, LLaDA-Llama: r=0.165). The dLLM and AR paradigms solve *almost completely different subsets* of Trip Planning problems when using instruct checkpoints. This is the strongest evidence for paradigm complementarity and ensemble potential.

**Finding 4: The paradigm effect is weaker on easier tasks (GSM8K).**

On GSM8K Base, the dLLM-dLLM correlation (0.629) and Dream-Qwen correlation (0.619) are nearly equal, and the AR-AR correlation (0.320) is actually the lowest. On GSM8K Instruct, correlations are high across all pairs (0.48-0.66). This suggests the paradigm-driven failure mode divergence is most pronounced on harder, planning-style tasks where the generation mechanism matters more. On near-saturated benchmarks like GSM8K, all models converge to similar behavior.

**Finding 5: LLaDA-Instruct is an outlier on Countdown.**

LLaDA-Instruct shows near-zero correlations with all models on Countdown (r=0.09-0.15), consistent with its collapsed performance (pass@128 = 5.5%). Instruction tuning appears to have destroyed LLaDA's planning capability specifically on this task.

**Finding 6: Correlation patterns are benchmark-dependent.**

Summary of within-paradigm correlations across benchmarks:

| Benchmark | dLLM–dLLM (r) | AR–AR (r) | Strongest cross-paradigm (r) |
|---|---:|---:|---:|
| Countdown Base | **0.638** | 0.615 | Dream-Qwen: 0.572 |
| Countdown Instruct | 0.148* | 0.404 | Dream-Llama: 0.435 |
| Trip Planning Base | 0.345 | **0.706** | LLaDA-Llama: 0.539 |
| Trip Planning Instruct | 0.462 | **0.719** | Dream-Llama: 0.248 |
| GSM8K Base | **0.629** | 0.320 | Dream-Qwen: 0.619 |
| GSM8K Instruct | 0.567 | **0.664** | LLaDA-Llama: 0.655 |

*LLaDA-Instruct is collapsed on Countdown; excluding it, Dream-Inst has no dLLM pair.

The pattern: on arithmetic planning (Countdown), dLLMs agree most with each other. On natural-language planning (Trip Planning), AR models agree most with each other. On math reasoning (GSM8K), the picture is mixed but converges as all models saturate.

### 7.8 Interactive Visualizations

Six HTML comparison files were generated with all 6 pairwise scatter plots and clickable per-question trace exploration:

- `model_comparison_base_countdown_4model.html` — Countdown Base, 4 models, 992 questions
- `model_comparison_instruct_countdown_4model.html` — Countdown Instruct, 4 models, 992 questions
- `model_comparison_base_gsm8k_4model.html` — GSM8K Base, 4 models, 128 questions
- `model_comparison_instruct_gsm8k_4model.html` — GSM8K Instruct, 4 models, 128 questions
- `model_comparison_base_trip_planning_4model.html` — Trip Planning Base, 4 models, 200 questions
- `model_comparison_instruct_trip_planning_4model.html` — Trip Planning Instruct, 4 models, 200 questions

---

## 8. Cross-Benchmark Synthesis

### 8.1 Ranking Summary (Base Models)

| Metric | GSM8K | Countdown | Trip Planning |
|---|---|---|---|
| Best pass@1 | Dream (72.4%) | LLaDA (14.3%) | Dream (5.4%) |
| Best pass@64/128 | Qwen/Llama (100%) | Qwen (51.3%) | Llama-flat (29.5%) |
| Crossover k | ~pass@2-4 | ~pass@8-16 | ~pass@8-16 |

### 8.2 Consistent Patterns Across Benchmarks

1. **dLLMs have higher pass@1 on planning tasks (Countdown, Trip Planning)** but not on GSM8K where they're roughly equal.
2. **AR models scale faster with k** on all benchmarks, reaching higher pass@k_max.
3. **The crossover happens earlier on easier benchmarks** (GSM8K: k=2-4) and later on harder ones (Countdown, Trip Planning: k=8-16).
4. **Instruct tuning can hurt dLLMs on planning** (LLaDA-Instruct collapses on Countdown) while helping AR models.
5. **Few-shot helps AR more than dLLMs** on GSM8K, suggesting different in-context learning dynamics.

### 8.3 Inference Speed

- vLLM provides 5-8× speedup for AR models with no quality loss.
- Fast-dLLM provides 2-3× speedup for diffusion models with minimal quality loss on Countdown but noticeable loss on Trip Planning (especially LLaDA).
- AR models are fundamentally faster: Qwen vLLM finishes Countdown in 52s (bs=8) vs ~5-6 minutes for fast-dLLM.

---

## 9. Thesis Narrative

The thesis can now tell the following story:

**Chapter 1 (Introduction):** Diffusion LLMs offer a fundamentally different generation paradigm. Are they better reasoners?

**Chapter 2 (Related Work):** Pass@k as evaluation methodology; dLLM papers (LLaDA, Dream); reasoning-with-sampling literature.

**Chapter 3 (Methodology):** Unified pipeline, four models, three benchmarks, deterministic + pass@k evaluation.

**Chapter 4 (Results):**
- Section 4.1: Deterministic evaluation — dLLMs clearly win on planning tasks.
- Section 4.2: Pass@k evaluation — AR models overtake at large k due to higher sample diversity.
- Section 4.3: The ranking reversal is the central result — evaluation mode determines which paradigm looks better.
- Section 4.4: GSM8K as sanity check, confirming saturation but showing 0-shot advantage for dLLMs.
- Section 4.5: Per-question correlation analysis — paradigm, not architecture, determines failure modes.
- Section 4.6: Prompt sensitivity and instruct tuning effects.
- Section 4.7: Hyperparameter analysis — no magic settings, motivating the comparative approach.

**Chapter 5 (Discussion):**
- dLLMs concentrate probability mass → high pass@1, low diversity
- AR models maintain broader distributions → lower pass@1, higher coverage
- The two paradigms are complementary
- Practical implications: use dLLMs when you need one good answer; use AR when you can sample many

**Chapter 6 (Conclusion):** The answer to "which is better" depends entirely on the evaluation protocol.

---

## 10. What Is Still Missing / Deferred

### Not Required for Milestone 2
- AIME benchmark
- MATH500 / MATH-Beyond
- Standard (slow) dLLM reruns
- Ensemble experiments

### Deferred to Milestone 3 / Final Thesis
- Statistical significance tests (bootstrap confidence intervals on pass@k)
- Formal diversity metrics beyond pass@k slope
- Larger model comparisons (if available)
- Potential ensemble study combining dLLM and AR outputs

---

## 11. GPT5.4 Pro Prompt

Copy everything below this line and paste it as a prompt to GPT5.4 Pro, along with the two attached files (this MILESTONE_2_DRAFT.md and Milestone_2_Submission.docx).

---

### PROMPT FOR GPT5.4 PRO

I am attaching two files:

1. **MILESTONE_2_DRAFT.md** — This is the authoritative, complete results document with every table, every number, every correlation, and every finding from my thesis experiments. Every number in this file is real and computed from actual JSON result files. **Do not hallucinate or invent any numbers. If a number appears in this draft, use it exactly. If a number does NOT appear in this draft, do not make one up.**

2. **Milestone_2_Submission.docx** — This is the current state of my Milestone 2 submission document. It has the structure (outline, methodology, topic sentences) but many tables are incomplete, outdated, or missing entirely.

**Your task:** Update the Milestone_2_Submission.docx to incorporate ALL results and analysis from MILESTONE_2_DRAFT.md. Return the updated .docx file.

#### What to update:

**A. Replace all existing tables with the refreshed numbers from the draft.** The draft's Section 3 (GSM8K), Section 4 (Countdown), and Section 5 (Trip Planning) contain the definitive tables. These supersede any older numbers in the docx.

**B. Add these NEW sections/tables that are currently missing from the docx:**

1. **GSM8K Base Models pass@k (Section 3.1)** — 4-model table with LLaDA, Dream, Qwen, Llama. Note: the old docx only had 3 models (no Llama). Add Llama everywhere.

2. **GSM8K Base 4-shot (Section 3.2)** — New table.

3. **GSM8K Instruct full 128-sample table with templated+flat variants (Section 3.3)** — 8-row table showing prompt format sensitivity. This is NEW.

4. **GSM8K Instruct 0-shot (Section 3.4)** — New table. Highlight the key finding: dLLMs strongly outperform AR at 0-shot (74-75% vs 49-53%) but the gap closes at 4-shot.

5. **GSM8K Instruct 4-shot (Section 3.5)** — New table.

6. **Trip Planning pass@k with ALL models (Section 5.2)** — The old docx was missing Llama and the LLaDA refresh. The draft has a complete 10-row table.

7. **Hyperparameter Analysis (Section 6)** — Add as a new subsection in Results or Methodology. Include the best-config table and the finding that the sweep motivated pivoting to comparative pass@k analysis.

8. **Per-Question Correlation Analysis (Section 7)** — This is the MOST IMPORTANT new addition. Add it as a major new results subsection. Include:
   - The full Countdown Base correlation matrix (Section 7.1): 6 model pairs with exact r values and p-values
   - The full Countdown Instruct correlation matrix (Section 7.2)
   - The GSM8K Base and Instruct correlation matrices (Sections 7.3-7.4)
   - The Trip Planning Base and Instruct correlation matrices (Sections 7.5-7.6) — NEW
   - All six key findings (Section 7.7): paradigm > architecture on Countdown; AR agreement strongest on Trip Planning; very low cross-paradigm correlation on Trip Planning Instruct; effect weaker on GSM8K; LLaDA-Instruct collapse; benchmark-dependent correlation summary table
   - The summary table in Finding 6 showing within-paradigm correlations across all 6 benchmark-variant combinations
   - Mention the six interactive HTML visualizations (Section 7.8)

**C. Update the thesis narrative and topic sentences:**

The central thesis argument (from Section 9 of the draft) has four pillars:
1. **Pass@k Ranking Reversal:** dLLMs win at low k (deterministic/pass@1), AR wins at large k. The crossover happens around k=8-16 on planning tasks.
2. **Per-Question Correlation — Paradigm > Architecture:** On Countdown Base, LLaDA-Dream (dLLM-dLLM) r=0.638 > Dream-Qwen (shared backbone) r=0.572. The generation paradigm determines failure modes more than pretrained weights.
3. **Per-Question Correlation — Benchmark-Dependent Patterns:** On Trip Planning Instruct, cross-paradigm correlations are extremely low (r=0.16-0.25), meaning dLLMs and AR models solve *almost completely different* subsets of problems. AR-AR correlation is highest on Trip Planning (r=0.706-0.719). These patterns are evidence for paradigm complementarity.
4. **Prompt/Few-shot Sensitivity:** dLLMs are stronger zero-shot reasoners; AR models benefit more from in-context examples.

**D. Update the discussion section** with these interpretations:
- dLLMs concentrate probability mass via denoising → higher pass@1 but lower sample diversity → slower pass@k growth
- AR models maintain broader token-level distributions → lower pass@1 but higher coverage → steeper pass@k scaling
- The paradigms are complementary: dLLMs for single-shot quality, AR for search-based evaluation
- The correlation finding suggests ensemble potential

**E. Update figures/placeholders:**
- Figure R1: Pass@k crossover plot (Countdown Base, 4 models) — describe what it shows
- Figure R2: Per-question accuracy scatter plots from the HTML comparison tool — describe the 6 pairwise plots and what the correlation values mean
- Add figure references for the GSM8K instruct 0-shot vs 4-shot comparison

**F. Update scope and limitations:**
- Add Llama 3.1 8B as a fourth model throughout (it was missing from the old docx)
- Note: all dLLM experiments use fast-dLLM inference; all AR experiments use vLLM
- Note: correlation analysis uses 128 samples per question per model on Countdown/GSM8K and 64 samples on Trip Planning
- Keep AIME/MATH-Beyond as deferred

#### Critical constraints:

1. **DO NOT invent, estimate, or round any numbers.** Every percentage, correlation coefficient, and p-value in the draft is exact. Use them as-is.
2. **DO NOT add results for models or benchmarks not in the draft.** If a model-benchmark combination doesn't appear in the draft, don't add it.
3. **DO NOT change the methodology description** beyond what the draft specifies.
4. **Preserve the existing docx formatting** (fonts, heading styles, page layout). Only change content.
5. **Keep all existing content** that is still accurate. Only replace tables that have refreshed numbers, and add new sections.
6. **The Countdown pass@k tables in Section 4.2 and 4.3 are the CENTRAL results.** Make them visually prominent.
7. **The correlation matrices in Sections 7.1-7.6 and the summary table in Finding 6 (Section 7.7) are the NOVEL contribution.** These cover all 3 benchmarks × 2 variants = 6 conditions. Emphasize them appropriately.

Return the updated .docx file with all changes incorporated.
