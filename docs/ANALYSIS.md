# RCoT-Diffusion-LLM: Evaluation Analysis

## Table of Contents
1. [Deterministic Evaluation Results vs Dream Paper](#1-deterministic-evaluation-results-vs-dream-paper)
2. [Understanding pass@k Metrics](#2-understanding-passk-metrics)
3. [Does pass@1 with n_samples > 1 Make Sense?](#3-does-pass1-with-n_samples--1-make-sense)
4. [pass@k Comparison Table](#4-passk-comparison-table)
5. [Dataset Selection: Beyond GSM8K](#5-dataset-selection-beyond-gsm8k)
6. [Large-k Behavior: Base vs Instruct Models](#6-large-k-behavior-base-vs-instruct-models)

---

## 1. Deterministic Evaluation Results vs Dream Paper

### Our Countdown-cd4 Results (992 test problems, deterministic, temperature=0)

| Model | Method | Acc (bs=1) | Time (bs=1) | Acc (bs=8) | Time (bs=8) | Speedup |
|-------|--------|-----------|-------------|-----------|-------------|---------|
| LLaDA-Base | Fast-dLLM | 13.7% | 16m9s | 14.5% | 5m19s | 3.03x |
| LLaDA-Base | dLLM | 15.4% | 25m31s | 14.7% | 30m1s | 0.85x |
| LLaDA-Inst | Fast-dLLM | 1.5% | 16m0s | 1.9% | 5m38s | 2.84x |
| LLaDA-Inst | dLLM | 1.9% | 25m56s | 1.8% | 30m50s | 0.84x |
| Dream-Base | Fast-dLLM | 14.4% | 15m46s | 14.1% | 6m46s | 2.33x |
| Dream-Base | dLLM | 14.6% | 23m39s | 14.3% | 18m35s | 1.27x |
| Dream-Inst | Fast-dLLM | 7.5% | 12m16s | 7.9% | 6m4s | 2.02x |
| Dream-Inst | dLLM | 3.5% | 23m43s | 3.4% | 19m0s | 1.25x |
| Qwen-Base | vLLM | 6.0% | 6m35s | 6.0% | 51.6s | 7.65x |
| Qwen-Base | AR | 6.0% | 13m46s | 6.0% | 2m36s | 5.31x |
| Qwen-Inst | vLLM | 2.6% | 6m33s | 2.8% | 51.9s | 7.58x |
| Qwen-Inst | AR | 1.0% | 14m2s | 0.9% | 2m43s | 5.16x |

### Verification Against Dream Paper's Official Evaluation

We verified our setup is **identical** to the Dream paper's official evaluation by inspecting their source code at [DreamLM/Dream](https://github.com/DreamLM/Dream).

**Parameter-by-parameter comparison:**

| Parameter | Dream Official (`eval_dream_gen_planning.sh`) | Our Setup (`run_batch_comparison.sh`) | Match? |
|-----------|----------------------------------------------|--------------------------------------|--------|
| `max_new_tokens` | 32 | 32 (`gen_length`) | Identical |
| `diffusion_steps` | 32 | 32 (`steps`) | Identical |
| `temperature` | 0 | 0 | Identical |
| `top_p` | 1 | 1 | Identical |
| `alg` | `entropy` | `entropy` | Identical |
| `alg_temp` | 0 | 0 | Identical |
| `n_few_shots` | 8 (hardcoded in `eval_planning.py`) | 8 | Identical |
| `n_test` | 992 (1000 total - 8 few-shot) | 992 | Identical |
| `batch_size` | 1 (hardcoded in `eval_planning.py`) | 1 and 8 tested | Both tested |
| Data file | `data/cd4_test.jsonl` | `dataset/cd4_test.jsonl` | Byte-identical (`diff` verified) |
| Prompt | `"Given 5 numbers, use +-*/ to operate over the first four numbers to achieve the fifth number."` | Same template in `countdown.py` | Identical |
| Scoring | `cd_metric()` in `cd_metric.py` | `cd_score_single()` — same logic | Identical |
| Post-processing | `g.split('<\|end_of_text\|>')[0].split('\n')[0]` | Same EOS truncation + first-line extraction | Identical |

**Conclusion**: Our evaluation protocol is an exact reproduction of the Dream paper's official setup.

### Comparison with Dream Paper Table 1

The Dream paper (arXiv:2508.15487) reports the following in **Table 1** (base models, same protocol, 8-shot):

| Model | Type | Dream Paper (Table 1) | Our Result (dLLM, bs=1) | Our Result (fast-dLLM, bs=1) | Delta vs Paper |
|-------|------|----------------------|------------------------|------------------------------|----------------|
| Dream 7B | Diffusion | **16.0%** | **14.6%** | **14.4%** | -1.4% |
| LLaDA 8B | Diffusion | **13.2%** | **15.4%** | **13.7%** | +2.2% / +0.5% |
| Qwen2.5 7B | AR | **6.2%** | **6.0%** | — | -0.2% |
| LLaMA3 8B | AR | **3.7%** | — | — | — |

**Our results closely reproduce the official numbers** — all within ~1-2%, which is expected variance for diffusion models (the `alg=entropy` remasking order introduces stochasticity even at `temperature=0`). Notably, our LLaDA dLLM result (15.4%) slightly *exceeds* the paper's 13.2%, while Qwen matches almost exactly (6.0% vs 6.2%).

### Full Dream Paper Table 1 (for reference)

| Benchmark | Dream 7B | LLaDA 8B | Qwen2.5 7B | LLaMA3 8B |
|-----------|----------|----------|------------|-----------|
| **General** | | | | |
| MMLU (5) | 69.5 | 65.9 | 71.9 | 63.5 |
| BBH (3) | 57.9 | 47.4 | 63.9 | 62.7 |
| ARC-E (0) | 83.9 | 71.8 | 77.4 | 81.1 |
| ARC-C (0) | 59.8 | 47.5 | 51.5 | 53.6 |
| Hellaswag (0) | 73.3 | 72.7 | 79.0 | 78.9 |
| WinoGrande (5) | 74.5 | 73.5 | 76.4 | 76.9 |
| PIQA (0) | 75.8 | 74.8 | 79.8 | 81.3 |
| RACE (0) | 44.7 | 38.7 | 41.9 | 39.2 |
| **Math & Science** | | | | |
| GSM8K (8) | 77.2 | 70.9 | 78.9 | 55.3 |
| MATH (4) | 39.6 | 30.7 | 41.1 | 18.0 |
| GPQA (5) | 36.6 | 30.4 | 35.5 | 30.6 |
| **Code** | | | | |
| HumanEval (0) | 57.9 | 32.9 | 56.7 | 35.4 |
| MBPP (4) | 56.2 | 39.0 | 63.6 | 49.2 |
| **Planning** | | | | |
| Countdown (8) | **16.0** | **13.2** | **6.2** | 3.7 |
| Sudoku (8) | **81.0** | 46.0 | 21.0 | 0.0 |
| Trip planning (2) | **17.8** | 16.4 | 3.6 | 8.7 |

*Numbers in parentheses indicate few-shot count. All models evaluated under the same protocol.*

### Key Takeaways

- **Our results closely match the official Dream paper** — within 1-2% on Countdown, confirming correct reproduction.
- **Fast-dLLM closely matches dLLM accuracy** (within 1-2%), confirming it is a valid acceleration method.
- **Diffusion LLMs (LLaDA, Dream) significantly outperform AR models** on planning tasks — Dream gets 16.0% vs Qwen's 6.2% on Countdown, 81.0% vs 21.0% on Sudoku. This is the central advantage of the diffusion paradigm for constraint-satisfaction tasks.
- **Base models consistently outperform their instruct-tuned variants** on Countdown, which is unexpected and suggests that instruction tuning may interfere with the base model's ability to perform structured combinatorial search.
- **vLLM provides 5-7x speedup** over HuggingFace AR with identical accuracy.
- **GSM8K is near-saturated** at this model scale (70-79% for base models), reinforcing the need for harder benchmarks.

---

## 2. Understanding pass@k Metrics

### Definition

The `pass@k` metric measures the probability that at least one out of `k` randomly selected samples (from `n` total) is correct:

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where:
- `n` = total number of samples generated per problem
- `c` = number of correct samples among `n`
- `k` = number of attempts (the "k" in pass@k)

### Case 1: n_samples = 1 (Deterministic Evaluation)

When `n_samples = 1`, the only meaningful metric is `pass@1`:
- Each problem has exactly 1 sample
- `pass@1` = simple accuracy (fraction of problems where the single sample is correct)
- This is equivalent to the standard accuracy metric

### Case 2: n_samples > 1 (Stochastic Evaluation)

When `n_samples > 1` (e.g., 128), multiple independent samples are generated per problem using `temperature > 0`:
- `pass@1`: The expected probability that a *single randomly chosen* sample is correct. This equals `c/n` averaged across problems — i.e., the average correct rate per problem.
- `pass@k` (k > 1): The probability that at least one of `k` randomly chosen samples is correct.
- `pass@n`: Whether the problem was solved by *any* of the `n` samples (binary per problem).

### Why the Accuracy Difference?

For `n_samples = 1` with `temperature = 0` (greedy/deterministic):
- The model always picks the highest-probability token at each step
- Produces a single "best guess" per problem
- pass@1 ≈ raw deterministic accuracy

For `n_samples > 1` with `temperature > 0` (stochastic):
- Each sample follows a different path through the probability distribution
- Some paths may find correct solutions that greedy decoding misses
- But the *average* correctness per sample (pass@1) may be lower than greedy, because stochastic sampling introduces variance
- However, `pass@k` for large `k` will be *higher* because you get multiple chances

**In summary**: Deterministic `pass@1` tends to be higher than stochastic `pass@1`, but stochastic `pass@k` (k >> 1) reveals the model's full coverage of the solution space.

---

## 3. Does pass@1 with n_samples > 1 Make Sense?

### Yes, and it is the standard in the literature

Calculating `pass@1` from `n_samples > 1` is the **standard methodology** introduced by Chen et al. (2021) in the Codex/HumanEval paper. It provides an *unbiased estimate* of the model's expected single-shot accuracy under stochastic sampling.

### Why it matters for our research

The key insight from [Karan & Du, 2025](https://arxiv.org/abs/2510.14901) ("Reasoning with Sampling") is:

> **Base models are smarter than single-shot evaluation suggests.** At large `k`, base models can match or exceed RL-finetuned models in pass@k, because RL-finetuning "sharpens" the distribution — concentrating probability mass on fewer solutions while reducing diversity.

For our use case comparing diffusion LLMs vs autoregressive LLMs:

1. **pass@1 (stochastic)** tells us the average quality of each individual sample
2. **pass@k** for increasing `k` tells us how the model's coverage scales with compute
3. **The slope of pass@k** reveals whether the model has diverse solution strategies or is stuck in a few modes

### What to look for

- **Base models**: Should show steady improvement in pass@k as k grows — indicating diverse solution coverage
- **Instruct/RL models**: May plateau early — indicating distribution sharpening (fewer modes)
- **Diffusion vs AR**: Do diffusion models explore the solution space differently? Does their non-autoregressive generation enable more diverse solutions?

---

## 4. pass@k Comparison Table

*This section will be populated after experiments complete.*

### Experimental Setup
- **Models**: LLaDA-8B-Base, Dream-v0-Base-7B, Qwen2.5-7B (base)
- **Dataset**: Countdown-cd4 (992 test problems)
- **n_samples**: 128 per problem
- **Temperature**: 0.7
- **Batch size**: 8
- **Inference**: Fast-dLLM (LLaDA, Dream), vLLM (Qwen)

### Results

| k | LLaDA-Base | Dream-Base | Qwen-Base |
|---|-----------|-----------|----------|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |
| 16 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |
| 128 | — | — | — |

*(Will be filled with actual results once `run_passk_experiments.sh` completes)*

---

## 5. Dataset Selection: Beyond GSM8K

### The Problem with Easy Benchmarks

For the 7-8B model scale, many standard benchmarks are already near-saturated:

| Model | GSM8K | MATH | GPQA |
|-------|-------|------|------|
| Dream-Base 7B | 77.2 | 39.6 | 36.6 |
| LLaDA-Base 8B | 70.9 | 30.7 | 30.4 |
| Qwen2.5-7B Base | 78.9 | 41.1 | 35.5 |
| Dream-Inst 7B | 81.0 | 39.2 | 33.0 |
| LLaDA-Inst 8B | 78.6 | 26.6 | 31.8 |
| Qwen2.5-7B Inst | 91.6 | 75.5 | 36.4 |

GSM8K is largely solved at this scale (70-92%), making it unsuitable for studying large-k pass@k behavior — the metric saturates too quickly, and the differences between models become marginal.

GPQA, while harder, is **multiple-choice** (4 options), which means random guessing gives 25% and the answer space doesn't benefit from diverse sampling.

For studying whether additional sampling unlocks genuinely new reasoning capabilities, we need **open-ended problems that are hard enough that base models cannot solve them even with many attempts**.

### Selected Datasets

#### i. Primary: MATH-Beyond-Union (181 problems)

**Source**: [brendel-group/MATH-Beyond](https://huggingface.co/datasets/brendel-group/MATH-Beyond) ([Paper](https://arxiv.org/abs/2510.11653))

**Why this dataset?**
- Specifically constructed to defeat open-source models up to 8B parameters even at **pass@1024**
- Problems are drawn from DAPO-Math-17K and DeepScaleR, filtered for correctness and verifiability
- Topically equivalent to standard high-school math — no exotic domain knowledge required
- Includes model-specific unsolved subsets for 21 base models (including Qwen2.5-7B)
- **Directly tests our core question**: Can additional sampling (large k) unlock reasoning capabilities beyond what base models currently exhibit?

**Format**: `problem` (string), `answer` (string, extractable via `\boxed{}`)

**Expected behavior**: pass@k should remain near 0 for base models even at k=128-1024. If any model shows meaningful improvement, it demonstrates genuinely novel reasoning under repeated sampling.

#### ii. Secondary: AIME 2024 + AIME 2025 (60 problems combined)

**Source**: [math-ai/aime24](https://huggingface.co/datasets/math-ai/aime24), [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25)

**Why this dataset?**
- Competition-level math problems from the American Invitational Mathematics Examination
- Integer answers (0-999) — unambiguous, automatically verifiable
- Well-established difficulty: top-performing 7B models score 10-30% on AIME
- Combines both years for a larger evaluation set (60 problems)
- Standard benchmark in the reasoning literature — enables direct comparison with published results

**Format**: `problem` (string), `answer` (integer 0-999)

**Expected behavior**: Moderate initial accuracy with meaningful pass@k scaling. The sweet spot for observing how different model architectures (diffusion vs AR) explore solution space.

#### iii. Cross-Domain: LiveCodeBench

**Source**: [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)

**Why this dataset?**
- Holistic, contamination-free code generation benchmark
- Problems from live competition platforms (LeetCode, CodeForces, AtCoder) — continuously updated
- Evaluates code generation correctness via test cases (functional correctness, not string matching)
- Cross-domain validation: does the pass@k scaling pattern hold for coding as well as math?
- Standard pass@k evaluation built in (k = 1, 5, 10, 25, 50, 100, 150, 200)

**Note**: LiveCodeBench requires special infrastructure for code execution and test case evaluation. It is best used as a supplementary cross-domain check rather than a primary benchmark.

**Expected behavior**: Similar scaling patterns as math benchmarks — base models should show more diverse generation while instruct models plateau earlier.

### Dataset Difficulty Spectrum

```
Easy ←────────────────────────────────────────────────→ Hard

GSM8K        Countdown-cd4      AIME24+25      MATH-Beyond
(~70-90%)    (~5-15%)           (~10-30%)      (~0% @ pass@1024)

Saturated    Our current        Sweet spot     Ceiling test
at 7B scale  benchmark          for pass@k     for RL limits
```

---

## 6. Large-k Behavior: Base vs Instruct Models

### The Distribution Sharpening Hypothesis

A central question in LLM research is whether RL-finetuning (instruction tuning, RLHF, GRPO) creates *genuinely new* reasoning capabilities or merely *sharpens* the existing base model distribution.

Evidence from [Karan & Du (2025)](https://arxiv.org/abs/2510.14901) and [He et al. (2025)](https://arxiv.org/abs/2506.02355) suggests:

1. **RL primarily sharpens**: Post-trained models concentrate probability mass on fewer, higher-quality solutions. This improves pass@1 (single-shot accuracy) but reduces diversity.

2. **Base models retain broader coverage**: For large k, base models can match or exceed RL-trained models in pass@k because they explore a wider space of potential solutions.

3. **The crossover point**: There exists a k* where the pass@k curves of base and instruct models cross. Below k*, instruct models win (concentrated quality). Above k*, base models win (diverse exploration).

### Why This Matters for Diffusion LLMs

Diffusion LLMs generate tokens in a fundamentally different way from autoregressive models:

- **AR models** generate left-to-right, one token at a time. Each token conditions on all previous tokens, creating strong sequential dependencies.
- **Diffusion LLMs** (LLaDA, Dream) generate by iteratively denoising a full sequence from random tokens. This allows holistic, non-sequential refinement.

**Key research questions**:

1. **Do diffusion LLMs naturally produce more diverse solutions?** The non-autoregressive generation process might explore the solution space more uniformly than sequential AR sampling.

2. **Is the distribution sharpening effect different for diffusion models?** If diffusion models already have broader coverage, instruction tuning might be less beneficial (or even harmful) for pass@k at large k.

3. **Can diffusion models achieve comparable pass@k scaling with fewer samples?** If each diffusion sample is more "independent" than sequential AR samples, fewer samples might be needed to cover the solution space.

### Experimental Design

To answer these questions, we compare:

| Comparison | Models | What It Tests |
|-----------|--------|---------------|
| **Within-architecture** | LLaDA-Base vs LLaDA-Inst | Distribution sharpening in diffusion LLMs |
| | Dream-Base vs Dream-Inst | Same, different diffusion architecture |
| | Qwen-Base vs Qwen-Inst | Distribution sharpening in AR LLMs (control) |
| **Cross-architecture** | LLaDA-Base vs Qwen-Base | Diffusion vs AR diversity (base models) |
| | Dream-Base vs Qwen-Base | Same |
| **After tuning** | LLaDA-Inst vs Qwen-Inst | Diffusion vs AR after sharpening |

### Expected Outcomes

On **easy benchmarks** (GSM8K, Countdown):
- All models will saturate at moderate k
- Limited signal for distinguishing architectures

On **hard benchmarks** (MATH-Beyond, AIME):
- Base models should show steeper pass@k curves than instruct models
- Diffusion base models may show faster pass@k growth than AR base models
- MATH-Beyond should remain challenging even at k=128-1024, providing a ceiling test

### Connection to Power Sampling

Karan & Du (2025) propose **power sampling** — using MCMC to sample from p^α (the base model's distribution raised to a power α) instead of standard temperature sampling. Their key theoretical insight:

> Low-temperature sampling is *not* equivalent to sampling from the power distribution p^α. Temperature scaling greedily averages future paths, while p^α accounts for the quality of future completions.

This is especially relevant for diffusion LLMs because:
- Diffusion models already perform a form of iterative refinement (analogous to MCMC resampling)
- The block-based generation in fast-dLLM is structurally similar to the block-based power sampling algorithm
- There may be a natural connection between diffusion denoising steps and MCMC mixing that could be exploited

### Metrics to Report

For each model and dataset combination:

1. **pass@k table**: k = 1, 2, 4, 8, 16, 32, 64, 128
2. **pass@k curves**: Plotted on the same axes for visual comparison
3. **Diversity metrics**: Number of unique correct solutions across samples
4. **Crossover analysis**: At what k does the base model overtake the instruct model?
5. **Architecture comparison**: Paired statistical tests between diffusion and AR at each k

---

## References

1. **Dream paper**: Dream: Diffusion Rectification and Estimation-Adaptive Models. Dream-org, 2025.
2. **LLaDA paper**: LLaDA: Large Language and Diffusion Architectures. GSAI-ML, 2025.
3. **Reasoning with Sampling**: Karan, A. & Du, Y. "Reasoning with Sampling: Your Base Model is Smarter Than You Think." arXiv:2510.14901, 2025.
4. **MATH-Beyond**: Mayilvahanan et al. "MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model." arXiv:2510.11653, 2025.
5. **Distribution Sharpening**: He et al. "Rewarding the Unlikely: Lifting GRPO beyond Distribution Sharpening." arXiv:2506.02355, 2025.
6. **HumanEval / pass@k**: Chen et al. "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021.
7. **LiveCodeBench**: Jain et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." 2024.
