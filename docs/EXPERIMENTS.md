# Experiment Guide

This document describes all experiments that can be run with `run_evaluation.sh`,
the unified evaluation script for comparing diffusion LLMs (LLaDA, Dream) against
autoregressive LLMs (Qwen) across base and instruct variants.

## Quick Reference

```bash
# Single command, all experiments controlled via flags:
./run_evaluation.sh --experiment <type> [options]
```

---

## Experiment Types

### 1. `accuracy` — Deterministic Single-Sample Accuracy

**Purpose:** Measure greedy-decode accuracy (pass@1 with temp=0) and verify that
fast inference paths (fast-dLLM, vLLM) match their slow counterparts (dLLM, AR-HF).

**Defaults:**
- `temperature=0`, `n_samples=1`, `batch_size=8`, `method=both`

**What it answers:**
- What is each model's deterministic accuracy on a given task?
- Does fast-dLLM match dLLM in output quality? Does vLLM match HF generate?

**Example commands:**
```bash
# All 6 models, fast + slow, on countdown_cd4
./run_evaluation.sh -E accuracy

# Only base models on MATH500
./run_evaluation.sh -E accuracy -d math -v base

# Dream instruct only, fast path only
./run_evaluation.sh -E accuracy -d countdown_cd4 -m dream -v instruct --method fast
```

**Replaces:** `run_deterministic_comparison.sh`

---

### 2. `passk` — Pass@k Scaling Analysis

**Purpose:** Generate `n` samples per question (with temperature>0) and compute
pass@k for k=1,2,4,8,...,128. This is the core experiment for studying how
accuracy scales with repeated sampling — the central question of the project.

**Defaults:**
- `temperature=0.7`, `n_samples=128`, `batch_size=8`, `method=fast`

**What it answers:**
- How does pass@k grow with k for base vs instruct models?
- Do base models overtake instruct models at large k (distribution sharpening)?
- Do diffusion LLMs explore the solution space differently than AR models?
- See [docs/ANALYSIS.md](ANALYSIS.md) for the theoretical framework.

**Example commands:**
```bash
# Pass@k on countdown_cd4 (all 6 models, n=128)
./run_evaluation.sh -E passk -d countdown_cd4

# Only instruct models on countdown_cd4
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct

# Pass@k on hard math benchmarks
./run_evaluation.sh -E passk -d math           # MATH500 (200 problems)
./run_evaluation.sh -E passk -d aime           # AIME 2024+2025 (60 problems)
./run_evaluation.sh -E passk -d math_beyond    # MATH-Beyond (181 problems)
./run_evaluation.sh -E passk -d trip_planning  # Trip Planning (200 problems)

# Reduced n_samples for faster iteration
./run_evaluation.sh -E passk -d math -n 32

# Only LLaDA base on AIME
./run_evaluation.sh -E passk -d aime -m llada -v base
```

**Replaces:** `run_passk_experiments.sh`, `run_hard_benchmarks.sh`

---

### 3. `batch` — Batch Size Comparison

**Purpose:** Compare batch_size=1 vs batch_size=8 to measure throughput speedup
and verify that batching doesn't degrade accuracy.

**Defaults:**
- `temperature=0`, `n_samples=1`, `method=both`
- Runs all models at bs=1, then all models at bs=8

**What it answers:**
- How much wall-clock speedup does batching provide?
- Is accuracy identical between bs=1 and bs=8?
- How do fast and slow inference paths compare at different batch sizes?

**Example commands:**
```bash
# Full batch comparison on countdown_cd4
./run_evaluation.sh -E batch

# Dream only
./run_evaluation.sh -E batch -m dream

# On a different dataset
./run_evaluation.sh -E batch -d gsm8k
```

**Replaces:** `run_batch_comparison.sh`

---

### 4. `speed` — Inference Speed Comparison

**Purpose:** Compare fast vs slow inference paths in wall-clock time. Functionally
identical to `accuracy` (both use temp=0, n=1, method=both) but the intent is
to compare timing rather than correctness. Results include wall-clock time per
batch in the output JSON.

**Defaults:**
- Same as `accuracy`: `temperature=0`, `n_samples=1`, `batch_size=8`, `method=both`

**What it answers:**
- How much faster is fast-dLLM vs standard dLLM?
- How much faster is vLLM vs HuggingFace `generate()`?
- What is the overall throughput (questions/second) for each method?

**Example commands:**
```bash
# Speed comparison on countdown_cd4
./run_evaluation.sh -E speed

# Compare at different batch sizes manually
./run_evaluation.sh -E speed -B 1
./run_evaluation.sh -E speed -B 16
```

**Replaces:** `run_deterministic_comparison.sh` (speed-focused use)

---

### 5. `sweep` — Hyperparameter Grid Search

**Purpose:** Grid search over `gen_length × steps × block_length` to find optimal
diffusion generation parameters. Only applicable to diffusion models (LLaDA, Dream).

**Defaults:**
- `temperature=0.7`, `n_samples=128`, `batch_size=1`, `method=fast`
- Grid: gen_length ∈ {128, 256} × steps ∈ {32,64,128[,256]} × block ∈ {32,64,128[,256]}

**What it answers:**
- What is the optimal steps-to-gen_length ratio for quality vs speed?
- How does block_length affect generation quality?
- At what point do diminishing returns set in for more diffusion steps?

**Example commands:**
```bash
# Sweep LLaDA on GSM8K
./run_evaluation.sh -E sweep -m llada -d gsm8k

# Sweep Dream on countdown
./run_evaluation.sh -E sweep -m dream -d countdown_cd4
```

**Replaces:** `run_experiments.sh`

---

## Global Options Reference

| Flag | Short | Values | Default | Description |
|------|-------|--------|---------|-------------|
| `--experiment` | `-E` | accuracy, passk, batch, speed, sweep | *required* | Experiment type |
| `--dataset` | `-d` | see below | countdown_cd4 | Dataset to evaluate on |
| `--model` | `-m` | llada, dream, qwen, llama, all | all | Model family |
| `--variant` | `-v` | base, instruct, all | all | Model variant |
| `--method` | | fast, slow, both | per-experiment | Inference method |
| `--n_samples` | `-n` | integer | per-experiment | Samples per question |
| `--batch_size` | `-B` | integer | per-experiment | Batch size |
| `--temp` | `-t` | float | per-experiment | Temperature |
| `--few_shot` | `-f` | integer | per-dataset | Few-shot examples |
| `--num_evals` | `-e` | integer | per-dataset | Number of test problems |
| `--gen_length` | `-g` | integer | per-dataset | Max generation tokens |
| `--steps` | `-s` | integer | per-dataset | Diffusion steps |
| `--block_length` | `-b` | integer | per-dataset | Block length |
| `--output_dir` | `-o` | path | results | Output directory |

---

## Datasets

| Dataset | ID | Problems | Few-shot | gen_length | steps | Description |
|---------|----|----------|----------|------------|-------|-------------|
| Countdown cd4 | `countdown_cd4` | 992 | 8 | 32 | 32 | 4-number arithmetic (Dream official) |
| Countdown cd3 | `countdown` | 992 | 8 | 24 | 24 | 3-number arithmetic |
| Countdown cd5 | `countdown_cd5` | 992 | 8 | 24 | 24 | 5-number arithmetic |
| GSM8K | `gsm8k` | 256 | 4 | 256 | 256 | Grade-school math |
| MATH500 | `math` | 200 | 4 | 1024 | 512 | Competition math (subsample, seed=42) |
| AIME | `aime` | 60 | 4 | 1024 | 512 | AIME 2024 + 2025 combined |
| MATH-Beyond | `math_beyond` | 181 | 4 | 1024 | 512 | Hard math beyond standard benchmarks |
| Trip Planning | `trip_planning` | 200 | 2 | 256 | 256 | Multi-city travel planning (Dream official) |
| Sudoku | `sudoku` | 256 | 8 | 24 | 24 | Sudoku constraint satisfaction |

**Notes:**
- Few-shot values are for base models. Instruct models automatically use 0-shot.
- `math` and `trip_planning` use reproducible subsampling (seed=42) to ensure the
  same 200 problems are evaluated across all 6 models.
- AIME and MATH-Beyond use all available problems.

---

## Models

| Family | Base | Instruct | Type | Inference |
|--------|------|----------|------|-----------|
| LLaDA | `GSAI-ML/LLaDA-8B-Base` | `GSAI-ML/LLaDA-8B-Instruct` | Diffusion | fast-dLLM / dLLM |
| Dream | `Dream-org/Dream-v0-Base-7B` | `Dream-org/Dream-v0-Instruct-7B` | Diffusion | fast-dLLM / dLLM |
| Qwen | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | Autoregressive | vLLM / AR-HF |
| LLaMA | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | Autoregressive | vLLM / AR-HF |

**Model-specific fast-dLLM parameters:**
- **Dream**: `alg=confidence_threshold`, `dual_cache=True`, `cache_refresh_steps=4`
- **LLaDA**: `alg=entropy`, `dual_cache=True`, `cache_refresh_steps=0`

---

## Inference Methods

| Method | Diffusion Models | AR Models | Speed |
|--------|-----------------|-----------|-------|
| **fast** | fast-dLLM (KV-cache optimized) | vLLM (PagedAttention) | Fast |
| **slow** | dLLM (standard diffusion_generate) | AR-HF (HuggingFace generate) | Slow |

The `fast` methods use inference optimizations (KV-cache for diffusion, PagedAttention
for AR) that provide 2-5x speedup with negligible quality loss. The `slow` methods
use the reference implementations and serve as ground truth for correctness.

---

## Output Files

Results are saved as JSON files in the `results/` directory with filenames encoding
all parameters:

```
# Diffusion (fast-dLLM):
{model}_{dataset}_{gen_length}_{steps}_{block}_{batch_size}_{temp}_{few_shot}_{num_evals}_{n_samples}_generations_{prompt_mode}_fast_dllm.json

# Diffusion (slow dLLM):
{model}_{dataset}_{gen_length}_{steps}_{block}_{batch_size}_{temp}_{few_shot}_{num_evals}_{n_samples}_generations_{prompt_mode}_dllm.json

# AR (vLLM):
{model}_{dataset}_{gen_length}_{batch_size}_{temp}_{few_shot}_{num_evals}_{n_samples}_generations_{prompt_mode}_vllm.json

# AR (slow HF):
{model}_{dataset}_{gen_length}_{batch_size}_{temp}_{few_shot}_{num_evals}_{n_samples}_generations_{prompt_mode}_ar.json
```

Where `{prompt_mode}` resolves to one of:
- `base_native`
- `instruct_templated`
- `instruct_flat`
- `instruct_single_turn`

Each JSON file contains:
- `generations`: list of per-question results with generated text, extracted answers, and ground truth
- `metrics`: timing and configuration metadata

Comparison tables are saved as `results/{experiment}_{dataset}_comparison.json`.

---

## Prompt Modes (3-Way Comparison)

The `--prompt_mode` flag and `--variant all3` enable a controlled comparison
that disentangles instruction tuning from chat template effects. See
[docs/PROMPT_MODES.md](PROMPT_MODES.md) for full details.

| Mode | Checkpoint | Prompt | When to use |
|------|-----------|--------|-------------|
| `base_native` | Base | Flat string | Default for base models |
| `instruct_flat` | Instruct | Same flat string as base | Ablation: instruct weights, base prompt |
| `instruct_templated` | Instruct | Model-native chat template | Default and recommended reporting mode for instruct models |
| `instruct_single_turn` | Instruct | One user chat turn containing the full flat prompt | Diagnostic parity check for chat-wrapped but non-conversational serialization |

```bash
# Full 3-way comparison (base + instruct_templated + instruct_flat)
./run_evaluation.sh -E passk -d countdown_cd4 -v all3

# Run instruct-flat ablation separately
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_flat

# Run a single-turn chat parity check
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_single_turn

# Run multiple instruct prompt modes in one command
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_templated --prompt_mode instruct_flat
```

**Result filenames:**
- `base_native` runs include `_base_native`
- `instruct_templated` runs include `_instruct_templated`
- `instruct_flat` runs include `_instruct_flat`
- `instruct_single_turn` runs include `_instruct_single_turn`

---

## Countdown Guidance For LLaDA-Instruct

For local `countdown_cd4` runs, the high-signal settings are simpler:

1. Keep `block_length=32`.
2. Use `instruct_templated` as the main instruct reporting mode.
3. Treat `instruct_flat` and `instruct_single_turn` as ablations.

Rationale:

- `block_length=8` consistently hurt local countdown performance.
- Explicit EOS suppression did not improve local countdown runs and in some direct
  tests made them much worse.
- The remaining low scores are mostly due to equation chains that are formatted
  correctly but arithmetically invalid, not empty outputs or obvious EOS collapse.

So for countdown in this repo, we no longer surface EOS-specific benchmark flags
in the main experiment workflow.

---

## Common Workflows

### Full pass@k study (base vs instruct, all models)

```bash
# 1. Run on each dataset
./run_evaluation.sh -E passk -d countdown_cd4
./run_evaluation.sh -E passk -d math
./run_evaluation.sh -E passk -d trip_planning

# 2. Or run one model at a time to fit GPU memory
./run_evaluation.sh -E passk -d countdown_cd4 -m llada
./run_evaluation.sh -E passk -d countdown_cd4 -m dream
./run_evaluation.sh -E passk -d countdown_cd4 -m qwen
```

### Verify fast path correctness before large runs

```bash
# Quick accuracy check: fast vs slow should match within ~0.5%
./run_evaluation.sh -E accuracy -d countdown_cd4 -m llada -v base
```

### Iterate quickly with fewer samples

```bash
# Use n=16 for fast iteration, then scale up
./run_evaluation.sh -E passk -d math -n 16
# Once validated, run full:
./run_evaluation.sh -E passk -d math -n 128
```

### Run only missing models (skip logic)

The script automatically skips completed experiments (checks output file existence
and completeness). So you can safely re-run the same command — it will only execute
what's missing.

```bash
# If LLaDA finished but Dream OOM'd, just re-run:
./run_evaluation.sh -E passk -d countdown_cd4
# LLaDA is skipped, Dream resumes.
```

---

## Mapping from Old Scripts

| Old Script | Equivalent Command |
|---|---|
| `run_experiments.sh llada` | `./run_evaluation.sh -E sweep -m llada -d gsm8k` |
| `run_batch_comparison.sh` | `./run_evaluation.sh -E batch` |
| `run_batch_comparison.sh -d countdown_cd4 dream` | `./run_evaluation.sh -E batch -d countdown_cd4 -m dream` |
| `run_deterministic_comparison.sh all` | `./run_evaluation.sh -E accuracy` |
| `run_deterministic_comparison.sh -v base -m fast llada` | `./run_evaluation.sh -E accuracy -v base --method fast -m llada` |
| `run_instruct_comparison.sh -v base all` | `./run_evaluation.sh -E passk -v base -n 16 -B 1` |
| `run_instruct_comparison.sh -d countdown -v instruct qwen` | `./run_evaluation.sh -E passk -d countdown -v instruct -m qwen -n 16` |
| *(new)* LLaMA 3.1 8B pass@k on countdown | `./run_evaluation.sh -E passk -d countdown_cd4 -m llama` |
| *(new)* LLaMA 3.1 8B base accuracy on MATH500 | `./run_evaluation.sh -E accuracy -d math -m llama -v base` |
| `run_passk_experiments.sh -v instruct` | `./run_evaluation.sh -E passk -d countdown_cd4 -v instruct` |
| `run_passk_experiments.sh -v all` | `./run_evaluation.sh -E passk -d countdown_cd4` |
| `run_hard_benchmarks.sh -d math` | `./run_evaluation.sh -E passk -d math -n 64` |
| `run_hard_benchmarks.sh -d aime -m llada` | `./run_evaluation.sh -E passk -d aime -m llada -n 64` |
| *(new)* 3-way prompt comparison on cd4 | `./run_evaluation.sh -E passk -d countdown_cd4 -v all3` |
| *(new)* instruct-flat ablation only | `./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_flat` |
