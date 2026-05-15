# RCoT-Diffusion-LLM

Comparative evaluation of diffusion LLMs and autoregressive LLMs for reasoning, planning, and sampling-based performance.

## Environment Setup

The evaluation scripts expect to be run from the repository root and use a local
Python virtual environment named `.venv`.

```bash
cd RCoT-Diffusion-LLM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`flash-attn` can be sensitive to the local CUDA/PyTorch build. If installation
from `requirements.txt` fails at that package, install the core requirements
first and then install FlashAttention separately:

```bash
grep -v '^flash-attn' requirements.txt > /tmp/requirements-no-flash-attn.txt
pip install -r /tmp/requirements-no-flash-attn.txt
pip install flash-attn --no-build-isolation
```

The model checkpoints are loaded from Hugging Face. Log in before running
experiments, especially for gated models such as Llama:

```bash
huggingface-cli login
```

The main experiments require a CUDA GPU with enough memory for 7B--8B models.
For quick checks, use small `--num_evals`, `--n_samples`, and `--batch_size`
values before launching full runs.

## Running Evaluations

Use `run_evaluation.sh` for the thesis experiments. The script activates
`.venv`, selects the correct inference path for each model family, builds
prompts, runs generation, scores outputs, and writes JSON artifacts for later
analysis.

```bash
./run_evaluation.sh --experiment <accuracy|passk|batch|speed|sweep> [options]
```

Useful common options:

- `-d, --dataset`: `gsm8k`, `countdown_cd4`, `trip_planning`, `math`, `aime`, `math_beyond`
- `-m, --model`: `llada`, `dream`, `qwen`, `llama`, `all`, or a comma-separated list
- `-v, --variant`: `base`, `instruct`, `all`, or `all3`
- `-n, --n_samples`: number of completions per question
- `-e, --num_evals`: number of evaluation questions
- `-B, --batch_size`: inference batch size
- `-o, --output_dir`: directory for generated JSON artifacts
- `--prompt_mode`: `auto`, `instruct_templated`, `instruct_flat`, or `instruct_single_turn`

### Quick Smoke Tests

Run a tiny autoregressive evaluation and write JSON files under
`results/smoke/`:

```bash
./run_evaluation.sh \
  --experiment accuracy \
  -d countdown_cd4 \
  -m qwen \
  -v base \
  -e 5 \
  -n 1 \
  -B 1 \
  -o results/smoke
```

Run a tiny diffusion evaluation:

```bash
./run_evaluation.sh \
  --experiment accuracy \
  -d countdown_cd4 \
  -m llada \
  -v base \
  --method fast \
  -e 5 \
  -n 1 \
  -B 1 \
  -o results/smoke
```

Each command should create a `*_generations_*.json` file in the chosen output
directory. These files contain the prompts, generated completions, parsed
correctness fields, and run metadata used by the analysis scripts.

### Full Thesis-Style Runs

Countdown-cd4 pass@k for all base and instruction-tuned models:

```bash
./run_evaluation.sh --experiment passk -d countdown_cd4 -v all -o results
```

GSM8K pass@k:

```bash
./run_evaluation.sh --experiment passk -d gsm8k -v all -o results
```

Trip Planning pass@k:

```bash
./run_evaluation.sh --experiment passk -d trip_planning -v all -o results
```

Prompt-mode ablation for instruction-tuned models:

```bash
./run_evaluation.sh \
  --experiment passk \
  -d countdown_cd4 \
  -v instruct \
  --prompt_mode instruct_templated \
  --prompt_mode instruct_flat \
  -o results
```

By default, `passk` uses stochastic sampling and stores many completions per
question. The resulting JSON filenames encode the model, dataset, generation
length, diffusion steps or AR batch size, temperature, few-shot count,
evaluation-set size, sample count, prompt mode, and inference backend.

Example filename patterns:

- Diffusion fast path:
  `results/GSAI-ML_LLaDA-8B-Base_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json`
- Autoregressive vLLM path:
  `results/Qwen_Qwen2.5-7B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json`

The generated JSON files are the inputs for pass@k comparison, parser analysis,
qualitative inspection, and thesis figure/table generation.

## Milestone 2

The current milestone-2 writeup draft lives in `docs/MILESTONE_2_DRAFT.md`.

Main milestone-2 benchmarks currently tracked:

- GSM8K
- Countdown-cd4
- Trip Planning

## Latest Countdown Refresh

These refreshed fast-only Countdown pass@k runs are now available:

- `results/milestone2_countdown_base_refresh/passk_countdown_cd4_comparison.json`
- `results/milestone2_countdown_instruct_refresh/passk_countdown_cd4_comparison.json`

### Base Models

| Model | pass@1 | pass@16 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|
| LLaDA-8B-Base [fast-dllm] | 14.29% | 21.94% | 25.56% | 27.22% |
| Dream-v0-Base-7B [fast-dllm] | 11.30% | 28.57% | 35.17% | 37.60% |
| Qwen2.5-7B [vllm] | 4.04% | 30.14% | 45.70% | 51.31% |
| Llama-3.1-8B [vllm] | 2.70% | 24.75% | 42.90% | 49.50% |

### Instruct Models

| Model | pass@1 | pass@16 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|
| LLaDA-8B-Instruct [fast-dllm] | 2.41% | 4.14% | 5.16% | 5.54% |
| Dream-v0-Instruct-7B [fast-dllm] | 9.15% | 23.19% | 29.32% | 32.26% |
| Qwen2.5-7B-Instruct [vllm] | 7.84% | 18.12% | 21.28% | 22.08% |
| Llama-3.1-8B-Instruct [vllm] | 5.78% | 31.63% | 42.44% | 46.17% |

## Notes

- Deterministic Countdown still favors diffusion models.
- Large-k Countdown favors AR models, especially Qwen and Llama.
- For the full milestone interpretation and the GSM8K and Trip Planning tables, see `docs/MILESTONE_2_DRAFT.md`.
