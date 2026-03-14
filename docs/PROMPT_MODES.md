# Prompt Modes: 3-Way Comparison Framework

## Motivation

When evaluating base vs. instruct models, a naive comparison confounds two effects:

1. **Instruction tuning** — the weights were updated to follow instructions.
2. **Chat template formatting** — the prompt is restructured into multi-turn conversation format.

An instruct model that performs poorly with a flat prompt may simply never have seen flat prompts during fine-tuning — that is a prompt-format mismatch, not a capability loss. Conversely, high instruct_templated scores might be driven more by proper prompt formatting than by the instruction-tuned weights themselves.

To disentangle these effects, we implement a 3-way prompt comparison.

## The Three Modes

| Mode | Checkpoint | Prompt Format | Purpose |
|------|-----------|---------------|---------|
| `base_native` | Base | Flat completion string | Baseline: what can the base weights do? |
| `instruct_flat` | Instruct | Same flat string as base | Ablation: how do instruct weights perform with base-style prompts? |
| `instruct_templated` | Instruct | Model-native chat template (multi-turn) | Full instruct: weights + proper formatting together. |

### What Each Comparison Reveals

- **`instruct_flat` vs `base_native`** — Isolates the effect of instruction tuning on the weights while holding the prompt format constant. If instruct_flat >> base_native, the instruct weights are genuinely better at the task even without their native format. If instruct_flat << base_native, the instruct fine-tuning may have degraded flat-prompt performance (common when training only on chat-formatted data).

- **`instruct_templated` vs `instruct_flat`** — Isolates the contribution of chat template formatting. If instruct_templated >> instruct_flat, proper prompt formatting is critical for unlocking the instruct model's potential. This is expected for models like LLaDA-Instruct that were trained exclusively on chat-formatted data.

- **`instruct_templated` vs `base_native`** — The "headline" comparison, but confounded. A large gap could be driven by either the weights, the format, or both.

## Implementation

### Dataset Classes

All dataset classes support `prompt_mode` as an optional constructor argument:

```python
dataset = CTD4Dataset(
    tokenizer,
    num_examples=8,
    subsample=992,
    prompt_mode="instruct_flat",  # or "base_native" / "instruct_templated"
)
```

When `prompt_mode` is not specified, backward-compatible behavior is preserved:
- `is_base_model=True` → `base_native`
- `is_base_model=False` → `instruct_templated`

### Per-Dataset Prompt Structure

#### Countdown / Sudoku / GSM8K / MATH / AIME / MATH-Beyond / Counting Letters

These datasets inherit from `GSM8KDataset`. The prompt modes work as follows:

**`base_native` / `instruct_flat`** — Flat text prompt:
```
{system_prompt}

Question: {example_1}
Answer:
{formatted_answer_1}

Question: {example_2}
Answer:
{formatted_answer_2}

Question: {actual_question}
Answer:
```

**`instruct_templated`** — Multi-turn chat template:
```
[system] {system_prompt}
[user]   Question: {example_1}
[assistant] {formatted_answer_1}
[user]   Question: {example_2}
[assistant] {formatted_answer_2}
[user]   Question: {actual_question}
         Answer:
```

For **Countdown** specifically, the structure is more compact:
```
[system] Given 5 numbers, use +-*/ to operate over the first four numbers
         to achieve the fifth number. Output ONLY comma-separated equations
         like a+b=c,c-d=e with no explanation.
[user]   Input: 86,28,13,31,96
[assistant] 86+28=114,31-13=18,114-18=96
...
[user]   Input: {test_numbers}
```

#### Trip Planning

Trip planning uses a domain-specific `TASK:` format that doesn't decompose cleanly into user/assistant turns. All three modes keep the same prompt content; only the wrapping differs:

- **`base_native` / `instruct_flat`**: flat prompt + `\nSOLUTION: ` suffix
- **`instruct_templated`**: flat prompt wrapped in a single user message with chat template

### Evaluation Functions

All four evaluation functions (`evaluate_fast_dllm`, `evaluate_dllm`, `evaluate_vllm_model`, `evaluate_auto_regressive_model`) accept `prompt_mode=None`. When `None`, the dataset infers the mode from `is_base_model`.

### Result Filenames

The filename convention distinguishes `instruct_flat` results from standard runs:

- Standard (base_native or instruct_templated): `...generations_fast_dllm.json`
- instruct_flat: `...generations_flat_fast_dllm.json`

The `_flat` tag is inserted before the engine suffix.

## CLI Usage

### `run_evaluation.sh`

#### Single-mode run
```bash
# Standard base + instruct (templated) — same as before
./run_evaluation.sh -E passk -d countdown_cd4

# Instruct-flat ablation only
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_flat
```

#### Full 3-way comparison
```bash
# Runs: base(native) + instruct(templated) + instruct(flat)
./run_evaluation.sh -E passk -d countdown_cd4 -v all3
```

The comparison table generated at the end labels instruct_flat entries as `ModelName [flat]`.

#### Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--prompt_mode` | `auto`, `instruct_flat` | `auto` | Override prompt format for instruct models |
| `-v, --variant` | `base`, `instruct`, `all`, `all3` | `all` | `all3` = base + instruct + instruct_flat |

## Expected Effects on Countdown CD4 Accuracy

### Baseline Results (from previous runs)

| Model | Base (native) pass@1 | Instruct (old single-turn) pass@1 |
|-------|---------------------|------------------------------------|
| Dream 7B | ~16.0% | ~19.2% |
| LLaDA 8B | ~13.2% | ~0.95% |
| Qwen 2.5 7B | ~6.2% | ~1.6% |
| LLaMA 3.1 8B | ~3.7% | ~11.3% |

### Predicted Effect of Multi-Turn Template (instruct_templated)

The previous instruct runs used a single-turn prompt where the system prompt and all few-shot examples were concatenated into one user message. The new `instruct_templated` mode structures this as proper multi-turn conversation.

**Expected improvements:**

1. **LLaDA-Instruct** — Largest expected improvement (from ~1% to potentially 5-15%). LLaDA-Instruct was trained on multi-turn conversations; the old flat-style prompt was severely mismatched with its training distribution. The multi-turn format with system/user/assistant turns should dramatically improve format compliance and reduce truncated/verbose outputs.

2. **Dream-Instruct** — Moderate improvement expected (19% → 20-25%). Dream-Instruct already performed reasonably, suggesting some robustness to prompt format. Multi-turn should still help by making the few-shot examples more clearly delineated.

3. **Qwen-Instruct** — Significant improvement expected (1.6% → 5-12%). Similar to LLaDA, Qwen's instruct training heavily emphasizes chat-format inputs. The old prompt format led to verbose explanations instead of compact equations.

4. **LLaMA-Instruct** — Small improvement expected (11% → 12-16%). Already performed well with the old format, suggesting good prompt robustness.

### Predicted Effect of instruct_flat (Ablation)

Running instruct models with flat base-style prompts should generally show:

- **Degradation for LLaDA/Qwen-Instruct** — These models were heavily fine-tuned on chat data, so flat prompts are out-of-distribution. Expect performance close to or below the (already low) old instruct scores.
- **Minimal change for Dream/LLaMA-Instruct** — These may retain more flat-prompt capability, showing scores closer to their base counterparts.

### pass@k Scaling Predictions

For the countdown task, the 3-way comparison at higher k values should reveal:

| k | base_native | instruct_flat | instruct_templated |
|---|------------|---------------|-------------------|
| 1 | Moderate | Low (format mismatch) | Highest |
| 16 | Higher | Still limited | Much higher |
| 128 | Plateauing | May catch up slightly | Highest ceiling |

The key insight: **instruct_templated should show the steepest pass@k curve** because the model generates more format-compliant outputs, meaning more of the k samples produce parseable (and potentially correct) answers. Even if the underlying reasoning capability is similar, better format compliance yields higher effective coverage.

## Methodological Notes

1. **Identical few-shot content** — All three modes use the same few-shot examples (same data, same count). The only difference is how the content is serialized (flat string vs. multi-turn messages).

2. **Identical generation parameters** — Temperature, max tokens, top-p, and diffusion parameters are the same across modes for fair comparison.

3. **Identical evaluation questions** — The reproducible subsampling (seed=42) ensures the same test questions are used regardless of prompt mode.

4. **Apples-to-apples** — The `instruct_flat` mode uses the *exact same string* as `base_native`. This ensures any performance difference is attributable solely to the model weights, not the prompt format.

5. **LLaDA-Instruct EOS mitigation** — LLaDA-Instruct suffers from EOS overflow due to SFT padding with `<|endoftext|>`. The script auto-applies semi-autoregressive generation (smaller `block_length`) and EOS suppression (`logits_eos_inf`, `confidence_eos_eot_inf`) following the official LLaDA EVAL.md guidance. See `docs/EXPERIMENTS.md` for details.
