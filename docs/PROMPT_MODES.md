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
| `instruct_templated` | Instruct | Model-native chat template (multi-turn) | Recommended reporting mode for instruct checkpoints. |
| `instruct_single_turn` | Instruct | Model-native chat template (single user turn) | Diagnostic parity check, not a primary benchmark mode. |

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
    prompt_mode="instruct_flat",  # or "base_native" / "instruct_templated" / "instruct_single_turn"
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

**`instruct_single_turn`** — Single-turn chat template:
```
[user]   Given 5 numbers, use +-*/ to operate over the first four numbers
         to achieve the fifth number. Output ONLY comma-separated equations
         like a+b=c,c-d=e with no explanation.

         Input: 86,28,13,31,96
         Output: 86+28=114,31-13=18,114-18=96

         ...

         Input: {test_numbers}
         Output:
```

#### Trip Planning

Trip planning uses a domain-specific `TASK:` format that doesn't decompose cleanly into user/assistant turns. All three modes keep the same prompt content; only the wrapping differs:

- **`base_native` / `instruct_flat`**: flat prompt + `\nSOLUTION: ` suffix
- **`instruct_templated`**: flat prompt wrapped in a single user message with chat template

### Evaluation Functions

All four evaluation functions (`evaluate_fast_dllm`, `evaluate_dllm`, `evaluate_vllm_model`, `evaluate_auto_regressive_model`) accept `prompt_mode=None`. When `None`, the dataset infers the mode from `is_base_model`.

### Result Filenames

The filename convention now makes the resolved prompt mode explicit for every run:

- `base_native`: `...generations_base_native_fast_dllm.json`
- `instruct_templated`: `...generations_instruct_templated_fast_dllm.json`
- `instruct_flat`: `...generations_instruct_flat_fast_dllm.json`
- `instruct_single_turn`: `...generations_instruct_single_turn_fast_dllm.json`

The same convention is used for `fast_dllm`, `dllm`, `vllm`, and `ar`.
This means newly launched runs will not reuse older result files that were saved
under the previous implicit naming scheme.

## CLI Usage

### `run_evaluation.sh`

#### Single-mode run
```bash
# Standard base + instruct (templated) — same as before
./run_evaluation.sh -E passk -d countdown_cd4

# Instruct-flat ablation only
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_flat

# Single-turn chat parity check
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_single_turn

# Run multiple instruct prompt modes in one command
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_templated --prompt_mode instruct_flat

# Comma-separated form also works
./run_evaluation.sh -E passk -d countdown_cd4 -v instruct --prompt_mode instruct_templated,instruct_flat
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
| `--prompt_mode` | `auto`, `instruct_templated`, `instruct_flat`, `instruct_single_turn` | `auto` | Override prompt format for instruct models; may be repeated or passed as a comma-separated list |
| `-v, --variant` | `base`, `instruct`, `all`, `all3` | `all` | `all3` = base + instruct + instruct_flat |

## What To Look For On Countdown CD4

For countdown, the three prompt modes are best understood as:

- `instruct_templated` vs `instruct_single_turn`
- `instruct_single_turn` vs `instruct_flat`

That separates three different effects:

1. Whether the checkpoint needs chat wrapping at all.
2. Whether it benefits from a synthetic multi-turn few-shot conversation.
3. Whether the flat Dream-style prompt is simply a distribution mismatch.

Recent local `countdown_cd4` runs suggest EOS handling is not the main limiter for LLaDA-Instruct, so prompt serialization and task/checkpoint mismatch matter more than EOS suppression here.

## Recommended Reporting Mode

For the main countdown benchmark, this repo now treats `instruct_templated` as the
default instruct setting to keep and report.

Why:

1. It is the checkpoint-native evaluation mode for instruct models in this repo.
2. It performed best or tied-best for most instruct checkpoints we tested on countdown:
   - LLaDA-Instruct: `instruct_templated` > `instruct_flat` > `instruct_single_turn`
   - Qwen-Instruct: `instruct_templated` > `instruct_flat`
   - Llama-Instruct: `instruct_templated` ~= `instruct_flat`, with a slight edge to templated
3. `instruct_flat` is still valuable academically, but as an ablation that isolates
   the effect of instruction tuning under an identical flat prompt.
4. `instruct_single_turn` did not help on countdown and is now best treated as a
   diagnostic rather than a primary benchmark mode.

Dream-Instruct is the notable exception: on local countdown runs, `instruct_flat`
outperformed `instruct_templated`. That is useful analysis, but for a single
cross-model instruct benchmark we prefer one consistent rule, and `instruct_templated`
is the cleanest checkpoint-native choice.

## Methodological Notes

1. **Identical few-shot content** — All three modes use the same few-shot examples (same data, same count). The only difference is how the content is serialized (flat string vs. multi-turn messages).

2. **Identical generation parameters** — Temperature, max tokens, top-p, and diffusion parameters are the same across modes for fair comparison.

3. **Identical evaluation questions** — The reproducible subsampling (seed=42) ensures the same test questions are used regardless of prompt mode.

4. **Apples-to-apples** — The `instruct_flat` mode uses the *exact same string* as `base_native`. This ensures any performance difference is attributable solely to the model weights, not the prompt format.

5. **Why LLaDA-Instruct is still low on countdown** — The remaining gap is not primarily an EOS problem in local countdown runs. With `block_length=32`, outputs are usually non-empty and equation-shaped, but many are arithmetically invalid. That points more toward prompt/task mismatch and symbolic-consistency limits than early-EOS collapse.
