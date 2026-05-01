import os
import re
import json
import torch
import numpy as np
from collections import Counter


# Official Dream task descriptions, with a stricter instruct-mode suffix to
# discourage self-correction chatter on AR chat checkpoints.
_CD_TASKS = {
    "cd3": "Given 4 numbers, use +-*/ to operate over the first three numbers to achieve the last number.",
    "cd4": "Given 5 numbers, use +-*/ to operate over the first four numbers to achieve the fifth number.",
    "cd5": "Given 6 numbers, use +-*/ to operate over the first five numbers to achieve the last number.",
}
_CD_BASE_FORMAT = " Output ONLY comma-separated equations like a+b=c,c-d=e with no explanation."
_CD_INSTRUCT_FORMAT = (
    " Output ONLY {num_equations} comma-separated equations like a+b=c,c-d=e."
    " No words, no explanation, no retries, no comments."
    " Stop immediately after the final equation reaches the target."
)

CD_FILES = {
    "cd3": "cd3_test.jsonl",
    "cd4": "cd4_test.jsonl",
    "cd5": "cd5_test.jsonl",
}

# ---------------------------------------------------------------------------
# Prompt modes for 3-way comparison
#
#   base_native          — base checkpoint + plain completion prompt.
#   instruct_flat        — instruct checkpoint + same flat string as base.
#                          Isolates the effect of instruction tuning on the
#                          weights while holding the prompt format constant.
#                          NOTE: this does NOT purely measure "instruction
#                          tuning quality" — an instruct model that was
#                          fine-tuned exclusively on chat-formatted data may
#                          degrade on flat prompts, so poor instruct_flat
#                          performance can reflect prompt-format mismatch
#                          rather than capability loss.
#   instruct_templated   — instruct checkpoint + model-native chat template
#                          with each few-shot example serialized as its own
#                          user/assistant turn.
#   instruct_single_turn — instruct checkpoint + model-native chat template
#                          with the entire flat prompt packed into one user
#                          message. This is often closer to standard chat-eval
#                          serialization than synthetic multi-turn few-shot.
#
# Comparing (instruct_flat vs base_native) reveals how much flat-text
# capability the instruct weights retained.
# Comparing (instruct_templated vs instruct_flat) reveals how much the
# chat template helps (or is necessary) for the instruct checkpoint.
# ---------------------------------------------------------------------------
VALID_PROMPT_MODES = (
    "base_native",
    "instruct_flat",
    "instruct_templated",
    "instruct_single_turn",
)

_COUNTDOWN_EOS_MARKERS = (
    "<|endoftext|>",
    "<|eot_id|>",
    "<|im_end|>",
    "</s>",
)
_COUNTDOWN_BINARY_EQ_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def _required_equation_count(difficulty):
    return {"cd3": 2, "cd4": 3, "cd5": 4}[difficulty]


def _countdown_instruction(difficulty, strict=False):
    task = _CD_TASKS[difficulty]
    if strict:
        return task + _CD_INSTRUCT_FORMAT.format(
            num_equations=_required_equation_count(difficulty)
        )
    return task + _CD_BASE_FORMAT


def _normalize_countdown_prediction(pred):
    """Keep the first logical countdown line and strip common chat/eos artifacts."""
    if pred is None:
        return ""

    cleaned = (
        str(pred)
        .replace(r"\div", "/")
        .replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
    )
    lines = cleaned.splitlines()
    cleaned = lines[0].strip() if lines else cleaned.strip()

    for marker in _COUNTDOWN_EOS_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    return cleaned


def _extract_leading_subequations(pred):
    """Recover the leading comma-separated equation chain before chatter starts."""
    normalized = _normalize_countdown_prediction(pred)
    if not normalized:
        return ""

    subequations = []
    for raw_subeq in normalized.split(","):
        subeq = raw_subeq.strip()
        if not subeq:
            break

        match = _COUNTDOWN_BINARY_EQ_RE.fullmatch(subeq)
        if not match:
            break

        left_a, op, left_b, right = match.groups()
        subequations.append(f"{left_a}{op}{left_b}={right}")

    return ",".join(subequations)


def cd_score_single(input_str, pred):
    """Score a single countdown prediction using the official Dream cd_metric.

    Args:
        input_str: comma-separated numbers string, e.g. "86,28,13,31,96"
                   (operands followed by target as last element)
        pred:      model prediction, e.g. "86+28=114,31-13=18,114-18=96"

    Returns True if the prediction is correct.
    """
    def check_eq(left_str, right_str):
        left_match = _COUNTDOWN_BINARY_EQ_RE.fullmatch(f"{left_str}={right_str}")
        if not left_match:
            return False

        try:
            return eval(left_str, {"__builtins__": None}, {}) == float(right_str)
        except Exception:
            return False

    pred = _extract_leading_subequations(pred)
    if not pred:
        return False

    subequations = pred.split(',')
    match = True
    query_numbers = Counter(input_str.split(',')[:-1])

    for subeq in subequations:
        try:
            left, right = subeq.split('=')
            match &= check_eq(left.strip(), right.strip())
            left_side_numbers = re.findall(r'(\d+)(?=[+\-*/=])', subeq)
            query_numbers.subtract(left_side_numbers)
            query_numbers.update({right.strip(): 1})
        except Exception:
            match = False
        if not match:
            break

    answer = input_str.split(',')[-1].strip()
    pred_ans = pred.split('=')[-1].strip()

    query_numbers.subtract({answer: 1})
    numbers_match = all(v == 0 for v in query_numbers.values())

    return match and (answer == pred_ans) and numbers_match


class CTDDataset(torch.utils.data.Dataset):
    """Countdown dataset matching the official Dream evaluation.

    Uses the exact same prompt format, data files, and few-shot structure
    as https://github.com/DreamLM/Dream/blob/main/eval/eval_planning.py

    Supports cd3 (3 operands), cd4 (4 operands), cd5 (5 operands).
    The first `num_examples` entries from the data file are used as
    few-shot demonstrations (official default: 8).

    Prompt modes
    ------------
    prompt_mode controls how the prompt string is serialized:
      - "base_native"        : flat completion prompt (for base checkpoints)
      - "instruct_flat"      : same flat string, no chat template (ablation)
      - "instruct_templated" : multi-turn chat template (for instruct models)
      - "instruct_single_turn": one user chat turn containing the full prompt

    If prompt_mode is not provided, it is inferred from is_base_model
    for backward compatibility.
    """

    def __init__(
        self,
        tokenizer,
        num_examples=8,
        subsample=256,
        is_base_model=False,
        difficulty="cd3",
        prompt_mode=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.difficulty = difficulty

        # Resolve prompt_mode: explicit value takes priority, otherwise
        # infer from the legacy is_base_model flag.
        if prompt_mode is not None:
            if prompt_mode not in VALID_PROMPT_MODES:
                raise ValueError(
                    f"Invalid prompt_mode={prompt_mode!r}. "
                    f"Must be one of {VALID_PROMPT_MODES}"
                )
            self.prompt_mode = prompt_mode
        else:
            self.prompt_mode = "base_native" if is_base_model else "instruct_templated"

        # Keep is_base_model as a derived attribute for backward compat
        # with any external code that reads it.
        self.is_base_model = (self.prompt_mode == "base_native")

        # Set padding_side on the tokenizer object rather than passing it
        # per-call, since some tokenizers don't honor it as a kwarg.
        self.tokenizer.padding_side = "left"

        self._load_data()
        self._build_prompt_prefix()

        # _subsample_rng = np.random.RandomState(42)
        n_test = len(self.test_data)
        if 0 < subsample < n_test:
            self.subsample = np.random.choice(n_test, subsample, replace=False)
            # self.subsample = _subsample_rng.choice(n_test, subsample, replace=False)
        else:
            self.subsample = np.arange(n_test)

        print(f"evaluating {len(self.subsample)} examples (countdown {difficulty})")
        print(f"Prompt mode: {self.prompt_mode}")

    # ------------------------------------------------------------------
    # Data loading (unchanged)
    # ------------------------------------------------------------------

    def _load_data(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(cur_path, CD_FILES[self.difficulty])
        all_data = []
        with open(data_file, "r") as f:
            for line in f:
                all_data.append(json.loads(line))

        self.few_shot_data = all_data[: self.num_examples]
        self.test_data = all_data[self.num_examples :]
        print(
            f"Countdown {self.difficulty}: {len(all_data)} total, "
            f"{self.num_examples} few-shot, {len(self.test_data)} test"
        )

    # ------------------------------------------------------------------
    # Prompt construction (unchanged)
    # ------------------------------------------------------------------

    def _build_prompt_prefix(self):
        """Build the flat prompt prefix with task description and few-shot examples.

        Used by both base_native and instruct_flat modes.
        """
        strict = self.prompt_mode != "base_native"
        template = _countdown_instruction(self.difficulty, strict=strict) + "\n\n"
        if self.num_examples > 0:
            examples = "\n\n".join(
                f"Input: {d['input']}\nOutput: {d['output']}"
                for d in self.few_shot_data
            )
            template += examples + "\n\n"
        self.prompt_prefix = template

    # ------------------------------------------------------------------
    # 3-way prompt dispatch
    # ------------------------------------------------------------------

    def create_flat_prompt(self, input_str):
        """Flat completion prompt — used by base_native and instruct_flat.

        Returns a plain string with the task description, few-shot examples,
        and the final Input:/Output: pair. No chat template is applied.
        """
        return self.prompt_prefix + f"Input: {input_str}\nOutput: "

    def create_templated_instruct_prompt(self, input_str):
        """Multi-turn chat prompt — used by instruct_templated.

        Structures the same content as the flat prompt into the model's
        native chat template:
          system  = task description
          user_i  = "Input: {numbers}"
          asst_i  = "{equations}"   (for each few-shot example)
          user_N  = "Input: {test_numbers}"   (final turn, model generates)
        """
        messages = [
            {
                "role": "system",
                "content": _countdown_instruction(self.difficulty, strict=True),
            },
        ]
        for d in self.few_shot_data:
            messages.append({"role": "user", "content": f"Input: {d['input']}"})
            messages.append({"role": "assistant", "content": d["output"]})
        messages.append({"role": "user", "content": f"Input: {input_str}"})
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def create_single_turn_instruct_prompt(self, input_str):
        """Single-turn chat prompt with the full flat prompt in one user turn.

        This avoids fabricating a multi-turn conversation out of the few-shot
        examples and is usually closer to how chat checkpoints are evaluated in
        standard chat-generation benchmarks.
        """
        prompt = self.create_flat_prompt(input_str).rstrip()
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def create_prompt(self, input_str):
        """Dispatch to the appropriate prompt builder based on prompt_mode."""
        if self.prompt_mode in ("base_native", "instruct_flat"):
            return self.create_flat_prompt(input_str)
        if self.prompt_mode == "instruct_single_turn":
            return self.create_single_turn_instruct_prompt(input_str)
        return self.create_templated_instruct_prompt(input_str)

    # ------------------------------------------------------------------
    # Dataset interface (unchanged)
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.subsample)

    def __getitem__(self, idx):
        item = self.test_data[self.subsample[idx].item()]
        input_str = item["input"]
        prompt = self.create_prompt(input_str)
        # Return input_str as both question and answer context for cd_score_single
        return prompt, input_str, input_str

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
        }


class CTD4Dataset(CTDDataset):
    """Countdown cd4 (4 operands) — the variant used in the official Dream paper."""

    def __init__(self, tokenizer, **kwargs):
        kwargs.setdefault("difficulty", "cd4")
        super().__init__(tokenizer, **kwargs)


class CTD5Dataset(CTDDataset):
    """Countdown cd5 (5 operands) — hardest variant."""

    def __init__(self, tokenizer, **kwargs):
        kwargs.setdefault("difficulty", "cd5")
        super().__init__(tokenizer, **kwargs)


class CTDLegacyDataset(CTDDataset):
    """Countdown using the original countdown_cd3_test.jsonl data.

    That file has a different format: {"input": "30,100,93", "output": "23"}
    where input has only the operand numbers and output is just the target.
    This class converts it to the official format on-the-fly so the same
    prompt template and cd_score_single scoring work unchanged.

    Few-shot examples are sourced from the official cd3_test.jsonl (which
    has solution steps).
    """

    def __init__(self, tokenizer, **kwargs):
        kwargs.setdefault("difficulty", "cd3")
        super().__init__(tokenizer, **kwargs)

    def _load_data(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))

        # Few-shot examples from official cd3 data (has solution steps)
        cd3_file = os.path.join(cur_path, CD_FILES["cd3"])
        cd3_data = []
        with open(cd3_file, "r") as f:
            for line in f:
                cd3_data.append(json.loads(line))
        self.few_shot_data = cd3_data[: self.num_examples]

        # Test data from the legacy file (different format)
        legacy_file = os.path.join(cur_path, "countdown_cd3_test.jsonl")
        raw = []
        with open(legacy_file, "r") as f:
            for line in f:
                raw.append(json.loads(line))

        # Convert to official format: input = "num1,num2,num3,target"
        self.test_data = []
        for item in raw:
            combined_input = f"{item['input']},{item['output']}"
            self.test_data.append({"input": combined_input, "output": ""})

        print(
            f"Countdown legacy: {len(raw)} test, "
            f"{self.num_examples} few-shot (from official cd3)"
        )
