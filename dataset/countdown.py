import os
import re
import json
import torch
import numpy as np
from collections import Counter


# Official Dream prompt templates (from eval_planning.py)
CD_TEMPLATES = {
    "cd3": "Given 4 numbers, use +-*/ to operate over the first three numbers to achieve the last number.",
    "cd4": "Given 5 numbers, use +-*/ to operate over the first four numbers to achieve the fifth number.",
    "cd5": "Given 6 numbers, use +-*/ to operate over the first five numbers to achieve the last number.",
}

CD_FILES = {
    "cd3": "cd3_test.jsonl",
    "cd4": "cd4_test.jsonl",
    "cd5": "cd5_test.jsonl",
}


def cd_score_single(input_str, pred):
    """Score a single countdown prediction using the official Dream cd_metric.

    Args:
        input_str: comma-separated numbers string, e.g. "86,28,13,31,96"
                   (operands followed by target as last element)
        pred:      model prediction, e.g. "86+28=114,31-13=18,114-18=96"

    Returns True if the prediction is correct.
    """
    def check_eq(left_str, right_str):
        m = re.match(r'(\d+)([+\-*/])(\d+)', left_str)
        if m:
            try:
                return eval(left_str) == float(right_str)
            except Exception:
                return False
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
    """

    def __init__(
        self,
        tokenizer,
        num_examples=8,
        subsample=256,
        is_base_model=False,
        difficulty="cd3",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.is_base_model = is_base_model
        self.difficulty = difficulty

        self._load_data()
        self._build_prompt_prefix()

        n_test = len(self.test_data)
        if 0 < subsample < n_test:
            self.subsample = np.random.choice(n_test, subsample, replace=False)
        else:
            self.subsample = np.arange(n_test)

        print(f"evaluating {len(self.subsample)} examples (countdown {difficulty})")
        print(f"Model type: {'Base' if is_base_model else 'Instruct'}")

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

    def _build_prompt_prefix(self):
        """Build the prompt prefix with task description and few-shot examples."""
        template = CD_TEMPLATES[self.difficulty] + "\n\n"
        if self.num_examples > 0:
            examples = "\n\n".join(
                f"Input: {d['input']}\nOutput: {d['output']}"
                for d in self.few_shot_data
            )
            template += examples + "\n\n"
        self.prompt_prefix = template

    def create_prompt(self, input_str):
        prompt_text = self.prompt_prefix + f"Input: {input_str}\nOutput: "
        if self.is_base_model:
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

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
            prompts, padding_side="left", return_tensors="pt", padding="longest"
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
