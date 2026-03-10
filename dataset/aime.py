import re

import numpy as np
from dataset.gsm8k import GSM8KDataset
from datasets import load_dataset


AIME_SYSTEM_PROMPT = """You are a math expert. You will be given a competition math problem from the American Invitational Mathematics Examination (AIME). Solve it step by step. The answer is always an integer between 000 and 999. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

_BOXED_RE = re.compile(r'\\boxed\{(\d+)\}')


def _extract_aime_answer(raw):
    """Normalise ground-truth to a plain integer string.

    AIME24 stores answers as '\\boxed{204}', AIME25 stores them as '70'.
    """
    m = _BOXED_RE.search(str(raw))
    if m:
        return m.group(1)
    return str(raw).strip()


class AIME24Dataset(GSM8KDataset):
    """AIME 2024 (30 problems).

    Competition-level math problems requiring integer answers (0-999).
    Source: https://huggingface.co/datasets/math-ai/aime24

    Schema: id, problem, solution (\\boxed{N}), url
    """

    _answer_col = "solution"

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=AIME_SYSTEM_PROMPT,
        subsample=-1,
        is_base_model=False,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample, is_base_model)

    def load_test_dataset(self):
        self.dataset = load_dataset("math-ai/aime24", split="test")
        print(f"AIME 2024: {len(self.dataset)} problems loaded")

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        train_data = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        samples = np.random.choice(range(len(train_data)), self.num_examples, replace=False)
        few_shot_examples = []
        for example_idx in samples:
            item = train_data[int(example_idx)]
            few_shot_examples.append(
                {"question": item["problem"], "answer": item["solution"]}
            )
        return few_shot_examples

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        question = item["problem"]
        raw_answer = item[self._answer_col]
        answer = _extract_aime_answer(raw_answer)
        prompt = self.create_prompt(question)
        return prompt, question, answer


class AIME25Dataset(AIME24Dataset):
    """AIME 2025 (30 problems).

    Source: https://huggingface.co/datasets/math-ai/aime25

    Schema: problem, answer (plain int), id
    """

    _answer_col = "answer"

    def load_test_dataset(self):
        self.dataset = load_dataset("math-ai/aime25", split="test")
        print(f"AIME 2025: {len(self.dataset)} problems loaded")


class AIMECombinedDataset(AIME24Dataset):
    """Combined AIME 2024 + 2025 (60 problems).

    Merges both years by normalising the different schemas:
    AIME24 has 'solution' (\\boxed{N}), AIME25 has 'answer' (plain int).
    We unify into 'problem' + 'answer' columns.
    """

    _answer_col = "answer"

    def load_test_dataset(self):
        ds24 = load_dataset("math-ai/aime24", split="test")
        ds25 = load_dataset("math-ai/aime25", split="test")

        rows = []
        for item in ds24:
            rows.append({
                "problem": item["problem"],
                "answer": _extract_aime_answer(item["solution"]),
            })
        for item in ds25:
            rows.append({
                "problem": item["problem"],
                "answer": _extract_aime_answer(item["answer"]),
            })

        from datasets import Dataset as HFDataset
        self.dataset = HFDataset.from_list(rows)
        print(f"AIME 2024+2025 combined: {len(self.dataset)} problems loaded")
