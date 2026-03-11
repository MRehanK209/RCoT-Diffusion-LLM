import re

import numpy as np
from dataset.gsm8k import GSM8KDataset
from datasets import load_dataset
from metrics.parsers import Parser


MATH_BEYOND_SYSTEM_PROMPT = """You are a math expert. You will be given a math problem to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


def _extract_boxed_for_fewshot(solution_text):
    """Extract the content inside \\boxed{} from a MATH-style solution."""
    return Parser.extract_answer_boxed(solution_text)


def _strip_boxed_from_solution(solution_text):
    """Remove the \\boxed{...} from solution text to get clean reasoning."""
    return re.sub(r'\$?\\boxed\{[^}]+\}\$?\.?', '', solution_text).strip()


class MATHBeyondDataset(GSM8KDataset):
    """MATH-Beyond Union set (181 problems).

    Problems are deliberately constructed to defeat open-source models
    up to 8B parameters even at pass@1024, making them ideal for
    studying whether additional sampling (large k) can unlock new
    reasoning capabilities.

    Source: https://huggingface.co/datasets/brendel-group/MATH-Beyond
    """

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH_BEYOND_SYSTEM_PROMPT,
        subsample=-1,
        is_base_model=False,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample, is_base_model)

    def load_test_dataset(self):
        self.dataset = load_dataset("brendel-group/MATH-Beyond", split="test")
        print(f"MATH-Beyond: {len(self.dataset)} problems loaded")

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

    def create_few_shot_prompt(self):
        """Build few-shot prompt from MATH-style solutions (\\boxed{} answers)."""
        few_shot_examples = self.load_few_shot_examples()

        if not few_shot_examples:
            self.few_shot_prompt = ""
            return

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            full_solution = example["answer"]

            gold = _extract_boxed_for_fewshot(full_solution)
            reasoning = _strip_boxed_from_solution(full_solution)

            formatted_answer = (
                f"<reasoning>\n{reasoning}\n</reasoning>\n"
                f"<answer>\n\\boxed{{{gold}}}\n</answer>"
            )
            formatted_examples.append(
                f"Question: {input_text}\nAnswer:\n{formatted_answer}"
            )

        self.few_shot_prompt = "\n\n".join(formatted_examples)
        if self.num_examples > 0:
            print(f"Created {len(formatted_examples)} few-shot examples")

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
