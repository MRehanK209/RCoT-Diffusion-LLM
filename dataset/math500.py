import torch

import random
from dataset.gsm8k import GSM8KDataset
from datasets import load_dataset
from metrics.parsers import Parser, is_equiv


MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a math problem to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
        is_base_model=False,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample, is_base_model)

    def load_test_dataset(self):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def load_few_shot_examples(self):
        """Load few-shot examples from MATH dataset."""
        if self.num_examples <= 0:
            return []
        train_data = load_dataset("EleutherAI/hendrycks_math", ("algebra"), split="train")
        samples = random.sample(range(len(train_data)), self.num_examples)
        few_shot_examples = []
        for example_idx in samples:
            problem = train_data[example_idx]["problem"]
            solution = train_data[example_idx]["solution"]
            # Format to match GSM8K structure (question/answer)
            few_shot_examples.append(
                {"question": problem, "answer": solution}
            )
        return few_shot_examples

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer