import numpy as np
from dataset.gsm8k import GSM8KDataset
from datasets import load_dataset


MATH_BEYOND_SYSTEM_PROMPT = """You are a math expert. You will be given a math problem to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


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

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
