import torch
import numpy as np
from datasets import load_dataset
from metrics.parsers import Parser

from .gsm8k import GSM8KDataset


COUNTING_LETTERS_SYSTEM_PROMPT = """You are a careful letter-counting assistant. You will be given a word and a letter, and you must count how many times that letter appears in the word.
First spell out the word letter by letter, then count the occurrences. Wrap the final count in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class CountingLettersDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=COUNTING_LETTERS_SYSTEM_PROMPT,
        subsample=-1,
        is_base_model=False,
        prompt_mode=None,
    ):
        super().__init__(
            tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
            is_base_model=is_base_model,
            prompt_mode=prompt_mode,
        )

    def load_test_dataset(self):
        self.dataset = load_dataset("mkurman/counting-letters-RL", split="test")

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        train_data = load_dataset("mkurman/counting-letters-RL", split="train")
        examples = np.random.choice(len(train_data), self.num_examples, replace=False)
        return [train_data[int(i)] for i in examples]

    def create_few_shot_prompt(self):
        few_shot_examples = self.load_few_shot_examples()

        if not few_shot_examples:
            self.few_shot_prompt = ""
            self._few_shot_messages = []
            return

        formatted_examples = []
        self._few_shot_messages = []
        for ex in few_shot_examples:
            question = ex["input"]
            word = ex["word"]
            letter = ex["letter"]
            spelling = ex["spelling"]
            count = int(ex["output"])

            reasoning_lines = [f"Let me spell out the word: {spelling}"]
            for i, ch in enumerate(word):
                if ch.lower() == letter.lower():
                    reasoning_lines.append(f"{i+1}: {ch} - match! ")
                else:
                    reasoning_lines.append(f"{i+1}: {ch} - not {letter}")
            reasoning_lines.append(f"The letter '{letter}' appears {count} time(s).")
            reasoning = "\n".join(reasoning_lines)

            formatted_answer = (
                f"<reasoning>\n{reasoning}\n</reasoning>\n"
                f"<answer>\n\\boxed{{{count}}}\n</answer>"
            )
            formatted_examples.append(f"Question: {question}\nAnswer:\n{formatted_answer}")
            self._few_shot_messages.append((f"Question: {question}", formatted_answer))

        self.few_shot_prompt = "\n\n".join(formatted_examples)
        if self.num_examples > 0:
            print(f"Created {len(formatted_examples)} few-shot examples")

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        question = item["input"]
        answer = int(item["output"])
        prompt = self.create_prompt(question)
        return prompt, question, answer
