import torch
import numpy as np
from datasets import load_dataset
from metrics.parsers import Parser


# ---------------------------------------------------------------------------
# Prompt modes — shared across all datasets that inherit from GSM8KDataset
#
#   base_native          — base checkpoint + plain completion prompt.
#   instruct_flat        — instruct checkpoint + same flat string as base,
#                          for apples-to-apples ablation of the checkpoint.
#                          NOTE: poor instruct_flat results may reflect
#                          prompt-format mismatch (the instruct model may
#                          never have seen flat prompts during fine-tuning),
#                          not necessarily capability loss.
#   instruct_templated   — instruct checkpoint + model-native chat template
#                          (multi-turn few-shot).
# ---------------------------------------------------------------------------
VALID_PROMPT_MODES = ("base_native", "instruct_flat", "instruct_templated")


GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        subsample=-1,
        is_base_model=False,
        prompt_mode=None,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt

        # Resolve prompt_mode: explicit value takes priority
        if prompt_mode is not None:
            if prompt_mode not in VALID_PROMPT_MODES:
                raise ValueError(
                    f"Invalid prompt_mode={prompt_mode!r}. "
                    f"Must be one of {VALID_PROMPT_MODES}"
                )
            self.prompt_mode = prompt_mode
        else:
            self.prompt_mode = "base_native" if is_base_model else "instruct_templated"

        self.is_base_model = (self.prompt_mode == "base_native")

        self.tokenizer.padding_side = "left"

        self.load_test_dataset()
        self.create_few_shot_prompt()

        _subsample_rng = np.random.RandomState(42)
        self.subsample = (
            _subsample_rng.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        print(f"Prompt mode: {self.prompt_mode}")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    # ------------------------------------------------------------------
    # Few-shot prompt construction
    # ------------------------------------------------------------------

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        train_data = load_dataset("gsm8k", "main", split="train")
        examples = np.random.choice(len(train_data), self.num_examples, replace=False)
        return [train_data[int(example)] for example in examples]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples.

        Populates:
          self.few_shot_prompt    — concatenated string for flat prompts
          self._few_shot_messages — list of (user_str, assistant_str) for
                                    multi-turn templated prompts
        """
        few_shot_examples = self.load_few_shot_examples()

        if not few_shot_examples:
            self.few_shot_prompt = ""
            self._few_shot_messages = []
            return

        formatted_examples = []
        self._few_shot_messages = []
        for example in few_shot_examples:
            input_text = example["question"]
            full_answer = example["answer"]

            gold = Parser.extract_answer_gsm8k(full_answer)

            if "####" in full_answer:
                reasoning_only = full_answer.split("####")[0].strip()
            else:
                reasoning_only = full_answer.strip()

            formatted_answer = f"<reasoning>\n{reasoning_only}\n</reasoning>\n<answer>\n\\boxed{{{gold}}}\n</answer>"
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{formatted_answer}")
            self._few_shot_messages.append((f"Question: {input_text}", formatted_answer))

        self.few_shot_prompt = "\n\n".join(formatted_examples)
        if self.num_examples > 0:
            print(f"Created {len(formatted_examples)} few-shot examples")

    # ------------------------------------------------------------------
    # 3-way prompt dispatch
    # ------------------------------------------------------------------

    def create_flat_prompt(self, input_text):
        """Flat completion prompt — used by base_native and instruct_flat.

        Same plain string regardless of model type.
        """
        if self.num_examples > 0:
            return f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        return f"{self.system_prompt}\n\nQuestion: {input_text}\nAnswer:\n"

    def create_templated_instruct_prompt(self, input_text):
        """Multi-turn chat prompt — used by instruct_templated.

        System message carries the task instruction; each few-shot example
        is a separate user/assistant turn so the model sees its native
        conversation format.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_q, asst_a in self._few_shot_messages:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": asst_a})
        messages.append({"role": "user", "content": f"Question: {input_text}\nAnswer:"})
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def create_prompt(self, input_text):
        """Dispatch to the appropriate prompt builder based on prompt_mode."""
        if self.prompt_mode in ("base_native", "instruct_flat"):
            return self.create_flat_prompt(input_text)
        return self.create_templated_instruct_prompt(input_text)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[self.subsample[idx].item()]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer

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
            "prompts": prompts
        }
