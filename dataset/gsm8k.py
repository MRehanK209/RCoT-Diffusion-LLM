import torch
import numpy as np
import random
from datasets import load_dataset
from metrics.parsers import Parser, is_equiv


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
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.is_base_model = is_base_model
        self.load_test_dataset()
        self.create_few_shot_prompt()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        print(f"Model type: {'Base' if is_base_model else 'Instruct'}")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def create_prompt(self, input_text):
        if self.is_base_model:
            # Base model: Plain text format without chat template
            # Don't prefill <reasoning> for base models - let them learn from examples
            if self.num_examples > 0:
                prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
            else:
                # Base models need examples! But if no examples, at least show format
                prompt = f"{self.system_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
            return prompt
        else:
            # Instruct model: Use chat template
            # Format the question properly
            if self.num_examples > 0:
                # With few-shot examples
                question_with_examples = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:"
            else:
                # Zero-shot: just the question
                question_with_examples = f"Question: {input_text}\nAnswer:"
            
            # Create messages with system prompt
            messages = [{"role": "user", "content": self.system_prompt + "\n\n" + question_with_examples}]
            user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # DON'T prefill <reasoning> - let the model generate it naturally
            # The system prompt already instructs the format
            return user_input

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        train_data = load_dataset("gsm8k", "main", split="train")
        examples = random.sample(range(len(train_data)), self.num_examples)
        return [train_data[example] for example in examples]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()
        
        if not few_shot_examples:
            self.few_shot_prompt = ""
            return

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            full_answer = example["answer"]
            
            # Extract the numeric answer and format with \boxed{}
            gold = Parser.extract_answer_gsm8k(full_answer)
            
            # Get reasoning part (before ####)
            if "####" in full_answer:
                reasoning_only = full_answer.split("####")[0].strip()
            else:
                reasoning_only = full_answer.strip()
            
            # Format answer with <reasoning> tags and \boxed{} for final answer
            formatted_answer = f"<reasoning>\n{reasoning_only}\n</reasoning>\n<answer>\n\\boxed{{{gold}}}\n</answer>"
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{formatted_answer}")
        
        self.few_shot_prompt = "\n\n".join(formatted_examples)
        if self.num_examples > 0:
            print(f"Created {len(formatted_examples)} few-shot examples")

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
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "questions": questions,
            "answers": answers,
            "prompts": prompts
        }