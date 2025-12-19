import json
import os
import random
import time
import warnings
import uuid
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


from dataset.gsm8k import GSM8KDataset
from dataset.math500 import MATH500Dataset
from dataset.countdown import CTDDataset
from dataset.sudoku import SudokuDataset
from metrics.parsers import Parser

from generate_fast import load_fast_diffusion_model_and_tokenizer, build_prompt
from generate import load_diffusion_model_and_tokenizer, build_prompt
from dotenv import load_dotenv

load_dotenv()

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

def setup_device():
    """Setup single GPU."""
    if torch.cuda.is_available():
        device = 0
        torch.cuda.set_device(device)
        print(f"Using cuda:{device}")
        return device
    else:
        print("WARNING: No GPU available, using CPU")
        return -1

setup = setup_device()

torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

device = "cuda" if setup == 0 else "cpu"

def load_auto_regressive_model_and_tokenizer(model_path, device="cuda"):

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,token=os.getenv("HF_TOKEN"))
    
    # Set padding token if not already set (needed for Qwen, Llama, etc.)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = model.to(device)

    return model, tokenizer

def evaluate_auto_regressive_model(
    model_name,
    data,
    num_evals_to_use,
    few_shot,
    batch_size,
    gen_length,
    temperature,
    top_p,
    n_samples,
    device = "cuda",
    output_dir="results"
):
    """
    Evaluate an auto-regressive model on a dataset.
    
    Args:
        model: The auto-regressive model to evaluate
        model_name: Name of the model (for saving results)
        tokenizer: The tokenizer
        dataloader: DataLoader with the evaluation dataset
        gen_length: Maximum number of new tokens to generate
        few_shot: Number of few-shot examples used
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        n_samples: Number of samples to generate per question
        output_dir: Directory to save results
    """
    model, tokenizer = load_auto_regressive_model_and_tokenizer(model_name, device)
    model.eval()

    if "base" not in model_name.lower() and "instruct" not in model_name.lower():
        model_name = model_name + "-base"
    is_base_model = 'base' in model_name.lower()
    
    dataset = DATASET_MAP[data](
            tokenizer,
            subsample=num_evals_to_use,
            num_examples=few_shot,
            is_base_model=is_base_model,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    total_processed = 0
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, desc=f"Evaluating (total samples: {len(dataloader.dataset)})"):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size)]
        raw_generations = [[] for _ in range(batch_size)]
        all_extracted_answers = [[] for _ in range(batch_size)]
        
        # Generate n_samples for each question in the batch
        for sample_idx in range(n_samples):
            # Auto-regressive generation
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Slice only the generated tokens (after the input)
            input_length = input_ids.shape[1]
            generated_texts = tokenizer.batch_decode(out[:, input_length:], skip_special_tokens=False)
            
            # Process each generation in the batch
            for j, text in enumerate(generated_texts):
                # Truncate at EOS tokens
                raw_generations[j].append(text)
                eos_markers = ['<|endoftext|>', '<|end|>', '<|im_end|>', '</s>', '<|eot_id|>']
                for marker in eos_markers:
                    if marker in text:
                        text = text.split(marker)[0]
                        break
                all_cleaned_texts[j].append(text)
                
                # Extract the boxed answer from the generation
                try:
                    extracted_answer = Parser.extract_answer_boxed(text)
                    # Try to convert to float for numerical answers
                    try:
                        extracted_answer = float(extracted_answer)
                    except (ValueError, TypeError):
                        # Keep as string if not a number
                        pass
                except Exception as e:
                    extracted_answer = None
                all_extracted_answers[j].append(extracted_answer)
        
        # Create results with lists of generations (or single value if n_samples=1)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": all_cleaned_texts[j] if n_samples > 1 else all_cleaned_texts[j][0],
                "extracted_answer": all_extracted_answers[j] if n_samples > 1 else all_extracted_answers[j][0],
                "ground_truth": gt_answers[j],
            }
            for j in range(batch_size)
        ]
        
        # Store results in memory
        all_generations.extend(example_result)
        
        total_processed += batch_size * n_samples
        wall_times.append(time.time() - start_time)

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed,
    }
    
    # Save results to file
    model_name = model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name}_{gen_length}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_ar.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "total_processed": metrics["total_processed"],
                },
                "generations": metrics["generations"],
            },
            f,
            indent=2,
        )
    print(f"Saved generations to {filename}")

    del model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

def evaluate_dllm(
    diffusion_model_name,
    data = "gsm8k",
    num_evals_to_use = 100,
    few_shot = 0,
    batch_size = 16,
    gen_length = 128,
    diffusion_steps = 64,
    temperature = 0.2,
    cfg_scale = 0.0,
    steps = 64,
    block_length = 32,
    remasking = "low_confidence",
    alg = "entropy",
    alg_temp = 0.0,
    top_p = 0.95,
    top_k = None,
    n_samples = 1,
    device = "cuda",
    output_dir = "results"
    ):

    diffusion_model, tokenizer = load_diffusion_model_and_tokenizer(diffusion_model_name, device)
    diffusion_model.eval()
    device = diffusion_model.device

    if "base" not in diffusion_model_name.lower() and "instruct" not in diffusion_model_name.lower():
        diffusion_model_name = diffusion_model_name + "-base"
    is_base_model = 'base' in diffusion_model_name.lower()
    
    dataset = DATASET_MAP[data](
            tokenizer,
            subsample=num_evals_to_use,
            num_examples=few_shot,
            is_base_model=is_base_model,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    total_processed = 0
    wall_times = []
    all_generations = []

    for batch in tqdm(dataloader, desc=f"Evaluating (total samples: {len(dataloader.dataset)})"):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size)]
        raw_generations = [[] for _ in range(batch_size)]
        all_extracted_answers = [[] for _ in range(batch_size)]
        
        # Generate n_samples for each question in the batch
        for sample_idx in range(n_samples):
            out = diffusion_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                temperature=temperature,
                num_diffusion_steps=steps,
                steps=steps,  # Dream uses 'steps'
                block_length=block_length,  # LLaDA only
                cfg_scale=cfg_scale,  # LLaDA only
                remasking=remasking,  # LLaDA only
                logits_eos_inf=True,  # LLaDA: Prevent early EOS during diffusion
                confidence_eos_eot_inf=False,  # LLaDA: Allow EOS in confidence calculation
                alg=alg,  # Dream only
                alg_temp=alg_temp,  # Dream only
                top_p=top_p,  # Dream only
                top_k=top_k,  # Dream only
            )

            # Slice only the generated tokens (after the input)
            input_length = input_ids.shape[1]
            generated_texts = tokenizer.batch_decode(out[:, input_length:], skip_special_tokens=False)
            
            # Process each generation in the batch
            for j, text in enumerate(generated_texts):
                # Truncate at EOS tokens
                raw_generations[j].append(text)
                eos_markers = ['<|endoftext|>', '<|end|>', '<|im_end|>', '</s>', '<|eot_id|>']
                for marker in eos_markers:
                    if marker in text:
                        text = text.split(marker)[0]
                        break
                all_cleaned_texts[j].append(text)
                
                # Extract the boxed answer from the generation
                try:
                    extracted_answer = Parser.extract_answer_boxed(text)
                    # Try to convert to float for numerical answers
                    try:
                        extracted_answer = float(extracted_answer)
                    except (ValueError, TypeError):
                        # Keep as string if not a number
                        pass
                except Exception as e:
                    extracted_answer = None
                all_extracted_answers[j].append(extracted_answer)
        
        # Create results with lists of generations (or single value if n_samples=1)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": all_cleaned_texts[j] if n_samples > 1 else all_cleaned_texts[j][0],
                "extracted_answer": all_extracted_answers[j] if n_samples > 1 else all_extracted_answers[j][0],
                "ground_truth": gt_answers[j],
            }
            for j in range(batch_size)
        ]
        
        # Store results in memory
        all_generations.extend(example_result)
        
        total_processed += batch_size * n_samples
        wall_times.append(time.time() - start_time)

        idx = random.randint(0, len(questions) - 1)

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed,
    }
    model_name = diffusion_model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name}_{gen_length}_{diffusion_steps}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_testing.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "total_processed": metrics["total_processed"],
                },
                "generations": metrics["generations"],
            },
            f,
            indent=2,
        )
    print(f"Saved generations to {filename}")
    
    del diffusion_model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

def evaluate_fast_dllm(
    diffusion_model_name,
    data = "gsm8k",
    num_evals_to_use = 100,
    few_shot = 0,
    batch_size = 16,
    gen_length = 128,
    diffusion_steps = 64,
    temperature = 0.2,
    cfg_scale = 0.0,
    steps = 64,
    block_length = 32,
    remasking = "low_confidence",
    alg = "confidence_threshold",
    alg_temp = 0.0,
    top_p = 0.95,
    top_k = None,
    n_samples = 1,
    use_cache=True,
    dual_cache=True,
    threshold = None,
    factor = None,
    output_dir = "results",
    device = "cuda",
    ):

    diffusion_model, tokenizer = load_fast_diffusion_model_and_tokenizer(diffusion_model_name, device)
    diffusion_model.eval()
    device = diffusion_model.device

    if "base" not in diffusion_model_name.lower() and "instruct" not in diffusion_model_name.lower():
        diffusion_model_name = diffusion_model_name + "-base"
    is_base_model = 'base' in diffusion_model_name.lower()

    dataset = DATASET_MAP[data](
            tokenizer,
            subsample=num_evals_to_use,
            num_examples=few_shot,
            is_base_model=is_base_model,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    total_processed = 0
    wall_times = []
    all_generations = []

    for batch in tqdm(dataloader, desc=f"Evaluating (total samples: {len(dataloader.dataset)})"):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        batch_size = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size)]
        raw_generations = [[] for _ in range(batch_size)]
        all_extracted_answers = [[] for _ in range(batch_size)]
        
        # Generate n_samples for each question in the batch
        for sample_idx in range(n_samples):
            out = diffusion_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_length,
                temperature=temperature,
                num_diffusion_steps=steps,
                steps=steps,  # Dream uses 'steps'
                block_length=block_length,  # LLaDA only
                cfg_scale=cfg_scale,  # LLaDA only
                remasking=remasking,  # LLaDA only
                logits_eos_inf=True,  # LLaDA: Prevent early EOS during diffusion
                confidence_eos_eot_inf=False,  # LLaDA: Allow EOS in confidence calculation
                alg=alg,  # Dream only
                alg_temp=alg_temp,  # Dream only
                top_p=top_p,  # Dream only
                top_k=top_k,  # Dream only
                use_cache=use_cache,
                dual_cache=dual_cache,
                threshold=threshold,
                factor=factor,
            )

            # Slice only the generated tokens (after the input)
            input_length = input_ids.shape[1]
            generated_texts = tokenizer.batch_decode(out[:, input_length:], skip_special_tokens=False)
            
            # Process each generation in the batch
            for j, text in enumerate(generated_texts):
                # Truncate at EOS tokens
                raw_generations[j].append(text)
                eos_markers = ['<|endoftext|>', '<|end|>', '<|im_end|>', '</s>', '<|eot_id|>']
                for marker in eos_markers:
                    if marker in text:
                        text = text.split(marker)[0]
                        break
                all_cleaned_texts[j].append(text)
                
                # Extract the boxed answer from the generation
                try:
                    extracted_answer = Parser.extract_answer_boxed(text)
                    # Try to convert to float for numerical answers
                    try:
                        extracted_answer = float(extracted_answer)
                    except (ValueError, TypeError):
                        # Keep as string if not a number
                        pass
                except Exception as e:
                    extracted_answer = None
                all_extracted_answers[j].append(extracted_answer)
        
        # Create results with lists of generations (or single value if n_samples=1)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": all_cleaned_texts[j] if n_samples > 1 else all_cleaned_texts[j][0],
                "raw_generations": raw_generations[j] if n_samples > 1 else raw_generations[j][0],
                "extracted_answer": all_extracted_answers[j] if n_samples > 1 else all_extracted_answers[j][0],
                "ground_truth": gt_answers[j],
            }
            for j in range(batch_size)
        ]
        
        # Store results in memory
        all_generations.extend(example_result)
        
        total_processed += batch_size * n_samples
        wall_times.append(time.time() - start_time)

        idx = random.randint(0, len(questions) - 1)

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed,
    }
    model_name = diffusion_model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name}_{gen_length}_{diffusion_steps}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_testing_fast_dllm.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "total_processed": metrics["total_processed"],
                },
                "generations": metrics["generations"],
            },
            f,
            indent=2,
        )
    print(f"Saved generations to {filename}")

    del diffusion_model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_model_name", default="GSAI-ML/LLaDA-8B-Base", type=str)
    args = parser.parse_args()

    diffusion_model_name = args.diffusion_model_name
    data = "gsm8k"
    num_evals_to_use = 256
    few_shot = 4
    batch_size = 1
    gen_length = 256
    diffusion_steps = 256
    temperature = 0.0
    cfg_scale = 0.0
    steps = 256
    block_length = 32
    remasking = "low_confidence"
    alg = "entropy"
    alg_temp = 0.0
    top_p = 0.95
    top_k = None
    n_samples = 1


    metrics = evaluate_dllm(
        diffusion_model_name, data, num_evals_to_use, few_shot, batch_size, gen_length, diffusion_steps, temperature, cfg_scale, steps, block_length, remasking, alg, alg_temp, top_p, top_k, n_samples
        )

if __name__ == "__main__":
    main()
