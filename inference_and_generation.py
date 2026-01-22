import json
import os
import random
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


from dataset.gsm8k import GSM8KDataset
from dataset.math500 import MATH500Dataset
from dataset.countdown import CTDDataset
from dataset.sudoku import SudokuDataset
from metrics.parsers import Parser

from generate_fast import load_fast_diffusion_model_and_tokenizer
from generate import load_diffusion_model_and_tokenizer
from dotenv import load_dotenv
from evaluate_pass_k import compute_metrics

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
    Evaluate an auto-regressive model on a dataset with incremental saving and resume capability.
    """
    # Determine output filename BEFORE loading model
    model_name_clean = model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name_clean}_{gen_length}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_ar.json"
    
    # Check for existing results and resume capability
    all_generations = []
    processed_questions = set()
    
    if os.path.exists(filename):
        print(f"\n{'='*80}")
        print(f"FOUND EXISTING RESULTS: {filename}")
        print(f"{'='*80}")
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
                all_generations = existing_data.get('generations', [])
                for gen in all_generations:
                    processed_questions.add(gen.get('question', ''))
                print(f"Loaded {len(all_generations)} existing generations")
                print(f"Will resume from question {len(all_generations) + 1}")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting fresh...")
            all_generations = []
            processed_questions = set()
    else:
        print(f"\n{'='*80}")
        print(f"STARTING NEW EVALUATION")
        print(f"Output will be saved to: {filename}")
        print(f"{'='*80}")

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
        shuffle=False,  # CRITICAL: Don't shuffle to maintain reproducibility
        collate_fn=dataset.collate_fn,
    )

    total_processed = len(all_generations) * n_samples if all_generations else 0
    wall_times = []
    device = model.device
    skipped_count = 0
    questions_remaining = len(dataset) - len(all_generations)

    # Create progress bar that shows actual questions processed, not batch iteration
    pbar = tqdm(total=questions_remaining, 
                desc=f"Processing remaining questions",
                initial=0)

    for batch in dataloader:
        # Check if this batch should be skipped (already processed)
        questions = batch["questions"]
        batch_questions_to_process = []
        batch_indices_to_process = []
        
        for j, question in enumerate(questions):
            if question not in processed_questions:
                batch_questions_to_process.append(question)
                batch_indices_to_process.append(j)
        
        if not batch_questions_to_process:
            skipped_count += len(questions)
            continue
        
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        prompts = batch["prompts"]

        batch_size_actual = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size_actual)]
        raw_generations = [[] for _ in range(batch_size_actual)]
        all_extracted_answers = [[] for _ in range(batch_size_actual)]
        
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
            for j in range(batch_size_actual)
        ]
        
        # Only add results for questions we actually processed
        for j, result in enumerate(example_result):
            if j in batch_indices_to_process:
                all_generations.append(result)
                processed_questions.add(result["question"])
        
        total_processed += len(batch_indices_to_process) * n_samples
        wall_times.append(time.time() - start_time)
        
        # Update progress bar for each question processed
        pbar.update(len(batch_indices_to_process))

        # CRITICAL: Save after EVERY batch
        avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
        with open(filename, "w") as f:
            json.dump(
                {
                    "metrics": {
                        "wall_time": avg_wall_time,
                        "total_processed": total_processed,
                        "num_completed": len(all_generations),
                        "num_remaining": len(dataset) - len(all_generations),
                    },
                    "generations": all_generations,
                },
                f,
                indent=2,
            )

    pbar.close()
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} already-processed questions")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"Total generations: {len(all_generations)}")
    print(f"Saved to: {filename}")
    print(f"{'='*80}")
    
    del model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return filename

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

    # Determine output filename BEFORE loading model
    model_name_clean = diffusion_model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name_clean}_{gen_length}_{diffusion_steps}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_testing.json"
    
    # Check for existing results and resume capability
    all_generations = []
    processed_questions = set()
    
    if os.path.exists(filename):
        print(f"\n{'='*80}")
        print(f"FOUND EXISTING RESULTS: {filename}")
        print(f"{'='*80}")
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
                all_generations = existing_data.get('generations', [])
                for gen in all_generations:
                    processed_questions.add(gen.get('question', ''))
                print(f"Loaded {len(all_generations)} existing generations")
                print(f"Will resume from question {len(all_generations) + 1}")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting fresh...")
            all_generations = []
            processed_questions = set()
    else:
        print(f"\n{'='*80}")
        print(f"STARTING NEW EVALUATION")
        print(f"Output will be saved to: {filename}")
        print(f"{'='*80}")

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
        shuffle=False,  # CRITICAL: Don't shuffle to maintain reproducibility
        collate_fn=dataset.collate_fn,
    )
    total_processed = len(all_generations) * n_samples if all_generations else 0
    wall_times = []
    skipped_count = 0
    questions_remaining = len(dataset) - len(all_generations)

    # Create progress bar that shows actual questions processed, not batch iteration
    pbar = tqdm(total=questions_remaining, 
                desc=f"Processing remaining questions",
                initial=0)

    for batch in dataloader:
        # Check if this batch should be skipped (already processed)
        questions = batch["questions"]
        batch_questions_to_process = []
        batch_indices_to_process = []
        
        for j, question in enumerate(questions):
            if question not in processed_questions:
                batch_questions_to_process.append(question)
                batch_indices_to_process.append(j)
        
        if not batch_questions_to_process:
            skipped_count += len(questions)
            continue
        
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        prompts = batch["prompts"]

        batch_size_actual = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size_actual)]
        raw_generations = [[] for _ in range(batch_size_actual)]
        all_extracted_answers = [[] for _ in range(batch_size_actual)]
        
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
                raw_generations[j].append(text)
                eos_markers = ['<|endoftext|>', '<|end|>', '<|im_end|>', '</s>', '<|eot_id|>']
                for marker in eos_markers:
                    if marker in text:
                        text = text.split(marker)[0]
                        break
                all_cleaned_texts[j].append(text)
                
                try:
                    extracted_answer = Parser.extract_answer_boxed(text)
                    try:
                        extracted_answer = float(extracted_answer)
                    except (ValueError, TypeError):
                        pass
                except Exception as e:
                    extracted_answer = None
                all_extracted_answers[j].append(extracted_answer)
        
        # Create results
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": all_cleaned_texts[j] if n_samples > 1 else all_cleaned_texts[j][0],
                "extracted_answer": all_extracted_answers[j] if n_samples > 1 else all_extracted_answers[j][0],
                "ground_truth": gt_answers[j],
            }
            for j in range(batch_size_actual)
        ]
        
        # Only add results for questions we actually processed
        for j, result in enumerate(example_result):
            if j in batch_indices_to_process:
                all_generations.append(result)
                processed_questions.add(result["question"])
        
        total_processed += len(batch_indices_to_process) * n_samples
        wall_times.append(time.time() - start_time)
        
        # Update progress bar for each question processed
        pbar.update(len(batch_indices_to_process))

        # CRITICAL: Save after EVERY batch
        avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
        with open(filename, "w") as f:
            json.dump(
                {
                    "metrics": {
                        "wall_time": avg_wall_time,
                        "total_processed": total_processed,
                        "num_completed": len(all_generations),
                        "num_remaining": len(dataset) - len(all_generations),
                    },
                    "generations": all_generations,
                },
                f,
                indent=2,
            )

        # Clean up GPU memory
        del out, input_ids, attention_mask
        torch.cuda.empty_cache()

    pbar.close()
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} already-processed questions")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"Total generations: {len(all_generations)}")
    print(f"Saved to: {filename}")
    print(f"{'='*80}")
    
    del diffusion_model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return filename

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

    # Determine output filename BEFORE loading model
    model_name_clean = diffusion_model_name.replace("/", "_")
    filename = f"{output_dir}/{model_name_clean}_{gen_length}_{diffusion_steps}_{few_shot}_{n_samples}_{num_evals_to_use}_{temperature}_generations_testing_fast_dllm.json"
    
    # Check for existing results and resume capability
    all_generations = []
    processed_questions = set()
    
    if os.path.exists(filename):
        print(f"\n{'='*80}")
        print(f"FOUND EXISTING RESULTS: {filename}")
        print(f"{'='*80}")
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
                all_generations = existing_data.get('generations', [])
                # Track which questions have been processed
                for gen in all_generations:
                    processed_questions.add(gen.get('question', ''))
                print(f"Loaded {len(all_generations)} existing generations")
                print(f"Will resume from question {len(all_generations) + 1}")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting fresh...")
            all_generations = []
            processed_questions = set()
    else:
        print(f"\n{'='*80}")
        print(f"STARTING NEW EVALUATION")
        print(f"Output will be saved to: {filename}")
        print(f"{'='*80}")

    diffusion_model, tokenizer = load_fast_diffusion_model_and_tokenizer(diffusion_model_name, device)
    diffusion_model.eval()
    device = diffusion_model.device

    if "base" not in diffusion_model_name.lower() and "instruct" not in diffusion_model_name.lower():
        diffusion_model_name = diffusion_model_name + "-base"
    is_base_model = 'base' in diffusion_model_name.lower()

    # Create dataset - seed ensures same questions every time
    dataset = DATASET_MAP[data](
            tokenizer,
            subsample=num_evals_to_use,
            num_examples=few_shot,
            is_base_model=is_base_model,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: Don't shuffle to maintain reproducibility
        collate_fn=dataset.collate_fn,
    )

    total_processed = len(all_generations) * n_samples if all_generations else 0
    wall_times = []
    skipped_count = 0
    questions_remaining = len(dataset) - len(all_generations)

    # Create progress bar that shows actual questions processed, not batch iteration
    pbar = tqdm(total=questions_remaining, 
                desc=f"Processing remaining questions",
                initial=0)

    for batch_idx, batch in enumerate(dataloader):
        # Check if this batch should be skipped (already processed)
        questions = batch["questions"]
        batch_questions_to_process = []
        batch_indices_to_process = []
        
        for j, question in enumerate(questions):
            if question not in processed_questions:
                batch_questions_to_process.append(question)
                batch_indices_to_process.append(j)
        
        # Skip if all questions in this batch are already processed
        if not batch_questions_to_process:
            skipped_count += len(questions)
            continue
        
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gt_answers = batch["answers"]
        prompts = batch["prompts"]

        batch_size_actual = len(questions)
        all_cleaned_texts = [[] for _ in range(batch_size_actual)]
        raw_generations = [[] for _ in range(batch_size_actual)]
        all_extracted_answers = [[] for _ in range(batch_size_actual)]
        
        # Generate n_samples for each question in the batch
        for sample_idx in range(n_samples):
            # Note: We DON'T reset seeds here for reproducibility
            # With temperature > 0: Sampling naturally creates diverse outputs
            # With temperature = 0: All samples will be identical (expected for greedy)
            # This ensures Pass@k results are reproducible across runs
            
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
            for j in range(batch_size_actual)
        ]
        
        # Only add results for questions we actually processed
        for j, result in enumerate(example_result):
            if j in batch_indices_to_process:
                all_generations.append(result)
                processed_questions.add(result["question"])
        
        total_processed += len(batch_indices_to_process) * n_samples
        wall_times.append(time.time() - start_time)
        
        # Update progress bar for each question processed
        pbar.update(len(batch_indices_to_process))

        # CRITICAL: Save after EVERY batch to prevent data loss
        avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
        with open(filename, "w") as f:
            json.dump(
                {
                    "metrics": {
                        "wall_time": avg_wall_time,
                        "total_processed": total_processed,
                        "num_completed": len(all_generations),
                        "num_remaining": len(dataset) - len(all_generations),
                    },
                    "generations": all_generations,
                },
                f,
                indent=2,
            )
        
        # CRITICAL: Clean up GPU memory after each batch to prevent 20GB->30GB+ leak
        # Without this, KV cache and intermediate tensors accumulate
        del out, input_ids, attention_mask
        torch.cuda.empty_cache()

    pbar.close()
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} already-processed questions")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"Total generations: {len(all_generations)}")
    print(f"Saved to: {filename}")
    print(f"{'='*80}")

    del diffusion_model
    del tokenizer
    del dataset
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return filename

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_model_name", default="GSAI-ML/LLaDA-8B-Base", type=str)
    args = parser.parse_args()

    diffusion_model_name = args.diffusion_model_name
    data = "countdown"
    num_evals_to_use = 20
    few_shot = 4
    batch_size = 1
    gen_length = 256
    diffusion_steps = 256
    temperature = 0.7
    cfg_scale = 0.0
    steps = 256
    block_length = 32
    remasking = "low_confidence"
    alg = "entropy"
    alg_temp = 0.0
    top_p = 0.95
    top_k = None
    n_samples = 128
    use_cache=True
    dual_cache=True
    threshold = 0.9
    factor = 1.0

    filename = evaluate_fast_dllm(
        diffusion_model_name, data, num_evals_to_use, few_shot, batch_size, gen_length, diffusion_steps, temperature, cfg_scale, steps, block_length, remasking, alg, alg_temp, top_p, top_k, n_samples, use_cache, dual_cache, threshold, factor
        )

    # filename = evaluate_auto_regressive_model(
    #     model_name=diffusion_model_name,
    #     data=data,
    #     num_evals_to_use=num_evals_to_use,
    #     few_shot=few_shot,
    #     batch_size=batch_size,
    #     gen_length=gen_length,
    #     temperature=temperature,
    #     top_p=top_p,
    #     n_samples=n_samples,
    # )
    
    compute_metrics(
        results_file=filename,
        samples_per_problem=32,
        k_values=[1, 2, 4, 8, 16, 32]
    )

if __name__ == "__main__":
    main()
