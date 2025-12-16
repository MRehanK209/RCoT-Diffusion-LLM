import argparse
import json
import os
import random
import time
import warnings
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import create_model
from dataset.gsm8k import GSM8KDataset
from dataset.math500 import MATH500Dataset
from dataset.countdown import CTDDataset
from dataset.sudoku import SudokuDataset
from metrics.parsers import Parser

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    """
    Set all random seeds for reproducibility.
    
    WARNING: This makes ALL generations DETERMINISTIC!
    
    ❌ DO NOT USE for pass@k evaluation (n_samples > 1)
       - All samples will be IDENTICAL
       - Defeats the purpose of generating multiple samples
    
    ✅ ONLY USE for:
       - Single-sample evaluation (n_samples = 1)
       - Reproducible experiments
       - Debugging
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These make PyTorch/cuDNN operations deterministic (no randomness)
    # For pass@k, we want these to be False to allow diversity
    torch.backends.cudnn.deterministic = True   # Enforce deterministic algorithms
    torch.backends.cudnn.benchmark = False       # Disable algorithm benchmarking


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

def evaluate(
    diffusion_model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.2,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    remasking="low_confidence",
    alg="entropy",
    alg_temp=0.0,
    top_p=0.95,
    top_k=50,
    n_samples=1,  # Number of samples to generate per question (for pass@k)
):
    """
    Evaluate the diffusion model using its generate() method.
    
    Works for both LLaDA and Dream models. Each model automatically
    filters out parameters it doesn't support.
    
    Args:
        diffusion_model: Instance of DiffusionModelBase (LLaDAModel, DreamModel, etc.)
        tokenizer: Tokenizer
        dataloader: DataLoader for evaluation data
        gen_length: Number of tokens to generate
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale (LLaDA only)
        steps: Number of diffusion steps
        block_length: Block length for generation (LLaDA only)
        remasking: Remasking strategy (LLaDA only)
        alg: Algorithm for diffusion (Dream only)
        alg_temp: Algorithm temperature (Dream only)
        top_p: Top-p sampling (Dream only)
        top_k: Top-k sampling (Dream only)
        n_samples: Number of samples to generate per question (for pass@k)
    """
    diffusion_model.model.eval()
    total_processed = 0
    wall_times = []
    all_generations = []
    device = diffusion_model.device

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
        all_extracted_answers = [[] for _ in range(batch_size)]
        
        # Generate n_samples for each question in the batch
        for sample_idx in range(n_samples):
            # Set a unique seed for each sample to ensure diversity
            # Use uuid to generate a unique seed for each iteration
            if n_samples > 1:
                unique_seed = int(uuid.uuid4().int % (2**31))  # Convert UUID to valid seed
                torch.manual_seed(unique_seed)
                torch.cuda.manual_seed_all(unique_seed)
                # Also seed numpy for Dream model's internal randomness
                np.random.seed(unique_seed % (2**32))
                # Note: We do NOT set random.seed here to keep dataset order stable
            
            # Pass all parameters - each model filters what it needs
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
    return metrics

if __name__ == "__main__":
    # Note: For pass@k evaluation with n_samples > 1, DO NOT set a fixed seed
    # or all generations will be identical. Commented out for diverse sampling.
    # init_seed(42)  # Uncomment only for reproducible single-sample evaluation

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    device = setup_device()
    
    # Explicitly enable non-deterministic mode for diverse sampling
    # This is CRITICAL for pass@k evaluation with n_samples > 1
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Suppress the harmless warning from HuggingFace about do_sample and temperature
    # Dream's diffusion_generate uses temperature directly, ignoring generation_config
    warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="gsm8k", 
                        choices=["gsm8k", "countdown", "math500", "sudoku"],
                        help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--gen_length", type=int, default=256, help="Generation length")
    parser.add_argument("--block_length", type=int, default=256, 
                        help="LLaDA: Block length for generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--diffusion_steps", type=int, default=256, help="Number of diffusion steps")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--num_evals", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--dont_save", action="store_true", help="Don't save results")
    parser.add_argument("--n_samples", type=int, default=1, 
                        help="Number of samples to generate per question (for pass@k evaluation)")
    
    # LLaDA-specific parameters
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="LLaDA: Classifier-free guidance scale")
    parser.add_argument("--remasking", type=str, default="low_confidence", 
                        choices=["low_confidence", "random"], 
                        help="LLaDA: Remasking strategy")
    
    # Dream-specific parameters
    parser.add_argument("--alg", type=str, default="entropy", 
                        choices=["entropy", "origin"],
                        help="Dream: Diffusion algorithm (entropy=stable/working, origin=random/diverse)")
    parser.add_argument("--alg_temp", type=float, default=0.0, 
                        help="Dream: Algorithm temperature (0=deterministic/stable, >0=random scheduling)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Dream: Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Dream: Top-k sampling")
    
    # Dataset seeding for reproducible question order
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Random seed for dataset shuffling (ensures same questions each run)")
    
    args = parser.parse_args()
    num_evals = {"gsm8k": 256, "math": 256, "countdown": 256, "sudoku": 256}

    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    
    # Detect if this is a Dream model or LLaDA
    is_dream = 'dream' in args.model_path.lower()
    
    # LLaDA models use AutoModel, Dream models also use AutoModel
    print("Loading with AutoModel + Flash Attention 2")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2 for speed
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # LLaDA uses left-padding
    if tokenizer.padding_side != 'left':
        print(f"Setting padding_side to 'left' (was '{tokenizer.padding_side}')")
        tokenizer.padding_side = 'left'

    # Create the appropriate diffusion model wrapper based on model_path
    print(f"\nDetecting model type from path: {args.model_path}")
    diffusion_model = create_model(model, tokenizer, args.model_path)
    print(f"Using model class: {diffusion_model.__class__.__name__}")
    
    # Detect if this is a base or instruct model
    is_base_model = 'base' in args.model_path.lower()
    print(f"Model variant: {'Base' if is_base_model else 'Instruct'}\n")

    # Seed for reproducible dataset question order
    # This ensures same questions are evaluated each run
    print(f"Setting dataset seed to {args.data_seed} for reproducible question order")
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        is_base_model=is_base_model,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    model_name = args.model_path.split("/")[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{args.few_shot}_{args.n_samples}_{args.temperature}_generations.json"
    print(f"Saving generations to {filename}")

    # Run evaluation (pass all parameters, each model filters what it needs)
    metrics = evaluate(
        diffusion_model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        alg=args.alg,
        alg_temp=args.alg_temp,
        top_p=args.top_p,
        top_k=args.top_k,
        n_samples=args.n_samples,  # Number of generations per question
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "generations": metrics["generations"],
                },
                f,
                indent=2,
            )
        print(f"Results saved to {filename}")
    else:
        print("Results not saved (--dont_save flag)")

    print(f"\nAverage wall time per batch: {metrics['wall_time']:.2f}s")
    print(f"Total samples processed: {metrics['total_processed']}")

