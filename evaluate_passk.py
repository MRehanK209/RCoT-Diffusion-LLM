import argparse
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
from dataset_loader import load_gsm8k, load_gsm8k_fewshot, load_aime25, load_humaneval, load_humaneval_fewshot
from dataset_loader import load_sudoku_4x4, load_countdown_cd3, load_trip_planning
from utils import (
    extract_answer,
    check_answer,
    check_code_correctness,
    extract_code_from_response,
    extract_sudoku_answer,
    check_sudoku_solution,
    sudoku_empty_cell_accuracy,
    extract_countdown_equation,
    check_countdown_solution,
    extract_trip_plan,
    check_trip_plan,
)
from typing import List, Dict, Tuple, Optional
import os
from metrics import compute_pass_at_k

# Supported diffusion models mapping
DIFFUSION_MODELS = {
    # LLaDA family
    'llada-8b': 'GSAI-ML/LLaDA-8B-Instruct',
    'llada-8b-base': 'GSAI-ML/LLaDA-8B-Base',
    'llada-1.5': 'GSAI-ML/LLaDA-1.5-8B-Instruct',
    'llada-1.5-base': 'GSAI-ML/LLaDA-1.5-8B-Base',
    'llada-moe': 'GSAI-ML/LLaDA-MoE-Instruct',  # 7B params, 1.4B active
    'llada-moe-base': 'GSAI-ML/LLaDA-MoE-Base',
    # Dream family
    'dream-7b': 'Dream-org/Dream-v0-Instruct-7B',
    'dream-7b-base': 'Dream-org/Dream-v0-Base-7B',
    # D1-LLaDA
    'd1-llada': 'multimodalart/d1-llada-8b',
}

# Models that use Dream's diffusion_generate API
DREAM_MODELS = ['dream-org', 'dream-v0']

# Models that use LLaDA's generate API (includes MoE and 1.5 variants)
LLADA_MODELS = ['llada', 'gsai-ml']


def is_diffusion_model(model_name: str) -> bool:
    """Check if the model is a diffusion-based LLM."""
    model_name_lower = model_name.lower()
    
    # Check if it's in our known diffusion models
    if any(key in model_name_lower for key in DIFFUSION_MODELS.keys()):
        return True
    
    # Check if it explicitly contains diffusion-related keywords
    if any(keyword in model_name_lower for keyword in ['llada', 'd1-', 'dream', 'diffusion', 'moe']):
        return True
    
    return False


def is_dream_model(model_name: str) -> bool:
    """Check if the model uses Dream's diffusion_generate API."""
    model_name_lower = model_name.lower()
    return any(keyword in model_name_lower for keyword in DREAM_MODELS)


def is_llada_model(model_name: str) -> bool:
    """Check if the model uses LLaDA's generate API (standard LLaDA, MoE, 1.5, D1)."""
    model_name_lower = model_name.lower()
    # LLaDA models but NOT Dream models
    if is_dream_model(model_name):
        return False
    return any(keyword in model_name_lower for keyword in LLADA_MODELS) or 'llada' in model_name_lower


def load_model_and_tokenizer(
    model_name: str, 
    model_type: Optional[str] = None,
    use_flash_attn: bool = True,
    use_compile: bool = False
):
    """
    Load model and tokenizer, supporting both autoregressive and diffusion models.
    
    Args:
        model_name: HuggingFace model name or path
        model_type: Optional explicit model type ('autoregressive' or 'diffusion')
        use_flash_attn: Whether to use Flash Attention 2 (default: True)
        use_compile: Whether to use torch.compile for optimization (default: False)
    
    Returns:
        Tuple of (model, tokenizer, is_diffusion, is_dream)
    """
    print(f"Loading model: {model_name}")
    
    # Determine if this is a diffusion model
    if model_type == 'diffusion':
        is_diff = True
    elif model_type == 'autoregressive':
        is_diff = False
    else:
        is_diff = is_diffusion_model(model_name)
    
    # Check if it's a Dream model (uses special API)
    is_dream = is_dream_model(model_name)
    
    print(f"Model type: {'Diffusion' if is_diff else 'Autoregressive'}")
    if is_dream:
        print(f"Using Dream's diffusion_generate API")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Common model loading kwargs
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        'device_map': "auto",
    }
    
    # Try to enable Flash Attention 2 for faster inference
    flash_attn_enabled = False
    if use_flash_attn:
        try:
            import flash_attn
            # Test that flash_attn actually works (not just imports)
            _ = flash_attn.__version__
            model_kwargs['attn_implementation'] = "flash_attention_2"
            flash_attn_enabled = True
            print(f"Using Flash Attention 2 (v{flash_attn.__version__}) for faster inference")
        except (ImportError, OSError, Exception) as e:
            # Catch ImportError, OSError (symbol errors), and other loading issues
            error_msg = str(e)
            if 'undefined symbol' in error_msg or 'cannot open' in error_msg:
                print(f"Flash Attention 2 binary incompatible with current PyTorch, using default attention")
                print(f"  (Rebuild with: pip uninstall flash-attn -y && pip install flash-attn --no-build-isolation)")
            else:
                print("Flash Attention 2 not installed, using default attention (install with: pip install flash-attn)")
    
    # Dream models use AutoModel instead of AutoModelForCausalLM
    try:
        if is_dream:
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (ImportError, ValueError) as e:
        # Fallback if flash attention fails during model load
        # ValueError: when model doesn't support flash attention
        # ImportError: when flash_attn package has issues
        if ('flash_attn' in str(e) or 'Flash Attention' in str(e)) and flash_attn_enabled:
            print(f"Flash Attention 2 not supported by this model, falling back to default attention...")
            model_kwargs.pop('attn_implementation', None)
            if is_dream:
                model = AutoModel.from_pretrained(model_name, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise
    
    model.eval()
    
    # Optional: compile model for faster inference (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, is_diff, is_dream


SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
""".strip()


COUNTDOWN_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. "
    "You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. "
    "After reasoning, provide only your final expression inside \\\\boxed{} tags without including an equals sign or the target number. "
    "For example: \\\\boxed{a + b * c}\n"
    "Respond in the following format:\n"
    "<reasoning>\nYour reasoning here\n</reasoning>\n"
    "<answer>\n\\\\boxed{...}\n</answer>"
)


TRIP_PLANNING_SYSTEM_PROMPT = """
You are a travel planning assistant. Given a user request and constraints, produce a feasible itinerary.

You MUST respond in the following format:
<reasoning>
Briefly explain how you satisfy the constraints.
</reasoning>
<answer>
Return ONLY valid JSON (no markdown) with this shape:
{
  "days": [
    {"day": 1, "city": "...", "activities": ["..."], "estimated_cost_usd": 0},
    ...
  ],
  "total_cost_usd": 0
}
</answer>
""".strip()


def create_prompt(
    question: str,
    n_fewshots: int = 0,
    is_base_model: bool = False,
    dataset: str = "gsm8k",
) -> str:
    """
    Create a prompt for the question with optional few-shot examples.
    
    Args:
        question: The question to answer
        n_fewshots: Number of few-shot examples
        is_base_model: If True, use continuation-style format for base models
        dataset: Dataset name for task-specific formatting
    """
    dataset = (dataset or "gsm8k").lower()

    if dataset in {"sudoku", "countdown", "trip_planning"}:
        system_prompt = {
            "sudoku": SUDOKU_SYSTEM_PROMPT,
            "countdown": COUNTDOWN_SYSTEM_PROMPT,
            "trip_planning": TRIP_PLANNING_SYSTEM_PROMPT,
        }[dataset]

        if n_fewshots > 0 and dataset != "trip_planning":
            # Few-shot for these structured tasks is intentionally disabled by default.
            # (You can add task-specific few-shot loaders later.)
            pass

        if is_base_model:
            prompt = f"{system_prompt}\n\n{question}\n\n<reasoning>\n"
            return prompt
        else:
            # Instruct-style wrapper; keep all task constraints inside the user turn.
            prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n\n"
            prompt += f"User: {system_prompt}\n\n{question}\n\nAssistant:<reasoning>\n"
            return prompt

    if is_base_model:
        # Format for base models (direct continuation, no instruction)
        prompt = ""
        
        if n_fewshots > 0:
            examples = load_gsm8k_fewshot(n_fewshots)
            for example in examples:
                prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
        
        prompt += f"Q: {question}\nA: Let's solve this step by step.\n"
        return prompt
    else:
        # Format for instruct models
        prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n\n"
        
        if n_fewshots > 0:
            examples = load_gsm8k_fewshot(n_fewshots)
            for example in examples:
                prompt += f"User: {example['question']}\n"
                prompt += f"Assistant: {example['answer']}\n\n"
        
        prompt += f"User: {question}\n"
        prompt += "Please reason step by step, and put your final answer within \\boxed{{}}. \n\n"
        prompt += "Assistant:"
        return prompt
    
    if n_fewshots > 0:
        examples = load_gsm8k_fewshot(n_fewshots)
        for example in examples:
            prompt += f"User: {example['question']}\n"
            prompt += f"Assistant: {example['answer']}\n\n"
    
    prompt += f"User: {question}\n"
    prompt += "Please reason step by step, and put your final answer within \\boxed{{}}. \n\n"
    prompt += "Assistant:"
    return prompt


def create_dream_chat_prompt(
    question: str,
    tokenizer,
    n_fewshots: int = 0,
    dataset: str = "gsm8k",
    is_base_model: bool = False,
) -> Dict:
    """Create a prompt for Dream models.

    - Instruct models: uses the tokenizer chat template
    - Base models: uses a plain continuation-style prompt (no chat template)
    """
    dataset = (dataset or "gsm8k").lower()

    if is_base_model:
        prompt = create_prompt(question, n_fewshots=n_fewshots, is_base_model=True, dataset=dataset)
        return tokenizer(prompt, return_tensors="pt", return_dict=True)

    messages = []
    
    # Add few-shot examples if requested
    if n_fewshots > 0:
        if dataset == 'humaneval':
            examples = load_humaneval_fewshot(n_fewshots)
            for example in examples:
                messages.append({"role": "user", "content": f"Complete the following Python function:\n\n{example['prompt']}"})
                messages.append({"role": "assistant", "content": example['solution']})
        else:
            examples = load_gsm8k_fewshot(n_fewshots)
            for example in examples:
                messages.append({"role": "user", "content": example['question']})
                messages.append({"role": "assistant", "content": example['answer']})
    
    # Add the current question (task-specific formatting)
    if dataset == "humaneval":
        question_with_instruction = (
            f"Complete the following Python function. Only output the code, no explanations.\n\n{question}"
        )
    elif dataset in {"sudoku", "countdown", "trip_planning"}:
        system_prompt = {
            "sudoku": SUDOKU_SYSTEM_PROMPT,
            "countdown": COUNTDOWN_SYSTEM_PROMPT,
            "trip_planning": TRIP_PLANNING_SYSTEM_PROMPT,
        }[dataset]
        question_with_instruction = f"{system_prompt}\n\n{question}"
    else:
        question_with_instruction = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    
    messages.append({"role": "user", "content": question_with_instruction})
    
    # Use the tokenizer's chat template to produce a single string prompt,
    # then optionally prefill <reasoning> like d1-style evaluation.
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if dataset in {"sudoku", "countdown", "trip_planning"}:
        prompt_str = prompt_str + "<reasoning>\n"

    return tokenizer(prompt_str, return_tensors="pt", return_dict=True)


def create_code_prompt(code_prompt: str, n_fewshots: int = 0) -> str:
    """Create a prompt for code completion (HumanEval)."""
    prompt = ""
    
    if n_fewshots > 0:
        examples = load_humaneval_fewshot(n_fewshots)
        for example in examples:
            prompt += f"# Complete the following function:\n{example['prompt']}\n"
            prompt += f"{example['solution']}\n\n"
    
    # Add the current code prompt
    prompt += f"# Complete the following function:\n{code_prompt}"
    return prompt

def generate_dream_samples(
    model,
    tokenizer,
    question: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_fewshots: int,
    diffusion_steps: int,
    alg: str,
    verbose: bool,
    batch_size: int = 8,
    dataset: str = 'gsm8k',
    is_base_model: bool = False
) -> List[str]:
    """
    Generate samples using Dream's diffusion_generate API with batch parallelism.
    
    Based on: https://github.com/DreamLM/Dream
    
    Args:
        batch_size: Number of samples to generate in parallel (default: 8)
        dataset: Dataset type for prompt formatting ('gsm8k', 'aime25', 'humaneval')
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Generating with Dream diffusion model (BATCHED)...")
        print(f"Diffusion steps: {diffusion_steps}, Algorithm: {alg}, Batch size: {batch_size}")
        print(f"{'='*80}\n")
    
    # Create chat-formatted inputs
    inputs = create_dream_chat_prompt(question, tokenizer, n_fewshots, dataset, is_base_model=is_base_model)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    input_length = input_ids.shape[1]
    
    outputs_list = []
    
    # Generate samples in batches for better throughput
    for batch_start in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - batch_start)
        
        # Replicate inputs for batch processing
        batched_input_ids = input_ids.repeat(current_batch_size, 1)
        batched_attention_mask = attention_mask.repeat(current_batch_size, 1)
        
        with torch.no_grad():
            output = model.diffusion_generate(
                batched_input_ids,
                attention_mask=batched_attention_mask,
                max_new_tokens=max_tokens,
                steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=0.0,
                output_history=False,
                return_dict_in_generate=True,
            )
        
        # Decode all sequences in the batch
        for seq_idx in range(current_batch_size):
            generated_ids = output.sequences[seq_idx][input_length:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Split by EOS token if present
            if tokenizer.eos_token:
                text = text.split(tokenizer.eos_token)[0]
            
            outputs_list.append(text)
            
            sample_num = batch_start + seq_idx + 1
            if verbose and sample_num <= 3:
                print(f"\nSample {sample_num}/{n_samples}:")
                print(f"{'-'*80}")
                print(text)
                print(f"{'-'*80}")
        
        # Explicitly delete tensors to free memory
        del output
        del batched_input_ids
        del batched_attention_mask
        
        # Aggressive GPU cache clearing between batches to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return outputs_list


def generate_diffusion_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    diffusion_steps: int,
    verbose: bool,
    batch_size: int = 8
) -> List[str]:
    """
    Generate samples using generic diffusion-based LLMs (e.g., LLaDA, D1-LLaDA, LLaDA-MoE).
    
    For models that use standard HuggingFace generate API with diffusion internally.
    Supports batch parallel generation for improved throughput.
    
    Args:
        batch_size: Number of samples to generate in parallel (default: 8)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    outputs_list = []
    
    if verbose:
        print(f"Generating {n_samples} samples with batch_size={batch_size}")
    
    # Generate samples in batches for better throughput
    for batch_start in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - batch_start)
        
        # Replicate inputs for batch processing
        batched_input_ids = inputs['input_ids'].repeat(current_batch_size, 1)
        batched_attention_mask = inputs['attention_mask'].repeat(current_batch_size, 1)
        
        with torch.no_grad():
            # Check if the model has a diffusion-specific generate method
            if hasattr(model, 'generate_diffusion'):
                outputs = model.generate_diffusion(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    max_new_tokens=max_tokens,
                    num_diffusion_steps=diffusion_steps,
                    temperature=temperature,
                )
            else:
                # Use standard generation with batched inputs
                # IMPORTANT: Diffusion models like LLaDA don't support KV cache
                # Handle temperature=0 by using greedy decoding
                gen_kwargs = {
                    'input_ids': batched_input_ids,
                    'attention_mask': batched_attention_mask,
                    'max_new_tokens': max_tokens,
                    'num_return_sequences': 1,  # Already batched via input replication
                    'use_cache': False,  # Disable KV cache for diffusion models
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
                
                if temperature == 0 or temperature < 0.01:
                    # Use greedy decoding for temperature=0
                    gen_kwargs['do_sample'] = False
                else:
                    # Use sampling for temperature > 0
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = temperature
                
                outputs = model.generate(**gen_kwargs)
            
            # Decode all sequences in the batch
            for seq_idx in range(current_batch_size):
                text = tokenizer.decode(outputs[seq_idx][input_length:], skip_special_tokens=True)
                outputs_list.append(text)
                
                sample_num = batch_start + seq_idx + 1
                if verbose and sample_num <= 3:
                    print(f"Generated diffusion sample {sample_num}/{n_samples}")
            
            # Explicitly delete outputs to free memory
            del outputs
            del batched_input_ids
            del batched_attention_mask
        
        # Aggressive GPU cache clearing between batches to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all GPU operations are complete
    
    return outputs_list


def generate_autoregressive_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    verbose: bool,
    batch_size: int = 8
) -> List[str]:
    """Generate samples using standard autoregressive models with batch parallelism."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    outputs_list = []
    effective_batch_size = min(n_samples, batch_size)
    
    if verbose:
        print(f"Generating {n_samples} samples with batch_size={effective_batch_size}")
    
    for batch_start in range(0, n_samples, effective_batch_size):
        current_batch = min(effective_batch_size, n_samples - batch_start)
        
        with torch.no_grad():
            gen_kwargs = {
                **inputs,
                'max_new_tokens': max_tokens,
                'num_return_sequences': current_batch,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id
            }
            
            if temperature == 0 or temperature < 0.01:
                # Use greedy decoding for temperature=0
                gen_kwargs['do_sample'] = False
            else:
                # Use sampling for temperature > 0
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = top_p
            
            outputs = model.generate(**gen_kwargs)
        
        for seq_idx, output in enumerate(outputs):
            text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            outputs_list.append(text)
            
            sample_num = batch_start + seq_idx + 1
            if verbose and sample_num <= 3:
                print(f"Generated AR sample {sample_num}/{n_samples}")
        
        # Explicitly delete outputs to free memory
        del outputs
        
        # Aggressive GPU cache clearing between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return outputs_list


def generate_samples(
    model,
    tokenizer,
    question: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_fewshots: int,
    is_diffusion: bool,
    is_dream: bool,
    diffusion_steps: int,
    diffusion_alg: str,
    verbose: bool,
    batch_size: int = 8,
    dataset: str = 'gsm8k',
    is_base_model: bool = False
) -> List[str]:
    """
    Generate samples using either diffusion or autoregressive models.
    
    Args:
        batch_size: Number of samples to generate in parallel (default: 8)
        dataset: Dataset type for prompt formatting ('gsm8k', 'aime25', 'humaneval')
        is_base_model: Whether to use base model prompting (continuation style)
    """
    if is_dream:
        # Dream models use their own API with batched generation
        return generate_dream_samples(
            model, tokenizer, question, n_samples,
            temperature, top_p, max_tokens, n_fewshots,
            diffusion_steps, diffusion_alg, verbose, batch_size, dataset, is_base_model
        )
    elif is_diffusion:
        # Other diffusion models (LLaDA, D1-LLaDA, LLaDA-MoE, etc.)
        if dataset == 'humaneval':
            prompt = create_code_prompt(question, n_fewshots)
        else:
            prompt = create_prompt(question, n_fewshots, is_base_model, dataset=dataset)
        
        if verbose:
            print(f"\n{'='*80}")
            print("PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        return generate_diffusion_samples(
            model, tokenizer, prompt, n_samples,
            temperature, max_tokens, diffusion_steps, verbose, batch_size
        )
    else:
        # Standard autoregressive models
        if dataset == 'humaneval':
            prompt = create_code_prompt(question, n_fewshots)
        else:
            prompt = create_prompt(question, n_fewshots, is_base_model, dataset=dataset)
        
        if verbose:
            print(f"\n{'='*80}")
            print("PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        return generate_autoregressive_samples(
            model, tokenizer, prompt, n_samples,
            temperature, top_p, max_tokens, verbose, batch_size
        )


def evaluate_problem(
    model,
    tokenizer,
    problem: Dict,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_fewshots: int,
    is_diffusion: bool,
    is_dream: bool,
    diffusion_steps: int,
    diffusion_alg: str,
    verbose: bool,
    batch_size: int = 8,
    dataset: str = 'gsm8k',
    is_base_model: bool = False
) -> Dict:
    question = problem['question']
    ground_truth = problem['answer']
    dataset = (dataset or "gsm8k").lower()
    
    # For HumanEval, we also need test cases and entry point
    is_code_eval = dataset == 'humaneval'
    test_code = problem.get('test', '') if is_code_eval else ''
    entry_point = problem.get('entry_point', '') if is_code_eval else ''
    prompt = problem.get('prompt', question) if is_code_eval else question
    
    if verbose:
        print(f"\n{'#'*80}")
        if is_code_eval:
            print(f"Problem {problem['idx']} ({problem.get('task_id', 'unknown')})")
            print(f"Entry point: {entry_point}")
        else:
            print(f"Problem {problem['idx']}: {question}")
            print(f"Ground Truth: {ground_truth}")
        print(f"{'#'*80}")
    
    responses = generate_samples(
        model, tokenizer, question, n_samples, 
        temperature, top_p, max_tokens, n_fewshots,
        is_diffusion, is_dream, diffusion_steps, diffusion_alg, verbose, batch_size, dataset, is_base_model
    )
    
    correct_count = 0
    correct_examples = []
    incorrect_examples = []
    
    # Task-specific accumulators (optional diagnostics)
    sudoku_accs: List[float] = []
    trip_plan_details_samples: List[Dict] = []

    for idx, response in enumerate(responses):
        if is_code_eval:
            is_correct = check_code_correctness(response, prompt, test_code, entry_point, timeout=5)
            pred_answer = extract_code_from_response(response, entry_point)[:200]  # Truncated for storage
        elif dataset == "sudoku":
            puzzle = problem.get("puzzle", "")
            pred_answer = extract_sudoku_answer(response)
            is_correct = check_sudoku_solution(pred_answer, puzzle, ground_truth)
            _, _, acc = sudoku_empty_cell_accuracy(pred_answer, puzzle, ground_truth)
            sudoku_accs.append(acc)
        elif dataset == "countdown":
            gt = ground_truth if isinstance(ground_truth, dict) else {}
            numbers = problem.get("numbers") or gt.get("numbers") or []
            target = problem.get("target") or gt.get("target")
            pred_answer = extract_countdown_equation(response)
            is_correct = False
            if target is not None:
                is_correct = check_countdown_solution(pred_answer, numbers, int(target))
        elif dataset == "trip_planning":
            constraints = problem.get("constraints") or (ground_truth if isinstance(ground_truth, dict) else {})
            plan = extract_trip_plan(response)
            is_correct, details = check_trip_plan(plan, constraints)
            trip_plan_details_samples.append(details)
            # Store a compact view for JSON outputs
            pred_answer = plan if plan is not None else None
        else:
            # Default: numeric math-style evaluation (GSM8K/AIME25)
            pred_answer = extract_answer(response)
            is_correct = bool(pred_answer) and check_answer(pred_answer, ground_truth)
        
        if is_correct:
            correct_count += 1
            if len(correct_examples) < 5:
                correct_examples.append({
                    'response': response,
                    'extracted_answer': pred_answer
                })
        else:
            if len(incorrect_examples) < 5:
                incorrect_examples.append({
                    'response': response,
                    'extracted_answer': pred_answer
                })
        
        if verbose and idx < 3:
            print(f"\nSample {idx + 1}:")
            print(f"{'-'*80}")
            print(response[:500] if len(response) > 500 else response)
            print(f"{'-'*80}")
            if not is_code_eval:
                print(f"Extracted Answer: {pred_answer}")
            print(f"Correct: {is_correct}")
    
    if verbose:
        print(f"\nCorrect: {correct_count}/{n_samples}")
    
    result = {
        'idx': problem['idx'],
        'question': question,
        'answer': ground_truth,
        'correct_count': correct_count,
        'total_samples': n_samples,
        'temperature': temperature,
        'n_fewshots': n_fewshots,
        'correct_examples': correct_examples,
        'incorrect_examples': incorrect_examples
    }

    # Task-specific metadata / diagnostics
    if dataset == "sudoku":
        result["puzzle"] = problem.get("puzzle", "")
        if sudoku_accs:
            result["sudoku_empty_cell_accuracy_max"] = max(sudoku_accs)
            result["sudoku_empty_cell_accuracy_mean"] = sum(sudoku_accs) / len(sudoku_accs)
    elif dataset == "countdown":
        result["numbers"] = problem.get("numbers", [])
        result["target"] = problem.get("target", None)
    elif dataset == "trip_planning":
        result["constraints"] = problem.get("constraints", {})
        if trip_plan_details_samples:
            # Save only a small slice to keep result files compact
            result["trip_plan_details_examples"] = trip_plan_details_samples[:3]
    
    # Add HumanEval-specific fields
    if is_code_eval:
        result['task_id'] = problem.get('task_id', '')
        result['entry_point'] = entry_point
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pass@k on GSM8K, AIME25, or HumanEval with support for diffusion LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Dream 7B with batch parallelism
  python evaluate_passk.py --model_name Dream-org/Dream-v0-Instruct-7B --n_samples 100 --batch_size 8
  
  # Evaluate LLaDA 8B on HumanEval
  python evaluate_passk.py --model_name GSAI-ML/LLaDA-8B-Instruct --dataset humaneval --n_samples 50
  
  # Evaluate LLaDA-MoE with custom k values
  python evaluate_passk.py --model_name GSAI-ML/LLaDA-MoE-Instruct --k_values 1 10 100 --n_samples 100
  
  # Evaluate LLaDA 1.5
  python evaluate_passk.py --model_name GSAI-ML/LLaDA-1.5-8B-Instruct --n_samples 256 --batch_size 4
        """
    )
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--model_type', type=str, default=None, choices=['autoregressive', 'diffusion', None],
                        help='Explicitly specify model type (auto-detected if not provided)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='gsm8k',
        choices=['gsm8k', 'aime25', 'humaneval', 'sudoku', 'countdown', 'trip_planning'],
        help='Dataset to evaluate on (default: gsm8k)',
    )
    parser.add_argument('--n_samples', type=int, default=64,
                        help='Number of samples per problem (default: 128)')
    parser.add_argument('--subset_size', type=int, default=512,
                        help='Number of problems to evaluate (default: all)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0],
                        help='List of temperatures to evaluate (default: [0])')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling threshold (default: 0.95)')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum generation length (default: 256)')
    parser.add_argument('--n_fewshots', type=int, default=0,
                        help='Number of few-shot examples to include (default: 0, max: 8, use 4+ for base models)')
    parser.add_argument('--base_model_prompt', action='store_true',
                        help='Use continuation-style prompts for base models (auto-detected from model name)')
    
    # Batch and throughput parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for parallel generation (default: 8)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 10],
                        help='List of k values for pass@k computation (default: [1, 10])')
    
    # Diffusion-specific parameters
    parser.add_argument('--diffusion_steps', type=int, default=128,
                        help='Number of diffusion steps for diffusion models (default: 64, lower=faster)')
    parser.add_argument('--diffusion_alg', type=str, default='entropy', 
                        choices=['origin', 'maskgit_plus', 'topk_margin', 'entropy'],
                        help='Remasking strategy for Dream models (default: entropy)')
    
    # Performance optimization flags
    parser.add_argument('--no_flash_attn', action='store_true',
                        help='Disable Flash Attention 2 (enabled by default)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster inference (experimental)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print generated outputs and detailed evaluation')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Local dataset directory for structured tasks (sudoku/countdown/trip_planning)')
    
    args = parser.parse_args()

    # Filter k_values to those <= n_samples
    args.k_values = [k for k in args.k_values if k <= args.n_samples]
    if not args.k_values:
        args.k_values = [1]
    
    # Auto-detect base model from name
    is_base_model = args.base_model_prompt or 'base' in args.model_name.lower()
    if is_base_model and args.n_fewshots == 0:
        print("⚠️  WARNING: Base model detected but n_fewshots=0. Base models need few-shot examples!")
        print("   Recommendation: Add --n_fewshots 4 or higher")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("RCoT-Diffusion-LLM Evaluation Framework (Batch Parallel)")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Model Type: {'Base (continuation)' if is_base_model else 'Instruct'}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Samples per problem: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"K values for pass@k: {args.k_values}")
    if args.subset_size:
        print(f"Evaluating on subset: {args.subset_size} problems")
    print(f"{'='*80}\n")
    
    model, tokenizer, is_diffusion, is_dream = load_model_and_tokenizer(
        args.model_name, 
        args.model_type,
        use_flash_attn=not args.no_flash_attn,
        use_compile=args.compile
    )
    
    # Load dataset
    if args.dataset == 'gsm8k':
        problems = load_gsm8k(args.subset_size)
    elif args.dataset == 'aime25':
        problems = load_aime25(args.subset_size)
    elif args.dataset == 'humaneval':
        problems = load_humaneval(args.subset_size)
        # For code generation, use longer max_tokens by default
        if args.max_tokens == 256:
            args.max_tokens = 512
            print(f"Auto-adjusted max_tokens to {args.max_tokens} for code generation")
    elif args.dataset == 'sudoku':
        problems = load_sudoku_4x4(args.subset_size, data_dir=args.data_dir)
    elif args.dataset == 'countdown':
        problems = load_countdown_cd3(args.subset_size, data_dir=args.data_dir)
    elif args.dataset == 'trip_planning':
        problems = load_trip_planning(args.subset_size, data_dir=args.data_dir)
    
    for temperature in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Evaluating on {args.dataset.upper()} with temperature={temperature}")
        print(f"Batch size: {args.batch_size}, Few-shot: {args.n_fewshots}")
        if is_diffusion:
            print(f"Diffusion steps: {args.diffusion_steps}")
            if is_dream:
                print(f"Diffusion algorithm: {args.diffusion_alg}")
        print(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        total_samples_generated = 0
        
        for problem in tqdm(problems, desc=f"T={temperature}", disable=args.verbose):
            problem_start_time = time.time()
            
            result = evaluate_problem(
                model, tokenizer, problem,
                args.n_samples, temperature,
                args.top_p, args.max_tokens,
                args.n_fewshots, is_diffusion, is_dream,
                args.diffusion_steps, args.diffusion_alg, args.verbose,
                args.batch_size, args.dataset, is_base_model
            )
            
            problem_time = time.time() - problem_start_time
            result['generation_time_seconds'] = problem_time
            result['samples_per_second'] = args.n_samples / problem_time if problem_time > 0 else 0
            
            results.append(result)
            total_samples_generated += args.n_samples
        
        total_time = time.time() - start_time
        
        # Compute pass@k metrics
        pass_k_results = compute_pass_at_k(results, args.k_values)
        
        # Print summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {len(problems)}")
        print(f"Samples per problem: {args.n_samples}")
        print(f"Total samples generated: {total_samples_generated}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {total_samples_generated / total_time:.2f} samples/second")
        print(f"\nPass@k Results:")
        for k, score in pass_k_results.items():
            print(f"  pass@{k}: {score:.4f} ({score*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # Create comprehensive output
        output_data = {
            'config': {
                'model_name': args.model_name,
                'dataset': args.dataset,
                'n_samples': args.n_samples,
                'batch_size': args.batch_size,
                'temperature': temperature,
                'top_p': args.top_p,
                'max_tokens': args.max_tokens,
                'n_fewshots': args.n_fewshots,
                'diffusion_steps': args.diffusion_steps if is_diffusion else None,
                'diffusion_alg': args.diffusion_alg if is_dream else None,
                'is_diffusion': is_diffusion,
                'is_dream': is_dream
            },
            'timing': {
                'total_time_seconds': total_time,
                'total_samples': total_samples_generated,
                'samples_per_second': total_samples_generated / total_time if total_time > 0 else 0,
                'problems_evaluated': len(problems)
            },
            'pass_at_k': pass_k_results,
            'results': results
        }
        
        # Create descriptive output filename
        model_short = args.model_name.split('/')[-1]
        output_file = os.path.join(
            args.output_dir,
            f"results_{model_short}_{args.dataset}_temp{temperature}_n{args.n_samples}_batch{args.batch_size}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved results to: {output_file}")
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
