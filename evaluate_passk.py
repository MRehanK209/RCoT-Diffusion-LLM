import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
from dataset_loader import load_gsm8k, load_gsm8k_fewshot, load_aime25
from utils import extract_answer, check_answer
from typing import List, Dict, Tuple, Optional
import os


# Supported diffusion models mapping
DIFFUSION_MODELS = {
    'llada-8b': 'GSAI-ML/LLaDA-8B-Instruct',
    'llada-8b-base': 'GSAI-ML/LLaDA-8B-Base',
    'dream-7b': 'Dream-org/Dream-v0-Instruct-7B',
    'dream-7b-base': 'Dream-org/Dream-v0-Base-7B',
    'd1-llada': 'd1-llada',  # Will need specific path when available
}

# Models that use Dream's diffusion_generate API
DREAM_MODELS = ['dream-org', 'dream-v0']


def is_diffusion_model(model_name: str) -> bool:
    """Check if the model is a diffusion-based LLM."""
    model_name_lower = model_name.lower()
    
    # Check if it's in our known diffusion models
    if any(key in model_name_lower for key in DIFFUSION_MODELS.keys()):
        return True
    
    # Check if it explicitly contains diffusion-related keywords
    if any(keyword in model_name_lower for keyword in ['llada', 'd1-', 'dream', 'diffusion']):
        return True
    
    return False


def is_dream_model(model_name: str) -> bool:
    """Check if the model uses Dream's diffusion_generate API."""
    model_name_lower = model_name.lower()
    return any(keyword in model_name_lower for keyword in DREAM_MODELS)


def load_model_and_tokenizer(model_name: str, model_type: Optional[str] = None):
    """
    Load model and tokenizer, supporting both autoregressive and diffusion models.
    
    Args:
        model_name: HuggingFace model name or path
        model_type: Optional explicit model type ('autoregressive' or 'diffusion')
    
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
    
    # Dream models use AutoModel instead of AutoModelForCausalLM
    if is_dream:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, is_diff, is_dream


def create_prompt(question: str, n_fewshots: int = 0) -> str:
    """Create a prompt for the question with optional few-shot examples."""
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


def create_dream_chat_prompt(question: str, tokenizer, n_fewshots: int = 0) -> Dict:
    """Create a chat-formatted prompt for Dream models."""
    messages = []
    
    # Add few-shot examples if requested
    if n_fewshots > 0:
        examples = load_gsm8k_fewshot(n_fewshots)
        for example in examples:
            messages.append({"role": "user", "content": example['question']})
            messages.append({"role": "assistant", "content": example['answer']})
    
    # Add the current question
    question_with_instruction = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages.append({"role": "user", "content": question_with_instruction})
    
    # Use the tokenizer's chat template
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        return_dict=True, 
        add_generation_prompt=True
    )
    
    return inputs


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
    verbose: bool
) -> List[str]:
    """
    Generate samples using Dream's diffusion_generate API.
    
    Based on: https://github.com/DreamLM/Dream
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Generating with Dream diffusion model...")
        print(f"Diffusion steps: {diffusion_steps}, Algorithm: {alg}")
        print(f"{'='*80}\n")
    
    # Create chat-formatted inputs
    inputs = create_dream_chat_prompt(question, tokenizer, n_fewshots)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    outputs_list = []
    
    # Generate samples one at a time for Dream models
    for i in range(n_samples):
        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=0.0,
                output_history=False,
                return_dict_in_generate=True,
            )
        
        # Decode the generated text
        generated_ids = output.sequences[0][len(input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Split by EOS token if present
        if tokenizer.eos_token:
            text = text.split(tokenizer.eos_token)[0]
        
        outputs_list.append(text)
        
        if verbose and i < 3:
            print(f"\nSample {i+1}/{n_samples}:")
            print(f"{'-'*80}")
            print(text)
            print(f"{'-'*80}")
    
    return outputs_list


def generate_diffusion_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    diffusion_steps: int,
    verbose: bool
) -> List[str]:
    """
    Generate samples using generic diffusion-based LLMs (e.g., LLaDA, D1-LLaDA).
    
    For models that use standard HuggingFace generate API with diffusion internally.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs_list = []
    
    # For diffusion models, we typically generate one at a time
    for i in range(n_samples):
        with torch.no_grad():
            # Check if the model has a diffusion-specific generate method
            if hasattr(model, 'generate_diffusion'):
                output = model.generate_diffusion(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_diffusion_steps=diffusion_steps,
                    temperature=temperature,
                )
            else:
                # Fallback to standard generation
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            outputs_list.append(text)
            
            if verbose and i < 3:
                print(f"Generated diffusion sample {i+1}/{n_samples}")
    
    return outputs_list


def generate_autoregressive_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    verbose: bool
) -> List[str]:
    """Generate samples using standard autoregressive models."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs_list = []
    batch_size = min(n_samples, 8)
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=current_batch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for output in outputs:
            text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            outputs_list.append(text)
    
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
    verbose: bool
) -> List[str]:
    """
    Generate samples using either diffusion or autoregressive models.
    """
    if is_dream:
        # Dream models use their own API
        return generate_dream_samples(
            model, tokenizer, question, n_samples,
            temperature, top_p, max_tokens, n_fewshots,
            diffusion_steps, diffusion_alg, verbose
        )
    elif is_diffusion:
        # Other diffusion models (LLaDA, D1-LLaDA, etc.)
        prompt = create_prompt(question, n_fewshots)
        if verbose:
            print(f"\n{'='*80}")
            print("PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        return generate_diffusion_samples(
            model, tokenizer, prompt, n_samples,
            temperature, max_tokens, diffusion_steps, verbose
        )
    else:
        # Standard autoregressive models
        prompt = create_prompt(question, n_fewshots)
        if verbose:
            print(f"\n{'='*80}")
            print("PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        return generate_autoregressive_samples(
            model, tokenizer, prompt, n_samples,
            temperature, top_p, max_tokens, verbose
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
    verbose: bool
) -> Dict:
    question = problem['question']
    ground_truth = problem['answer']
    
    if verbose:
        print(f"\n{'#'*80}")
        print(f"Problem {problem['idx']}: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'#'*80}")
    
    responses = generate_samples(
        model, tokenizer, question, n_samples, 
        temperature, top_p, max_tokens, n_fewshots,
        is_diffusion, is_dream, diffusion_steps, diffusion_alg, verbose
    )
    
    correct_count = 0
    correct_examples = []
    incorrect_examples = []
    
    for idx, response in enumerate(responses):
        pred_answer = extract_answer(response)
        is_correct = pred_answer and check_answer(pred_answer, ground_truth)
        
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
            print(response)
            print(f"{'-'*80}")
            print(f"Extracted Answer: {pred_answer}")
            print(f"Correct: {is_correct}")
    
    if verbose:
        print(f"\nCorrect: {correct_count}/{n_samples}")
    
    return {
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pass@k on GSM8K or AIME25 with support for diffusion LLMs (Dream 7B, LLaDA 8B, D1-LLaDA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Dream 7B
  python evaluate_passk.py --model_name Dream-org/Dream-v0-Instruct-7B --n_samples 100 --subset_size 10
  
  # Evaluate LLaDA 8B
  python evaluate_passk.py --model_name GSAI-ML/LLaDA-8B-Instruct --n_samples 100 --subset_size 10
  
  # Evaluate autoregressive model
  python evaluate_passk.py --model_name meta-llama/Llama-3.1-8B-Instruct --n_samples 100 --subset_size 10
        """
    )
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or path (e.g., Dream-org/Dream-v0-Instruct-7B, GSAI-ML/LLaDA-8B-Instruct)')
    parser.add_argument('--model_type', type=str, default=None, choices=['autoregressive', 'diffusion', None],
                        help='Explicitly specify model type (auto-detected if not provided)')
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'aime25'],
                        help='Dataset to evaluate on (default: gsm8k)')
    parser.add_argument('--n_samples', type=int, default=512,
                        help='Number of samples per problem (default: 512)')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of problems to evaluate (default: all)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.6],
                        help='List of temperatures to evaluate (default: [0.6])')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling threshold (default: 0.95)')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum generation length (default: 256)')
    parser.add_argument('--n_fewshots', type=int, default=0,
                        help='Number of few-shot examples to include (default: 0, max: 8)')
    
    # Diffusion-specific parameters
    parser.add_argument('--diffusion_steps', type=int, default=512,
                        help='Number of diffusion steps for diffusion models (default: 512)')
    parser.add_argument('--diffusion_alg', type=str, default='entropy', 
                        choices=['origin', 'maskgit_plus', 'topk_margin', 'entropy'],
                        help='Remasking strategy for Dream models (default: entropy)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print generated outputs and detailed evaluation')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("RCoT-Diffusion-LLM Evaluation Framework")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Samples per problem: {args.n_samples}")
    if args.subset_size:
        print(f"Evaluating on subset: {args.subset_size} problems")
    print(f"{'='*80}\n")
    
    model, tokenizer, is_diffusion, is_dream = load_model_and_tokenizer(args.model_name, args.model_type)
    
    if args.dataset == 'gsm8k':
        problems = load_gsm8k(args.subset_size)
    elif args.dataset == 'aime25':
        problems = load_aime25(args.subset_size)
    
    for temperature in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Evaluating on {args.dataset.upper()} with temperature={temperature}, n_fewshots={args.n_fewshots}")
        if is_diffusion:
            print(f"Diffusion steps: {args.diffusion_steps}")
            if is_dream:
                print(f"Diffusion algorithm: {args.diffusion_alg}")
        print(f"{'='*60}\n")
        
        results = []
        
        for problem in tqdm(problems, desc=f"T={temperature}", disable=args.verbose):
            result = evaluate_problem(
                model, tokenizer, problem,
                args.n_samples, temperature,
                args.top_p, args.max_tokens,
                args.n_fewshots, is_diffusion, is_dream,
                args.diffusion_steps, args.diffusion_alg, args.verbose
            )
            results.append(result)
        
        # Create descriptive output filename
        model_short = args.model_name.split('/')[-1]
        output_file = os.path.join(
            args.output_dir,
            f"results_{model_short}_{args.dataset}_temp{temperature}_n{args.n_samples}_fewshot{args.n_fewshots}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to: {output_file}")
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
