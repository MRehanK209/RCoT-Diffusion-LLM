#!/usr/bin/env python3
"""
Compare two model evaluation results:
1. Verify they used the same questions (seed verification)
2. Compute pass@k metrics for k=[2, 4, 8, 16, 32, 64, 128]
"""

import json
import re
import numpy as np
from typing import Optional, List, Dict, Any
import hashlib
from pathlib import Path

# Import pass@k functions from centralized metrics module
from metrics.pass_k import pass_at_k, compute_pass_at_k


# ============================================================================
# ANSWER EXTRACTION FUNCTIONS (User's specification)
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last match
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize answer for comparison."""
    if answer is None:
        return None
    
    answer = answer.strip()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace('%', '')
    answer = answer.strip()
    
    try:
        # Handle fractions
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    answer = str(num / denom)
        
        # Convert to number
        num = float(answer)
        if num == int(num):
            return str(int(num))
        else:
            return f"{num:.6f}".rstrip('0').rstrip('.')
    except:
        return answer.lower().strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from text, trying boxed first, then last number."""
    if text is None or not isinstance(text, str):
        return None
    
    # Try boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    
    # Fall back to last number
    last_num = extract_last_number(text)
    return normalize_answer(last_num)


def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth."""
    pred_norm = normalize_answer(str(prediction)) if prediction is not None else None
    gt_norm = normalize_answer(str(ground_truth)) if ground_truth is not None else None
    
    if pred_norm is None or gt_norm is None:
        return False
    
    # Exact string match
    if pred_norm == gt_norm:
        return True
    
    # Try numerical comparison with tolerance
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-4
    except:
        return pred_norm == gt_norm


# ============================================================================
# PASS@K METRIC FUNCTIONS
# ============================================================================
# Imported from metrics.pass_k module (centralized implementation)
# - pass_at_k(n, c, k): Calculate pass@k for a single problem
# - compute_pass_at_k(results, k_values): Compute pass@k for multiple k values


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_problem(generation_data: Dict, n_samples: int) -> Dict:
    """
    Evaluate a single problem with multiple samples.
    
    Args:
        generation_data: Dict with 'question', 'generations', 'ground_truth'
        n_samples: Expected number of samples
        
    Returns:
        Dict with evaluation results
    """
    question = generation_data.get('question', '')
    ground_truth = generation_data.get('ground_truth')
    generations = generation_data.get('generations', [])
    
    # Handle both list and single value
    if not isinstance(generations, list):
        generations = [generations]
    
    # Ensure we have the expected number of samples
    if len(generations) != n_samples:
        print(f"WARNING: Expected {n_samples} samples, got {len(generations)} for question: {question[:50]}...")
    
    correct_count = 0
    correct_examples = []
    incorrect_examples = []
    extracted_answers = []
    
    for idx, response in enumerate(generations):
        # Extract answer from generation
        pred_answer = extract_answer(response)
        extracted_answers.append(pred_answer)
        
        # Check if correct
        is_correct = check_answer(pred_answer, ground_truth)
        
        if is_correct:
            correct_count += 1
            if len(correct_examples) < 3:
                correct_examples.append({
                    'response': response[:200],
                    'extracted_answer': pred_answer
                })
        else:
            if len(incorrect_examples) < 3:
                incorrect_examples.append({
                    'response': response[:200],
                    'extracted_answer': pred_answer
                })
    
    return {
        'question': question,
        'answer': ground_truth,
        'correct_count': correct_count,
        'total_samples': len(generations),
        'extracted_answers': extracted_answers,
        'correct_examples': correct_examples,
        'incorrect_examples': incorrect_examples
    }


def load_and_evaluate(json_file: str, n_samples: int) -> tuple:
    """
    Load JSON file and evaluate all problems.
    
    Args:
        json_file: Path to results JSON file
        n_samples: Expected number of samples per problem
        
    Returns:
        Tuple of (questions, evaluation_results)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    generations = data.get('generations', [])
    
    questions = []
    results = []
    
    for gen in generations:
        # Extract question for seed verification
        questions.append(gen.get('question', ''))
        
        # Evaluate problem
        result = evaluate_problem(gen, n_samples)
        results.append(result)
    
    return questions, results


# ============================================================================
# SEED VERIFICATION
# ============================================================================

def verify_seeds(questions1: List[str], questions2: List[str], 
                 label1: str, label2: str) -> bool:
    """
    Verify that two runs used the same questions.
    
    Args:
        questions1: List of questions from run 1
        questions2: List of questions from run 2
        label1: Label for run 1
        label2: Label for run 2
        
    Returns:
        True if seeds match, False otherwise
    """
    print("\n" + "="*80)
    print("SEED VERIFICATION")
    print("="*80)
    
    # Compute hashes
    hash1 = hashlib.md5(''.join(questions1).encode()).hexdigest()
    hash2 = hashlib.md5(''.join(questions2).encode()).hexdigest()
    
    print(f"\n{label1}:")
    print(f"  Questions: {len(questions1)}")
    print(f"  MD5 Hash:  {hash1}")
    
    print(f"\n{label2}:")
    print(f"  Questions: {len(questions2)}")
    print(f"  MD5 Hash:  {hash2}")
    
    if hash1 == hash2 and len(questions1) == len(questions2):
        print("\n SEED VERIFICATION PASSED!")
        print("   - Both runs used IDENTICAL questions")
        print("   - Question order is IDENTICAL")
        print("   - Results are comparable")
        
        # Show sample questions
        print(f"\nSample questions (first 3):")
        for i in range(min(3, len(questions1))):
            print(f"  {i+1}. {questions1[i][:70]}...")
        
        return True
    else:
        print("\n SEED VERIFICATION FAILED!")
        print("   - Questions are DIFFERENT between runs")
        print("   - Results may not be comparable")
        
        # Find first difference
        for i in range(min(len(questions1), len(questions2))):
            if questions1[i] != questions2[i]:
                print(f"\nFirst difference at position {i+1}:")
                print(f"  {label1}: {questions1[i][:70]}...")
                print(f"  {label2}: {questions2[i][:70]}...")
                break
        
        return False


# ============================================================================
# RESULTS COMPARISON
# ============================================================================

def compare_results(results1: List[Dict], results2: List[Dict], 
                   label1: str, label2: str, k_values: List[int]):
    """
    Compare pass@k results between two models.
    
    Args:
        results1: Evaluation results from model 1
        results2: Evaluation results from model 2
        label1: Label for model 1
        label2: Label for model 2
        k_values: List of k values to compute pass@k for
    """
    print("\n" + "="*80)
    print("PASS@K METRICS COMPARISON")
    print("="*80)
    
    # Compute pass@k for both models
    pass_k_1 = compute_pass_at_k(results1, k_values)
    pass_k_2 = compute_pass_at_k(results2, k_values)
    
    # Compute basic statistics
    total_samples_1 = sum(r['total_samples'] for r in results1)
    total_correct_1 = sum(r['correct_count'] for r in results1)
    accuracy_1 = sum(1 for r in results1 if r['correct_count'] > 0) / len(results1)
    
    total_samples_2 = sum(r['total_samples'] for r in results2)
    total_correct_2 = sum(r['correct_count'] for r in results2)
    accuracy_2 = sum(1 for r in results2 if r['correct_count'] > 0) / len(results2)
    
    # Print comparison table
    print(f"\n{'Metric':<20} {label1:>20} {label2:>20} {'Δ':>15}")
    print("-"*80)
    print(f"{'Problems':<20} {len(results1):>20} {len(results2):>20}")
    print(f"{'Total Samples':<20} {total_samples_1:>20} {total_samples_2:>20}")
    print(f"{'Total Correct':<20} {total_correct_1:>20} {total_correct_2:>20}")
    print(f"{'Correct Rate':<20} {total_correct_1/total_samples_1:>19.2%} {total_correct_2/total_samples_2:>19.2%} {(total_correct_2/total_samples_2 - total_correct_1/total_samples_1):>+14.2%}")
    print(f"{'Accuracy (Any)':<20} {accuracy_1:>19.2%} {accuracy_2:>19.2%} {(accuracy_2 - accuracy_1):>+14.2%}")
    
    print("\n" + "-"*80)
    print(f"{'Pass@k':<20} {label1:>20} {label2:>20} {'Δ':>15}")
    print("-"*80)
    
    for k in sorted(k_values):
        score1 = pass_k_1[k]
        score2 = pass_k_2[k]
        delta = score2 - score1
        winner = "" if abs(delta) < 0.001 else ("✅" if delta > 0 else "")
        print(f"{'pass@' + str(k):<20} {score1:>19.2%} {score2:>19.2%} {delta:>+14.2%} {winner}")
    
    print("="*80)
    
    # Detailed statistics
    print(f"\n{label1} Statistics:")
    print(f"  Avg correct per problem: {np.mean([r['correct_count'] for r in results1]):.2f}")
    print(f"  Min correct: {min(r['correct_count'] for r in results1)}")
    print(f"  Max correct: {max(r['correct_count'] for r in results1)}")
    print(f"  Std correct: {np.std([r['correct_count'] for r in results1]):.2f}")
    
    print(f"\n{label2} Statistics:")
    print(f"  Avg correct per problem: {np.mean([r['correct_count'] for r in results2]):.2f}")
    print(f"  Min correct: {min(r['correct_count'] for r in results2)}")
    print(f"  Max correct: {max(r['correct_count'] for r in results2)}")
    print(f"  Std correct: {np.std([r['correct_count'] for r in results2]):.2f}")
    
    # Show example problems
    print(f"\n{'='*80}")
    print("SAMPLE PROBLEM COMPARISON")
    print("="*80)
    
    # Find problems where one model did much better
    diffs = []
    for r1, r2 in zip(results1, results2):
        diff = r2['correct_count'] - r1['correct_count']
        diffs.append((diff, r1, r2))
    
    diffs_sorted = sorted(diffs, key=lambda x: x[0])
    
    # Show where label2 did much better
    print(f"\nProblems where {label2} outperformed {label1} (top 3):")
    for diff, r1, r2 in diffs_sorted[-3:]:
        if diff > 0:
            print(f"\n  Q: {r1['question'][:70]}...")
            print(f"     {label1}: {r1['correct_count']}/{r1['total_samples']} correct")
            print(f"     {label2}: {r2['correct_count']}/{r2['total_samples']} correct")
            print(f"     Ground Truth: {r1['answer']}")
    
    # Show where label1 did much better
    print(f"\nProblems where {label1} outperformed {label2} (top 3):")
    for diff, r1, r2 in diffs_sorted[:3]:
        if diff < 0:
            print(f"\n  Q: {r1['question'][:70]}...")
            print(f"     {label1}: {r1['correct_count']}/{r1['total_samples']} correct")
            print(f"     {label2}: {r2['correct_count']}/{r2['total_samples']} correct")
            print(f"     Ground Truth: {r1['answer']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main comparison function."""
    # File paths
    dream_file = "/cephfs/users/bashir/RCoT-Diffusion-LLM/results/Dream-org_Dream-v0-Base-7B_256_256_4_128_256_0.7_generations_testing_fast_dllm.json"
    qwen_file = "/cephfs/users/bashir/RCoT-Diffusion-LLM/results/Qwen_Qwen2.5-7B-base_256_4_128_256_0.7_generations_ar.json"
    
    # Configuration
    n_samples = 128
    k_values = [2, 4, 8, 16, 32, 64, 128]
    
    print("="*80)
    print("MODEL COMPARISON: Dream vs Qwen")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Samples per problem: {n_samples}")
    print(f"  K values: {k_values}")
    print(f"  Dream file: {Path(dream_file).name}")
    print(f"  Qwen file:  {Path(qwen_file).name}")
    
    # Load and evaluate both models
    print("\n" + "-"*80)
    print("Loading Dream results...")
    dream_questions, dream_results = load_and_evaluate(dream_file, n_samples)
    print(f"  Loaded {len(dream_results)} problems")
    
    print("\nLoading Qwen results...")
    qwen_questions, qwen_results = load_and_evaluate(qwen_file, n_samples)
    print(f"  Loaded {len(qwen_results)} problems")
    
    # Verify seeds
    seeds_match = verify_seeds(
        dream_questions, qwen_questions,
        "Dream-v0-Base", "Qwen2.5-7B-base"
    )
    
    if not seeds_match:
        print("\n WARNING: Seeds don't match! Results may not be comparable.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Compare results
    compare_results(
        dream_results, qwen_results,
        "Dream-v0-Base", "Qwen2.5-7B-base",
        k_values
    )
    
    # Save detailed results
    output_file = "model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'dream': {
                'file': dream_file,
                'n_problems': len(dream_results),
                'n_samples': n_samples,
                'pass_at_k': compute_pass_at_k(dream_results, k_values),
                'total_correct': sum(r['correct_count'] for r in dream_results),
                'total_samples': sum(r['total_samples'] for r in dream_results),
            },
            'qwen': {
                'file': qwen_file,
                'n_problems': len(qwen_results),
                'n_samples': n_samples,
                'pass_at_k': compute_pass_at_k(qwen_results, k_values),
                'total_correct': sum(r['correct_count'] for r in qwen_results),
                'total_samples': sum(r['total_samples'] for r in qwen_results),
            },
            'seeds_match': seeds_match
        }, f, indent=2)
    
    print(f"\n Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
