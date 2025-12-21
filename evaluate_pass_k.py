#!/usr/bin/env python3
"""
Script to evaluate pass@k metrics on generation results.

This script demonstrates:
1. How to use pass_k.py with your results
2. Different formats of extracted_answer (single value vs list)
3. Normalization of extracted_answer formats
4. Computing pass@1, pass@k metrics
"""

import json
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
from metrics.pass_k import compute_pass_at_k, compute_detailed_metrics, pass_at_k


def normalize_answer(answer: Any) -> Union[float, str, None]:
    """
    Normalize extracted_answer to a single comparable value.
    
    Args:
        answer: Can be float, int, str, list, or None
        
    Returns:
        Normalized answer for comparison
    """
    if answer is None:
        return None
    
    # If it's already a number, return it
    if isinstance(answer, (int, float)):
        return float(answer)
    
    # If it's a list, try to extract the first valid answer
    if isinstance(answer, list):
        if len(answer) == 0:
            return None
        
        # Get the first element
        first_elem = answer[0]
        
        # If it's a string, try to extract boxed answer
        if isinstance(first_elem, str):
            extracted = extract_boxed_answer(first_elem)
            if extracted is not None:
                return extracted
            return first_elem
        
        # If it's already a number
        if isinstance(first_elem, (int, float)):
            return float(first_elem)
        
        return None
    
    # If it's a string, try to extract number or return as-is
    if isinstance(answer, str):
        # Try to parse as float
        try:
            return float(answer)
        except ValueError:
            # Check if it has boxed answer
            extracted = extract_boxed_answer(answer)
            if extracted is not None:
                return extracted
            return answer
    
    return None


def extract_boxed_answer(text: str) -> Union[float, str, None]:
    """
    Extract answer from \\boxed{...} format.
    
    Args:
        text: Text containing potential boxed answer
        
    Returns:
        Extracted answer or None
    """
    import re
    
    # Try to find \\boxed{...}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        answer_str = matches[-1]  # Take the last boxed answer
        
        # Try to convert to float
        try:
            # Remove common formatting
            answer_str = answer_str.replace(',', '').replace('$', '').strip()
            return float(answer_str)
        except ValueError:
            return answer_str
    
    return None


def load_generation_results(file_path: str) -> List[Dict]:
    """
    Load generation results from JSON file.
    
    Args:
        file_path: Path to the results JSON file
        
    Returns:
        List of generation dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data.get('generations', [])


def analyze_answer_formats(generations: List[Dict]) -> Dict[str, Any]:
    """
    Analyze different formats of extracted_answer in the dataset.
    
    Args:
        generations: List of generation results
        
    Returns:
        Dictionary with format statistics
    """
    formats = {
        'float': 0,
        'int': 0,
        'list': 0,
        'string': 0,
        'none': 0,
        'list_lengths': []
    }
    
    examples = {
        'float': [],
        'list': [],
        'string': []
    }
    
    for gen in generations:
        answer = gen.get('extracted_answer')
        
        if answer is None:
            formats['none'] += 1
        elif isinstance(answer, float):
            formats['float'] += 1
            if len(examples['float']) < 3:
                examples['float'].append({
                    'question': gen.get('question', '')[:100],
                    'answer': answer
                })
        elif isinstance(answer, int):
            formats['int'] += 1
        elif isinstance(answer, list):
            formats['list'] += 1
            formats['list_lengths'].append(len(answer))
            if len(examples['list']) < 3:
                examples['list'].append({
                    'question': gen.get('question', '')[:100],
                    'answer': answer[0] if answer else None,
                    'list_length': len(answer)
                })
        elif isinstance(answer, str):
            formats['string'] += 1
            if len(examples['string']) < 3:
                examples['string'].append({
                    'question': gen.get('question', '')[:100],
                    'answer': answer[:200]
                })
    
    return {
        'formats': formats,
        'examples': examples
    }


def prepare_pass_k_data(generations: List[Dict], samples_per_problem: int = 1) -> List[Dict]:
    """
    Prepare data for pass@k computation.
    
    Handles two formats:
    1. Single answer per question: extracted_answer is a single value
    2. Multiple answers per question: extracted_answer is a list of values
    
    Args:
        generations: List of generation results
        samples_per_problem: Number of samples per problem (expected)
        
    Returns:
        List of dictionaries with 'total_samples' and 'correct_count'
    """
    results = []
    
    for gen in generations:
        question = gen.get('question', '')
        ground_truth = normalize_answer(gen.get('ground_truth'))
        extracted_answers = gen.get('extracted_answer')
        
        # Check if extracted_answer is a list or single value
        if isinstance(extracted_answers, list):
            # Multiple samples per question (already in list format)
            correct_count = 0
            total_samples = len(extracted_answers)
            
            for extracted in extracted_answers:
                normalized_extracted = normalize_answer(extracted)
                
                is_correct = False
                if normalized_extracted is not None and ground_truth is not None:
                    if isinstance(normalized_extracted, (int, float)) and isinstance(ground_truth, (int, float)):
                        is_correct = abs(normalized_extracted - ground_truth) < 1e-4
                    else:
                        is_correct = str(normalized_extracted).strip().lower() == str(ground_truth).strip().lower()
                
                if is_correct:
                    correct_count += 1
            
            results.append({
                'total_samples': total_samples,
                'correct_count': correct_count,
                'question': question[:100],
                'extracted': extracted_answers[:3] if total_samples > 3 else extracted_answers,  # Show first 3
                'ground_truth': ground_truth
            })
        
        else:
            # Single sample per question
            extracted = normalize_answer(extracted_answers)
            
            is_correct = False
            if extracted is not None and ground_truth is not None:
                if isinstance(extracted, (int, float)) and isinstance(ground_truth, (int, float)):
                    is_correct = abs(extracted - ground_truth) < 1e-4
                else:
                    is_correct = str(extracted).strip().lower() == str(ground_truth).strip().lower()
            
            results.append({
                'total_samples': 1,
                'correct_count': 1 if is_correct else 0,
                'question': question[:100],
                'extracted': extracted,
                'ground_truth': ground_truth
            })
    
    return results


def compute_metrics(results_file: str, samples_per_problem: int = None, k_values: List[int] = None):
    """
    Compute pass@k metrics for a results file.
    
    Args:
        results_file: Path to results JSON file
        samples_per_problem: Number of samples per problem (auto-detected if None)
        k_values: List of k values to compute pass@k for (auto-detected if None)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {Path(results_file).name}")
    print(f"{'='*80}\n")
    
    # Load generations
    generations = load_generation_results(results_file)
    print(f"Loaded {len(generations)} generations")
    
    # Auto-detect samples_per_problem from the data
    if samples_per_problem is None:
        first_gen = generations[0] if generations else {}
        extracted = first_gen.get('extracted_answer')
        if isinstance(extracted, list):
            samples_per_problem = len(extracted)
            print(f"Auto-detected: {samples_per_problem} samples per problem")
        else:
            samples_per_problem = 1
            print(f"Auto-detected: Single sample per problem")
    
    # Auto-detect k_values based on samples_per_problem
    if k_values is None:
        if samples_per_problem == 1:
            k_values = [1]
        elif samples_per_problem <= 10:
            k_values = list(range(1, samples_per_problem + 1))
        else:
            # For large n, use powers of 2 and key values
            k_values = [1, 2, 4, 8, 16, 32]
            k_values = [k for k in k_values if k <= samples_per_problem]
            if samples_per_problem not in k_values:
                k_values.append(samples_per_problem)
        print(f"Auto-detected k_values: {k_values}")
    
    # Analyze answer formats
    print("\n--- Answer Format Analysis ---")
    format_analysis = analyze_answer_formats(generations)
    
    print("\nFormat Distribution:")
    for format_type, count in format_analysis['formats'].items():
        if format_type != 'list_lengths':
            print(f"  {format_type:10s}: {count:4d} ({count/len(generations)*100:5.1f}%)")
    
    if format_analysis['formats']['list_lengths']:
        list_lengths = format_analysis['formats']['list_lengths']
        print(f"\nList lengths (min/max/mean): "
              f"{min(list_lengths)}"
              f"/{max(list_lengths)}"
              f"/{np.mean(list_lengths):.1f}")
    
    # Prepare data for pass@k
    print("\n--- Computing Pass@k Metrics ---")
    pass_k_data = prepare_pass_k_data(generations, samples_per_problem)
    
    # Filter k_values to only those <= samples_per_problem
    valid_k_values = [k for k in k_values if k <= samples_per_problem]
    
    if not valid_k_values:
        print(f"Warning: No valid k values for samples_per_problem={samples_per_problem}")
        print(f"K values must be <= {samples_per_problem}")
        return
    
    # Compute metrics
    detailed_metrics = compute_detailed_metrics(pass_k_data, valid_k_values)
    
    print(f"\nProblems evaluated: {detailed_metrics['problems_evaluated']}")
    print(f"Total samples: {detailed_metrics['total_samples']}")
    print(f"Total correct: {detailed_metrics['total_correct']}")
    print(f"Correct rate: {detailed_metrics['correct_rate']:.4f} ({detailed_metrics['correct_rate']*100:.2f}%)")
    print(f"Accuracy (any correct): {detailed_metrics['accuracy']:.4f} ({detailed_metrics['accuracy']*100:.2f}%)")
    print(f"Avg correct per problem: {detailed_metrics['avg_correct_per_problem']:.2f}")
    
    print("\n" + "="*80)
    print("PASS@K RESULTS")
    print("="*80)
    for k, score in sorted(detailed_metrics['pass_at_k'].items()):
        print(f"  pass@{k:2d}: {score:.4f} ({score*100:.2f}%)")
    print("="*80)
    
    # Show some examples of correct/incorrect
    print("\n--- Sample Results ---")
    correct_examples = [r for r in pass_k_data if r['correct_count'] > 0][:3]
    incorrect_examples = [r for r in pass_k_data if r['correct_count'] == 0][:3]
    
    if correct_examples:
        print("\n✓ Correct examples:")
        for i, ex in enumerate(correct_examples, 1):
            print(f"  {i}. Q: {ex['question'][:80]}...")
            print(f"     Correct: {ex['correct_count']}/{ex['total_samples']}")
            if isinstance(ex.get('extracted'), list):
                print(f"     First answers: {ex.get('extracted')}")
            else:
                print(f"     Answer: {ex.get('extracted')}, GT: {ex.get('ground_truth')}")
    
    if incorrect_examples:
        print("\n✗ Incorrect examples:")
        for i, ex in enumerate(incorrect_examples, 1):
            print(f"  {i}. Q: {ex['question'][:80]}...")
            print(f"     Correct: {ex['correct_count']}/{ex['total_samples']}")
            if isinstance(ex.get('extracted'), list):
                print(f"     First answers: {ex.get('extracted')}")
            else:
                print(f"     Answer: {ex.get('extracted')}, GT: {ex.get('ground_truth')}")
    
    print(f"\n{'='*80}\n")
    
    return detailed_metrics


def main():
    """Main function to demonstrate usage."""
    
    # Example: Evaluate the LLaDA results with 32 samples per problem
    print("\n" + "="*80)
    print("Evaluating LLaDA-8B with 32 samples per problem")
    print("="*80)
    
    results_file = "/cephfs/users/bashir/RCoT-Diffusion-LLM/results/GSAI-ML_LLaDA-8B-Base_256_256_4_32_256_0.0_generations_testing_fast_dllm.json"
    
    if Path(results_file).exists():
        # Auto-detect everything - just pass the file!
        metrics = compute_metrics(
            results_file=results_file,
            # samples_per_problem and k_values will be auto-detected
        )
        
        # You can also manually specify:
        # metrics = compute_metrics(
        #     results_file=results_file,
        #     samples_per_problem=32,
        #     k_values=[1, 2, 4, 8, 16, 32]
        # )
    else:
        print(f"File not found: {results_file}")


if __name__ == "__main__":
    main()

