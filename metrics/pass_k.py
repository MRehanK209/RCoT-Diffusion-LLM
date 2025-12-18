import numpy as np
from scipy.special import comb
from typing import List, Dict, Any


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric for a single problem.
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of attempts to consider
    
    Returns:
        Probability of at least one correct solution in k attempts
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    
    return 1.0 - (comb(n - c, k) / comb(n, k))


def compute_pass_at_k(results: List[Dict], k_values: List[int]) -> Dict[int, float]:
    """
    Compute pass@k for multiple k values across all problems.
    
    Args:
        results: List of evaluation results, each with 'total_samples' and 'correct_count'
        k_values: List of k values to compute pass@k for
    
    Returns:
        Dictionary mapping k values to average pass@k scores
    """
    pass_k_results = {}
    for k in k_values:
        scores = []
        for result in results:
            n = result['total_samples']
            c = result['correct_count']
            scores.append(pass_at_k(n, c, k))
        pass_k_results[k] = float(np.mean(scores))
    
    return pass_k_results

def compute_detailed_metrics(results: List[Dict], k_values: List[int]) -> Dict[str, Any]:
    """
    Compute detailed metrics including pass@k, accuracy, and statistics.
    
    Args:
        results: List of evaluation results
        k_values: List of k values for pass@k
    
    Returns:
        Dictionary with detailed metrics
    """
    # Pass@k scores
    pass_k = compute_pass_at_k(results, k_values)
    
    # Accuracy (at least one correct)
    problems_with_correct = sum(1 for r in results if r['correct_count'] > 0)
    accuracy = problems_with_correct / len(results) if results else 0
    
    # Average correct per problem
    avg_correct = np.mean([r['correct_count'] for r in results]) if results else 0
    
    # Correct rate (total correct / total samples)
    total_correct = sum(r['correct_count'] for r in results)
    total_samples = sum(r['total_samples'] for r in results)
    correct_rate = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        'pass_at_k': pass_k,
        'accuracy': float(accuracy),
        'avg_correct_per_problem': float(avg_correct),
        'correct_rate': float(correct_rate),
        'problems_evaluated': len(results),
        'total_correct': int(total_correct),
        'total_samples': int(total_samples)
    }