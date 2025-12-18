import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from datasets import load_dataset


_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOCAL_DATASET_DIR = _REPO_ROOT / "dataset"


def _default_dataset_path(filename: str, data_dir: Optional[str] = None) -> str:
    base = Path(data_dir) if data_dir else _DEFAULT_LOCAL_DATASET_DIR
    return str(base / filename)


def load_gsm8k(subset_size: int = None) -> List[Dict]:
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    problems = []
    for idx, item in enumerate(dataset):
        if subset_size and idx >= subset_size:
            break
        
        problems.append({
            'idx': idx,
            'question': item['question'],
            'answer': item['answer'].split('####')[1].strip()
        })
    
    print(f"Loaded {len(problems)} problems from GSM8K")
    return problems


def load_gsm8k_fewshot(n_examples: int = 8) -> List[Dict]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    examples = []
    for idx, item in enumerate(dataset):
        if idx >= n_examples:
            break
        
        full_answer = item['answer']
        reasoning = full_answer.split('####')[0].strip()
        numeric_answer = full_answer.split('####')[1].strip()
        
        examples.append({
            'question': item['question'],
            'answer': f"{reasoning} \\boxed{{{numeric_answer}}}"
        })
    
    return examples


def load_aime25(subset_size: int = None) -> List[Dict]:
    print("Loading AIME25 dataset...")
    dataset = load_dataset("math-ai/aime25", split="test")
    
    problems = []
    for idx, item in enumerate(dataset):
        if subset_size and idx >= subset_size:
            break
        
        problems.append({
            'idx': idx,
            'question': item['problem'],
            'answer': item['answer']
        })
    
    print(f"Loaded {len(problems)} problems from AIME25")
    return problems


def load_sudoku_4x4(
    subset_size: int = None,
    data_dir: Optional[str] = None,
    file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load 4x4 Sudoku puzzles.

    Expected CSV columns: Puzzle, Solution
    - Puzzle: 16-char string (row-major), '0' denotes empty cells
    - Solution: 16-char string (digits 1-4)
    """
    path = file_path or _default_dataset_path("4x4_test_sudoku.csv", data_dir=data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sudoku dataset file not found: {path}\n"
            f"Provide --data_dir pointing to a folder containing 4x4_test_sudoku.csv, "
            f"or pass file_path explicitly."
        )

    problems: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if subset_size and idx >= subset_size:
                break
            puzzle = str(row["Puzzle"]).strip()
            solution = str(row["Solution"]).strip()
            problems.append(
                {
                    "idx": idx,
                    "question": f"Solve the following Sudoku puzzle: {puzzle}",
                    "answer": solution,
                    "puzzle": puzzle,
                }
            )

    print(f"Loaded {len(problems)} problems from 4x4 Sudoku ({path})")
    return problems


def load_countdown_cd3(
    subset_size: int = None,
    data_dir: Optional[str] = None,
    file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load Countdown (CD3) dataset.

    Expected JSONL lines with fields:
      - input: comma-separated ints, e.g. "30,100,93"
      - output: target int as string, e.g. "23"
    """
    path = file_path or _default_dataset_path("countdown_cd3_test.jsonl", data_dir=data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Countdown dataset file not found: {path}\n"
            f"Provide --data_dir pointing to a folder containing countdown_cd3_test.jsonl, "
            f"or pass file_path explicitly."
        )

    problems: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if subset_size and idx >= subset_size:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            numbers_str = item["input"]
            numbers = [int(num) for num in numbers_str.split(",")]
            target = int(item["output"])
            question = f"Numbers: {numbers}\nTarget: {target}"
            problems.append(
                {
                    "idx": idx,
                    "question": question,
                    "answer": {"numbers": numbers, "target": target},
                    "numbers": numbers,
                    "target": target,
                }
            )

    print(f"Loaded {len(problems)} problems from Countdown CD3 ({path})")
    return problems


def load_trip_planning(
    subset_size: int = None,
    data_dir: Optional[str] = None,
    file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a trip-planning dataset from JSONL.

    Expected JSONL schema per line:
      - query: str (user request)
      - constraints: dict (deterministic constraints to validate)
      - (optional) reference: str|dict (human reference; not required for deterministic checking)
    """
    path = file_path or _default_dataset_path("trip_planning_sample.jsonl", data_dir=data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trip-planning dataset file not found: {path}\n"
            f"Provide --data_dir pointing to a folder containing trip_planning_sample.jsonl (or your own JSONL), "
            f"or pass file_path explicitly."
        )

    problems: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if subset_size and idx >= subset_size:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            query = str(item["query"])
            constraints = item.get("constraints", {})
            problems.append(
                {
                    "idx": idx,
                    "question": query,
                    "answer": constraints,  # evaluated via constraint satisfaction
                    "constraints": constraints,
                    "reference": item.get("reference"),
                }
            )

    print(f"Loaded {len(problems)} problems from Trip Planning ({path})")
    return problems


def load_humaneval(subset_size: int = None) -> List[Dict]:
    """
    Load HumanEval dataset for code generation evaluation.
    
    Returns problems with:
        - idx: problem index
        - task_id: HumanEval task identifier (e.g., 'HumanEval/0')
        - question: the function prompt to complete
        - prompt: same as question (for compatibility)
        - canonical_solution: reference solution
        - test: test cases as a string
        - entry_point: function name to test
    """
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    problems = []
    for idx, item in enumerate(dataset):
        if subset_size and idx >= subset_size:
            break
        
        problems.append({
            'idx': idx,
            'task_id': item['task_id'],
            'question': item['prompt'],
            'prompt': item['prompt'],
            'canonical_solution': item['canonical_solution'],
            'test': item['test'],
            'entry_point': item['entry_point'],
            'answer': item['canonical_solution']  # For compatibility with evaluate_problem
        })
    
    print(f"Loaded {len(problems)} problems from HumanEval")
    return problems


def load_humaneval_fewshot(n_examples: int = 3) -> List[Dict]:
    """
    Load few-shot examples for HumanEval from the training portion (first few problems).
    """
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    examples = []
    # Use problems from index 50+ as few-shot examples (avoiding test problems)
    start_idx = 50
    for idx, item in enumerate(dataset):
        if idx < start_idx:
            continue
        if len(examples) >= n_examples:
            break
        
        examples.append({
            'prompt': item['prompt'],
            'solution': item['canonical_solution']
        })
    
    return examples