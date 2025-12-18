import re
import json
import sys
import signal
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager


# Timeout handling for code execution
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_code_from_response(response: str, entry_point: str = None) -> str:
    """
    Extract Python code from model response.
    Handles various formats: raw code, markdown code blocks, etc.
    """
    # Try to extract code from markdown code blocks
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the last code block (most likely the complete solution)
        return matches[-1].strip()
    
    # If no code blocks, try to find the function definition
    if entry_point:
        # Look for the function definition and everything after
        func_pattern = rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\).*)'
        match = re.search(func_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: return the response as-is (might be raw code)
    return response.strip()


def execute_code_safely(code: str, test_code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Execute code in a sandboxed subprocess with timeout.
    
    Args:
        code: The code to execute (function definition)
        test_code: The test code to run against the function
        timeout: Maximum execution time in seconds
    
    Returns:
        Tuple of (passed: bool, error_message: str)
    """
    # Combine the code and test
    full_code = f"{code}\n\n{test_code}\n\ncheck({code.split('(')[0].split()[-1]})"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:500]  # Truncate error message
    
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(temp_file)
        except:
            pass


def check_code_correctness(
    generated_code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: int = 5
) -> bool:
    """
    Check if generated code passes the test cases.
    
    Args:
        generated_code: The model's generated code completion
        prompt: The original function prompt/signature
        test: The test code string
        entry_point: The function name to test
        timeout: Maximum execution time in seconds
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Extract the actual code from the response
    code = extract_code_from_response(generated_code, entry_point)
    
    # If the prompt is not in the code, prepend it
    if prompt.strip() not in code:
        full_code = prompt + code
    else:
        full_code = code
    
    # Execute and check
    passed, error = execute_code_safely(full_code, test, timeout)
    return passed


def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> Optional[str]:
    if answer is None:
        return None
    
    answer = answer.strip()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace('%', '')
    answer = answer.strip()
    
    try:
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    answer = str(num / denom)
        
        num = float(answer)
        if num == int(num):
            return str(int(num))
        else:
            return f"{num:.6f}".rstrip('0').rstrip('.')
    except:
        return answer.lower().strip()


def extract_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    
    last_num = extract_last_number(text)
    return normalize_answer(last_num)


def check_answer(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm is None or gt_norm is None:
        return False
    
    if pred_norm == gt_norm:
        return True
    
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        return abs(pred_float - gt_float) < 1e-4
    except:
        return pred_norm == gt_norm


def _extract_tag_content(text: str, tag: str) -> Optional[str]:
    """
    Extract the last occurrence of content inside <tag>...</tag>.
    Returns None if not found.
    """
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


# -----------------------------
# Sudoku (4x4) helpers
# -----------------------------


def extract_sudoku_answer(text: str) -> Optional[str]:
    """
    Extract a 16-character 4x4 Sudoku solution string from model output.

    Preferred format (d1-style):
      <answer>
      1234... (16 digits)
      </answer>
    """
    ans = _extract_tag_content(text, "answer")
    if ans is not None:
        ans = re.sub(r"\s+", "", ans)
        # If answer includes other text, keep only digits.
        digits_only = "".join(re.findall(r"[0-9]", ans))
        if digits_only:
            ans = digits_only

    # Fallback: find any contiguous 16-digit string
    if not ans:
        m = re.search(r"([0-9]{16})", text)
        ans = m.group(1) if m else None

    if ans is None:
        return None

    ans = ans.strip()
    if len(ans) != 16:
        return None
    return ans


def _is_valid_4x4_sudoku_solution(solution: str) -> bool:
    if solution is None or len(solution) != 16:
        return False
    if not re.fullmatch(r"[1-4]{16}", solution):
        return False

    digits = set("1234")

    # Rows
    for r in range(4):
        row = solution[r * 4 : (r + 1) * 4]
        if set(row) != digits:
            return False

    # Cols
    for c in range(4):
        col = solution[c] + solution[c + 4] + solution[c + 8] + solution[c + 12]
        if set(col) != digits:
            return False

    # 2x2 boxes
    for br in (0, 2):
        for bc in (0, 2):
            box = []
            for r in range(br, br + 2):
                for c in range(bc, bc + 2):
                    box.append(solution[r * 4 + c])
            if set(box) != digits:
                return False

    return True


def sudoku_empty_cell_accuracy(pred: str, puzzle: str, solution: str) -> Tuple[int, int, float]:
    """
    Accuracy only over originally empty cells (where puzzle has '0').
    Returns: (correct_cells, empty_cells, accuracy)
    """
    if puzzle is None or solution is None or len(puzzle) != 16 or len(solution) != 16:
        return 0, 0, 0.0

    empty_indices = [i for i, ch in enumerate(puzzle) if ch == "0"]
    empty_cells = len(empty_indices)
    if empty_cells == 0:
        return 0, 0, 0.0

    if pred is None or len(pred) != 16:
        return 0, empty_cells, 0.0

    correct_cells = sum(1 for i in empty_indices if pred[i] == solution[i])
    return correct_cells, empty_cells, correct_cells / empty_cells


def check_sudoku_solution(pred: Optional[str], puzzle: str, solution: str) -> bool:
    """
    Strict correctness: must match ground-truth solution and be a valid 4x4 grid.
    Also enforces puzzle givens.
    """
    if pred is None:
        return False
    pred = pred.strip()
    if len(pred) != 16:
        return False

    # Enforce givens
    if puzzle and len(puzzle) == 16:
        for i, ch in enumerate(puzzle):
            if ch != "0" and pred[i] != ch:
                return False

    if not _is_valid_4x4_sudoku_solution(pred):
        return False

    return pred == solution


# -----------------------------
# Countdown helpers
# -----------------------------


def extract_countdown_equation(text: str) -> Optional[str]:
    """
    Extract an arithmetic expression for Countdown.

    Preferred format (d1-style):
      <answer>
      \\boxed{...}
      </answer>
    """
    ans = _extract_tag_content(text, "answer")
    candidate = ans if ans is not None else text

    # Try boxed
    m = re.search(r"\\boxed\s*\{([^}]*)\}", candidate)
    if m:
        expr = m.group(1).strip()
    else:
        # Fallback: last non-empty line
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        expr = lines[-1] if lines else ""

    expr = (
        expr.replace(r"\div", "/")
        .replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
    )
    return expr.strip() if expr.strip() else None


def _validate_countdown_equation(expr: str, available_numbers: list) -> bool:
    """
    Validate that expression only uses the available numbers and each number exactly once.
    """
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", expr)]
        return sorted(numbers_in_eq) == sorted(list(available_numbers))
    except Exception:
        return False


def _safe_eval_arithmetic(expr: str) -> Optional[float]:
    """
    Safely evaluate a basic arithmetic expression containing only numbers, + - * / ( ) . and whitespace.
    Returns None on failure.
    """
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, expr):
            return None
        return eval(expr, {"__builtins__": None}, {})
    except Exception:
        return None


def check_countdown_solution(
    expr: Optional[str],
    numbers: list,
    target: int,
    tol: float = 1e-5,
) -> bool:
    if expr is None:
        return False
    if not _validate_countdown_equation(expr, numbers):
        return False
    val = _safe_eval_arithmetic(expr)
    if val is None:
        return False
    try:
        return abs(float(val) - float(target)) < tol
    except Exception:
        return False


# -----------------------------
# Trip-planning helpers (deterministic constraints)
# -----------------------------


def extract_trip_plan(text: str) -> Optional[dict]:
    """
    Extract a JSON object from model output.

    Preferred format:
      <answer>
      { ...valid JSON... }
      </answer>
    """
    candidate = _extract_tag_content(text, "answer") or text

    # Try direct JSON parse first
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Fallback: grab the first {...} block (greedy but usually works for one JSON object)
    m = re.search(r"(\{[\s\S]*\})", candidate)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def check_trip_plan(plan: Optional[dict], constraints: dict) -> Tuple[bool, Dict[str, Any]]:
    """
    Deterministic rubric:
      - If constraints include num_days, require len(plan['days']) == num_days
      - If constraints include must_visit_cities, require all are present in day['city']
      - If constraints include budget_usd, require plan['total_cost_usd'] <= budget_usd (if present)
      - If constraints include must_include (strings), require each appears in any activity string

    Returns: (is_valid, details)
    """
    details: Dict[str, Any] = {}
    if plan is None or not isinstance(plan, dict):
        return False, {"reason": "no_json_plan"}

    days = plan.get("days")
    if not isinstance(days, list) or len(days) == 0:
        return False, {"reason": "missing_days_list"}

    # num_days
    num_days = constraints.get("num_days")
    if isinstance(num_days, int):
        details["num_days_expected"] = num_days
        details["num_days_actual"] = len(days)
        if len(days) != num_days:
            return False, {**details, "reason": "wrong_num_days"}

    # must_visit_cities
    must_cities = constraints.get("must_visit_cities") or constraints.get("cities")
    visited = []
    for d in days:
        if isinstance(d, dict) and isinstance(d.get("city"), str):
            visited.append(d["city"].strip())
    visited_lower = {c.lower() for c in visited if c}
    if isinstance(must_cities, list) and must_cities:
        missing = [c for c in must_cities if str(c).lower() not in visited_lower]
        details["missing_cities"] = missing
        if missing:
            return False, {**details, "reason": "missing_required_cities"}

    # budget_usd
    budget = constraints.get("budget_usd")
    total_cost = plan.get("total_cost_usd")
    if budget is not None and total_cost is not None:
        try:
            details["budget_usd"] = float(budget)
            details["total_cost_usd"] = float(total_cost)
            if float(total_cost) > float(budget):
                return False, {**details, "reason": "over_budget"}
        except Exception:
            return False, {**details, "reason": "budget_parse_error"}

    # must_include
    must_include = constraints.get("must_include")
    if isinstance(must_include, list) and must_include:
        activities_text = ""
        for d in days:
            if isinstance(d, dict):
                acts = d.get("activities")
                if isinstance(acts, list):
                    activities_text += " ".join(str(a) for a in acts) + " "
                elif isinstance(acts, str):
                    activities_text += acts + " "
        activities_text = activities_text.lower()
        missing_items = [str(x) for x in must_include if str(x).lower() not in activities_text]
        details["missing_must_include"] = missing_items
        if missing_items:
            return False, {**details, "reason": "missing_required_items"}

    return True, {**details, "reason": "ok"}