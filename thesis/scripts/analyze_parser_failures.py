#!/usr/bin/env python3
"""Quantify parser failure rates and parser-sensitivity effects for thesis runs."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from dataset.countdown import (
    _COUNTDOWN_BINARY_EQ_RE,
    _extract_leading_subequations,
    _normalize_countdown_prediction,
    cd_score_single,
)
from dataset.trip_planning import parse_trip_response, trip_score_single
from metrics.parsers import Parser


THESIS = ROOT / "thesis"
TABLES = THESIS / "tables"
IMAGES = THESIS / "images"


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    benchmark: str
    condition: str
    model: str
    paradigm: str
    prompt_mode: str
    path: str


def run_specs() -> list[RunSpec]:
    return [
        # GSM8K main and prompt/few-shot diagnostics
        RunSpec("gsm8k", "base_4shot_n128", "LLaDA", "dLLM", "base_native", "results/milestone2_gsm8k_base/GSAI-ML_LLaDA-8B-Base_256_256_32_1_0.7_4_128_128_generations_base_native_fast_dllm.json"),
        RunSpec("gsm8k", "base_4shot_n128", "Dream", "dLLM", "base_native", "results/milestone2_gsm8k_base/Dream-org_Dream-v0-Base-7B_256_256_32_1_0.7_4_128_128_generations_base_native_fast_dllm.json"),
        RunSpec("gsm8k", "base_4shot_n128", "Qwen", "AR", "base_native", "results/milestone2_gsm8k_base/Qwen_Qwen2.5-7B_256_1_0.7_4_128_128_generations_base_native_vllm.json"),
        RunSpec("gsm8k", "base_4shot_n128", "Llama", "AR", "base_native", "results/milestone2_gsm8k_base/meta-llama_Llama-3.1-8B_256_1_0.7_4_128_128_generations_base_native_vllm.json"),
        RunSpec("gsm8k", "instruct_0shot_n16", "LLaDA", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct_0shot/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_0_256_16_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_0shot_n16", "Dream", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct_0shot/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_0_256_16_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_0shot_n16", "Qwen", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct_0shot/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_0_256_16_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_0shot_n16", "Llama", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct_0shot/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_0_256_16_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_4shot_n16", "LLaDA", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct_4shot/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_4_256_16_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_4shot_n16", "Dream", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct_4shot/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_4_256_16_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_4shot_n16", "Qwen", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct_4shot/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_4_256_16_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_4shot_n16", "Llama", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct_4shot/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_4_256_16_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_templated_n128", "LLaDA", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_4_128_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_templated_n128", "Dream", "dLLM", "instruct_templated", "results/milestone2_gsm8k_instruct/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_4_128_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_templated_n128", "Qwen", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_4_128_128_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_templated_n128", "Llama", "AR", "instruct_templated", "results/milestone2_gsm8k_instruct/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_4_128_128_generations_instruct_templated_vllm.json"),
        RunSpec("gsm8k", "instruct_flat_n128", "LLaDA", "dLLM", "instruct_flat", "results/milestone2_gsm8k_instruct/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_4_128_128_generations_instruct_flat_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_flat_n128", "Dream", "dLLM", "instruct_flat", "results/milestone2_gsm8k_instruct/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_4_128_128_generations_instruct_flat_fast_dllm.json"),
        RunSpec("gsm8k", "instruct_flat_n128", "Qwen", "AR", "instruct_flat", "results/milestone2_gsm8k_instruct/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_4_128_128_generations_instruct_flat_vllm.json"),
        RunSpec("gsm8k", "instruct_flat_n128", "Llama", "AR", "instruct_flat", "results/milestone2_gsm8k_instruct/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_4_128_128_generations_instruct_flat_vllm.json"),
        # Countdown main and prompt diagnostics
        RunSpec("countdown", "base_refresh", "LLaDA", "dLLM", "base_native", "results/milestone2_countdown_base_refresh/GSAI-ML_LLaDA-8B-Base_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json"),
        RunSpec("countdown", "base_refresh", "Dream", "dLLM", "base_native", "results/milestone2_countdown_base_refresh/Dream-org_Dream-v0-Base-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json"),
        RunSpec("countdown", "base_refresh", "Qwen", "AR", "base_native", "results/milestone2_countdown_base_refresh/Qwen_Qwen2.5-7B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json"),
        RunSpec("countdown", "base_refresh", "Llama", "AR", "base_native", "results/milestone2_countdown_base_refresh/meta-llama_Llama-3.1-8B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json"),
        RunSpec("countdown", "instruct_refresh", "LLaDA", "dLLM", "instruct_templated", "results/milestone2_countdown_instruct_refresh/GSAI-ML_LLaDA-8B-Instruct_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("countdown", "instruct_refresh", "Dream", "dLLM", "instruct_templated", "results/milestone2_countdown_instruct_refresh/Dream-org_Dream-v0-Instruct-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("countdown", "instruct_refresh", "Qwen", "AR", "instruct_templated", "results/milestone2_countdown_instruct_refresh/Qwen_Qwen2.5-7B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json"),
        RunSpec("countdown", "instruct_refresh", "Llama", "AR", "instruct_templated", "results/milestone2_countdown_instruct_refresh/meta-llama_Llama-3.1-8B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json"),
        RunSpec("countdown", "prompt_diag", "LLaDA", "dLLM", "instruct_templated", "results/GSAI-ML_LLaDA-8B-Instruct_countdown_cd4_32_32_32_8_0.7_8_512_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("countdown", "prompt_diag", "LLaDA", "dLLM", "instruct_flat", "results/GSAI-ML_LLaDA-8B-Instruct_countdown_cd4_32_32_32_8_0.7_8_512_128_generations_instruct_flat_fast_dllm.json"),
        RunSpec("countdown", "prompt_diag", "Dream", "dLLM", "instruct_templated", "results/Dream-org_Dream-v0-Instruct-7B_countdown_cd4_32_32_32_8_0.7_8_512_128_generations_instruct_templated_fast_dllm.json"),
        RunSpec("countdown", "prompt_diag", "Dream", "dLLM", "instruct_flat", "results/Dream-org_Dream-v0-Instruct-7B_countdown_cd4_32_32_32_8_0.7_8_512_128_generations_instruct_flat_fast_dllm.json"),
        RunSpec("countdown", "prompt_diag", "Qwen", "AR", "instruct_templated", "results/Qwen_Qwen2.5-7B-Instruct_countdown_cd4_32_8_0.7_8_512_128_generations_instruct_templated_vllm.json"),
        RunSpec("countdown", "prompt_diag", "Qwen", "AR", "instruct_flat", "results/Qwen_Qwen2.5-7B-Instruct_countdown_cd4_32_8_0.7_8_512_128_generations_instruct_flat_vllm.json"),
        RunSpec("countdown", "prompt_diag", "Llama", "AR", "instruct_templated", "results/meta-llama_Llama-3.1-8B-Instruct_countdown_cd4_32_8_0.7_8_512_128_generations_instruct_templated_vllm.json"),
        RunSpec("countdown", "prompt_diag", "Llama", "AR", "instruct_flat", "results/meta-llama_Llama-3.1-8B-Instruct_countdown_cd4_32_8_0.7_8_512_128_generations_instruct_flat_vllm.json"),
        # Trip planning main stochastic runs and available flat variants
        RunSpec("trip_planning", "base_main", "LLaDA", "dLLM", "base_native", "results/GSAI-ML_LLaDA-8B-Base_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json"),
        RunSpec("trip_planning", "base_main", "Dream", "dLLM", "base_native", "results/Dream-org_Dream-v0-Base-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json"),
        RunSpec("trip_planning", "base_main", "Qwen", "AR", "base_native", "results/Qwen_Qwen2.5-7B_trip_planning_256_1_0.7_2_200_64_generations_vllm.json"),
        RunSpec("trip_planning", "base_main", "Llama", "AR", "base_native", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B_trip_planning_256_1_0.7_2_200_64_generations_base_native_vllm.json"),
        RunSpec("trip_planning", "instruct_main", "LLaDA", "dLLM", "instruct_templated", "results/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json"),
        RunSpec("trip_planning", "instruct_main", "Dream", "dLLM", "instruct_templated", "results/Dream-org_Dream-v0-Instruct-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json"),
        RunSpec("trip_planning", "instruct_main", "Qwen", "AR", "instruct_templated", "results/Qwen_Qwen2.5-7B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_vllm.json"),
        RunSpec("trip_planning", "instruct_main", "Llama", "AR", "instruct_templated", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_templated_vllm.json"),
        RunSpec("trip_planning", "instruct_flat", "LLaDA", "dLLM", "instruct_flat", "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_16_1_0.7_2_200_64_generations_instruct_flat_fast_dllm.json"),
        RunSpec("trip_planning", "instruct_flat", "Llama", "AR", "instruct_flat", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_flat_vllm.json"),
    ]


def load_generation_file(path: str) -> dict[str, Any]:
    with open(ROOT / path, "r") as handle:
        return json.load(handle)


def compare_numeric(pred: Any, gold: Any) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except Exception:
        return str(pred).strip().lower() == str(gold).strip().lower()


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FRACTION_RE = re.compile(r"[-+]?\d+\s*/\s*\d+")
_ANSWER_IS_RE = re.compile(r"answer\s+(?:is|=)\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I)
_HASH_RE = re.compile(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def parse_numeric_token(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip().replace(",", "")
    if not raw:
        return None
    if "/" in raw and re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", raw):
        num, den = raw.split("/", 1)
        den_val = float(den)
        if abs(den_val) < 1e-12:
            return None
        return float(num) / den_val
    try:
        return float(raw)
    except Exception:
        return None


def canonical_gsm_answer(text: str) -> float | None:
    return parse_numeric_token(Parser.extract_answer_boxed(text))


def alternate_gsm_answer(text: str) -> float | None:
    boxed = canonical_gsm_answer(text)
    if boxed is not None:
        return boxed
    match = _HASH_RE.search(text)
    if match:
        val = parse_numeric_token(match.group(1))
        if val is not None:
            return val
    match = _ANSWER_IS_RE.search(text)
    if match:
        val = parse_numeric_token(match.group(1))
        if val is not None:
            return val
    frac_matches = _FRACTION_RE.findall(text)
    if frac_matches:
        val = parse_numeric_token(frac_matches[-1])
        if val is not None:
            return val
    num_matches = _NUMBER_RE.findall(text)
    if num_matches:
        return parse_numeric_token(num_matches[-1])
    return None


def stringify_number(value: float | None) -> str | None:
    if value is None:
        return None
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def clean_trip_text(response: str) -> str:
    return response.split("<|endoftext|>")[0].split("\nTASK")[0]


def trip_gold_lists(gt_answer: str) -> tuple[list[str], list[int]]:
    cities_str, durations_str = gt_answer.split("||")
    cities = [x for x in cities_str.split("**") if x]
    durations = [int(x) for x in durations_str.split("**") if x]
    return cities, durations


def normalized_trip_plan(parsed_plan: list[tuple[str, int]]) -> str | None:
    if not parsed_plan:
        return None
    return " | ".join(f"{city}:{days}" for city, days in parsed_plan)


def extract_countdown_equation_bits(prediction: str) -> list[tuple[str, str, str, str]]:
    normalized = _normalize_countdown_prediction(prediction)
    if not normalized:
        return []
    bits: list[tuple[str, str, str, str]] = []
    for chunk in normalized.split(","):
        piece = chunk.strip()
        if not piece:
            break
        match = _COUNTDOWN_BINARY_EQ_RE.fullmatch(piece)
        if not match:
            break
        bits.append(match.groups())
    return bits


def chain_to_expression(prediction: str) -> str | None:
    bits = extract_countdown_equation_bits(prediction)
    if not bits:
        return None
    expr_map: dict[str, str] = {}
    for left_a, op, left_b, right in bits:
        a_expr = expr_map.get(left_a, left_a)
        b_expr = expr_map.get(left_b, left_b)
        expr_map[right] = f"({a_expr}{op}{b_expr})"
    return expr_map.get(bits[-1][3])


class ExpressionValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[int] = []
        self.invalid_operator = False

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.invalid_operator = True
            return
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            self.invalid_operator = True
            return
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)):
            self.invalid_operator = True
            return
        if isinstance(node.value, float) and abs(node.value - round(node.value)) > 1e-9:
            self.invalid_operator = True
            return
        self.values.append(int(round(float(node.value))))

    def generic_visit(self, node: ast.AST) -> None:
        self.invalid_operator = True


def safe_eval_expr(expr: str) -> tuple[float | None, str | None, list[int]]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None, "invalid syntax", []
    validator = ExpressionValidator()
    validator.visit(tree)
    if validator.invalid_operator:
        return None, "invalid operator", validator.values
    try:
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return None, "division by zero", validator.values
    except Exception:
        return None, "invalid syntax", validator.values
    return float(value), None, validator.values


def count_mismatch(values: list[int], allowed: list[int]) -> bool:
    return Counter(values) != Counter(allowed)


def countdown_alt_expression(prediction: str) -> str | None:
    expr = chain_to_expression(prediction)
    if expr:
        return expr
    normalized = _normalize_countdown_prediction(prediction)
    if not normalized:
        return None
    first = normalized.split("=", 1)[0].strip()
    if re.fullmatch(r"[\d+\-*/().\s]+", first):
        return first
    return None


def analyze_gsm_sample(raw: str, gold: float | None, canonical_from_file: Any = None) -> dict[str, Any]:
    canonical = parse_numeric_token(canonical_from_file) if canonical_from_file is not None else canonical_gsm_answer(raw)
    alternate = alternate_gsm_answer(raw)
    canonical_correct = gold is not None and canonical is not None and compare_numeric(canonical, gold)
    alternate_correct = gold is not None and alternate is not None and compare_numeric(alternate, gold)
    canonical_missing = canonical is None
    format_failure = canonical_missing and alternate is not None
    parse_failure = alternate is None
    validity_failure = canonical is not None and not canonical_correct
    category = "correct"
    if canonical_missing and alternate_correct:
        category = "boxed-only miss"
    elif parse_failure:
        category = "parse failure"
    elif format_failure:
        category = "format failure"
    elif validity_failure:
        category = "wrong numeric answer"
    return {
        "canonical_parsed": canonical is not None,
        "canonical_correct": canonical_correct,
        "alternate_correct": alternate_correct,
        "parse_failure": parse_failure,
        "format_failure": format_failure,
        "validity_failure": validity_failure,
        "canonical_answer": canonical,
        "alternate_answer": alternate,
        "normalized_answer": stringify_number(alternate),
        "valid_repr": stringify_number(alternate) if alternate is not None else None,
        "category": category,
    }


def analyze_countdown_sample(raw: str, gt_answer: str, canonical_from_file: Any = None) -> dict[str, Any]:
    operands = [int(float(x)) for x in gt_answer.split(",")[:-1]]
    target = int(float(gt_answer.split(",")[-1]))
    normalized = _normalize_countdown_prediction(raw)
    canonical_correct = canonical_from_file is not None
    canonical_chain = _extract_leading_subequations(raw) if not canonical_correct else normalized
    expr = countdown_alt_expression(raw)
    alt_value = None
    alt_error = None
    values: list[int] = []
    wrong_number_use = False
    if expr is not None:
        alt_value, alt_error, values = safe_eval_expr(expr)
        wrong_number_use = bool(values) and count_mismatch(values, operands)
    parse_failure = False
    format_failure = False
    validity_failure = False
    category = "correct"

    if not normalized:
        parse_failure = True
        category = "parse failure"
    elif not canonical_chain:
        if expr is None:
            parse_failure = True
            category = "parse failure"
        elif alt_error == "invalid operator":
            format_failure = True
            category = "invalid operator"
        elif alt_error == "invalid syntax":
            format_failure = True
            category = "invalid syntax"
        elif alt_error == "division by zero":
            validity_failure = True
            category = "division by zero"
        elif wrong_number_use:
            validity_failure = True
            category = "wrong number use"
        elif alt_value is not None and not compare_numeric(alt_value, target):
            validity_failure = True
            category = "wrong target"
        elif alt_value is not None and compare_numeric(alt_value, target):
            format_failure = True
            category = "malformed equation chain"
        else:
            parse_failure = True
            category = "parse failure"
    else:
        if canonical_chain != normalized:
            format_failure = True
            category = "malformed equation chain"
        if not canonical_correct and category == "correct":
            expr = expr or chain_to_expression(raw)
            if expr is None:
                format_failure = True
                category = "malformed equation chain"
            else:
                alt_value, alt_error, values = safe_eval_expr(expr)
                wrong_number_use = bool(values) and count_mismatch(values, operands)
                if alt_error == "division by zero":
                    validity_failure = True
                    category = "division by zero"
                elif alt_error == "invalid syntax":
                    format_failure = True
                    category = "invalid syntax"
                elif alt_error == "invalid operator":
                    format_failure = True
                    category = "invalid operator"
                elif wrong_number_use:
                    validity_failure = True
                    category = "wrong number use"
                else:
                    validity_failure = True
                    category = "wrong target"

    alt_correct = False
    if expr is not None and alt_error is None and not wrong_number_use and alt_value is not None:
        alt_correct = compare_numeric(alt_value, target)

    normalized_answer = None
    if canonical_chain:
        normalized_answer = canonical_chain.split("=")[-1].strip()
    elif alt_value is not None:
        normalized_answer = stringify_number(alt_value)

    return {
        "canonical_parsed": bool(canonical_chain),
        "canonical_correct": canonical_correct,
        "alternate_correct": alt_correct,
        "parse_failure": parse_failure,
        "format_failure": format_failure,
        "validity_failure": validity_failure,
        "normalized_answer": normalized_answer,
        "valid_repr": normalized_answer if (canonical_chain or expr is not None) else None,
        "category": category,
        "alternate_expression": expr,
    }


def compare_trip_plan(parsed: list[tuple[str, int]], gold_cities: list[str], gold_durations: list[int]) -> str:
    if not parsed:
        return "empty parsed plan"
    if len(parsed) < len(gold_cities):
        return "exact-match failure"
    for idx, (city, stay) in enumerate(parsed[: len(gold_cities)]):
        if city != gold_cities[idx]:
            return "city sequence mismatch"
        if stay != gold_durations[idx]:
            return "duration mismatch"
    return "correct"


def analyze_trip_sample(raw: str, trip_spec: str, canonical_from_file: Any = None) -> dict[str, Any]:
    cleaned = clean_trip_text(raw)
    gold_cities, gold_durations = trip_gold_lists(trip_spec)
    parsed = parse_trip_response(cleaned)
    total_days = re.search(r"European cities for (\d+) days", cleaned)
    visit_ranges = re.findall(r"\d+-\d+", cleaned)
    flight_lines = re.findall(r".*Day (\d+).*from (\w+) to (\w+)", cleaned)
    canonical_correct = canonical_from_file is not None
    parse_failure = False
    format_failure = False
    validity_failure = False
    category = "correct"

    if not total_days:
        parse_failure = True
        category = "missing total-day line"
    elif not visit_ranges:
        parse_failure = True
        category = "missing or malformed day ranges"
    elif not flight_lines:
        parse_failure = True
        category = "missing or malformed flight lines"
    elif not parsed:
        parse_failure = True
        category = "empty parsed plan"
    else:
        category = compare_trip_plan(parsed, gold_cities, gold_durations)
        if category != "correct":
            validity_failure = True
    return {
        "canonical_parsed": bool(parsed),
        "canonical_correct": canonical_correct,
        "alternate_correct": canonical_correct,
        "parse_failure": parse_failure,
        "format_failure": format_failure,
        "validity_failure": validity_failure,
        "normalized_answer": normalized_trip_plan(parsed),
        "valid_repr": normalized_trip_plan(parsed),
        "category": category,
    }


def sample_analysis(
    benchmark: str,
    raw: str,
    gold: Any,
    question: str | None = None,
    canonical_from_file: Any = None,
) -> dict[str, Any]:
    if benchmark == "gsm8k":
        return analyze_gsm_sample(raw, float(gold) if gold is not None else None, canonical_from_file)
    if benchmark == "countdown":
        return analyze_countdown_sample(raw, str(gold), canonical_from_file)
    if benchmark == "trip_planning":
        return analyze_trip_sample(raw, question or str(gold), canonical_from_file)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def iterate_samples(data: dict[str, Any]):
    for question_index, item in enumerate(data["generations"]):
        raws = item.get("raw_generations", item.get("generations", []))
        extracted = item.get("extracted_answer", [None] * len(raws))
        for sample_index, raw in enumerate(raws):
            existing = extracted[sample_index] if sample_index < len(extracted) else None
            yield question_index, sample_index, item["question"], item["ground_truth"], raw, existing


def summarize_run(spec: RunSpec, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    question_count = len(data["generations"])
    sample_count = len(data["generations"][0].get("raw_generations", data["generations"][0].get("generations", [])))
    stats = Counter()
    categories = Counter()
    gsm_delta = Counter()
    first_examples: dict[str, dict[str, Any]] = {}

    for q_idx, s_idx, question, gold, raw, existing in iterate_samples(data):
        result = sample_analysis(spec.benchmark, raw, gold, question, existing)
        stats["total_samples"] += 1
        stats["parsed_samples"] += int(result["canonical_parsed"])
        stats["parse_failures"] += int(result["parse_failure"])
        stats["format_failures"] += int(result["format_failure"])
        stats["validity_failures"] += int(result["validity_failure"])
        stats["correct_samples"] += int(result["canonical_correct"])
        stats["alternate_correct_samples"] += int(result["alternate_correct"])
        categories[result["category"]] += 1
        if result["category"] not in first_examples:
            first_examples[result["category"]] = {
                "question_index": q_idx,
                "sample_index": s_idx,
                "question": question,
                "raw": raw[:1200],
                "gold": gold,
            }
        if spec.benchmark == "gsm8k":
            gsm_delta["canonical_correct"] += int(result["canonical_correct"])
            gsm_delta["alternate_correct"] += int(result["alternate_correct"])
            gsm_delta["canonical_false_alternate_true"] += int((not result["canonical_correct"]) and result["alternate_correct"])
            gsm_delta["alternate_false_canonical_true"] += int(result["canonical_correct"] and (not result["alternate_correct"]))
        else:
            gsm_delta["canonical_false_alternate_true"] += int((not result["canonical_correct"]) and result["alternate_correct"])
            gsm_delta["alternate_false_canonical_true"] += int(result["canonical_correct"] and (not result["alternate_correct"]))

    total = stats["total_samples"] or 1
    parsed = stats["parsed_samples"] or 1
    summary = {
        "benchmark": spec.benchmark,
        "condition": spec.condition,
        "model": spec.model,
        "paradigm": spec.paradigm,
        "prompt_mode": spec.prompt_mode,
        "file": spec.path,
        "questions": question_count,
        "samples_per_question": sample_count,
        "total_samples": stats["total_samples"],
        "parsed_samples": stats["parsed_samples"],
        "parse_failures": stats["parse_failures"],
        "format_failures": stats["format_failures"],
        "validity_failures": stats["validity_failures"],
        "correct_samples": stats["correct_samples"],
        "alternate_correct_samples": stats["alternate_correct_samples"],
        "parse_failure_rate": stats["parse_failures"] / total,
        "correct_rate_parsed": stats["correct_samples"] / parsed,
        "correct_rate_all": stats["correct_samples"] / total,
    }
    detailed = {
        **summary,
        **gsm_delta,
        **{f"category::{k}": v for k, v in sorted(categories.items())},
        "examples": first_examples,
    }
    return summary, detailed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs() -> dict[str, list[dict[str, Any]]]:
    ensure_dirs()
    summaries: list[dict[str, Any]] = []
    gsm_rows: list[dict[str, Any]] = []
    countdown_rows: list[dict[str, Any]] = []
    trip_rows: list[dict[str, Any]] = []
    examples_by_benchmark: dict[str, dict[str, Any]] = defaultdict(dict)

    for spec in run_specs():
        data = load_generation_file(spec.path)
        summary, detailed = summarize_run(spec, data)
        summaries.append(summary)
        for category, example in detailed.pop("examples").items():
            examples_by_benchmark[spec.benchmark].setdefault(f"{spec.condition}:{spec.model}:{category}", example)
        if spec.benchmark == "gsm8k":
            gsm_rows.append(detailed)
        elif spec.benchmark == "countdown":
            countdown_rows.append(detailed)
        elif spec.benchmark == "trip_planning":
            trip_rows.append(detailed)

    write_csv(TABLES / "parser_failure_summary.csv", summaries)
    write_csv(TABLES / "parser_sensitivity_gsm8k.csv", gsm_rows)
    write_csv(TABLES / "parser_sensitivity_countdown.csv", countdown_rows)
    write_csv(TABLES / "parser_sensitivity_trip.csv", trip_rows)

    with open(TABLES / "parser_examples.json", "w") as handle:
        json.dump(examples_by_benchmark, handle, indent=2)

    return {
        "summary": summaries,
        "gsm": gsm_rows,
        "countdown": countdown_rows,
        "trip": trip_rows,
    }


def main() -> None:
    outputs = build_outputs()
    print(f"Wrote {len(outputs['summary'])} rows to {TABLES / 'parser_failure_summary.csv'}")
    print(f"Wrote {len(outputs['gsm'])} GSM8K sensitivity rows")
    print(f"Wrote {len(outputs['countdown'])} Countdown sensitivity rows")
    print(f"Wrote {len(outputs['trip'])} Trip Planning sensitivity rows")


if __name__ == "__main__":
    main()
