#!/usr/bin/env python3
"""Generate main-result summary, parser, and paired-bootstrap tables."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.append(str(Path(__file__).resolve().parent))
import analyze_parser_failures as apf  # noqa: E402


ROOT = apf.ROOT
TABLES = apf.TABLES
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260523


@dataclass(frozen=True)
class MainRun:
    benchmark: str
    condition: str
    model: str
    prompt_mode: str
    path: str
    ks: tuple[int, ...]


K_128 = (1, 16, 128)
K_64 = (1, 16, 64)


RUNS: tuple[MainRun, ...] = (
    MainRun("gsm8k", "GSM8K base 4-shot", "LLaDA-Base", "base", "results/milestone2_gsm8k_base/GSAI-ML_LLaDA-8B-Base_256_256_32_1_0.7_4_128_128_generations_base_native_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K base 4-shot", "Dream-Base", "base", "results/milestone2_gsm8k_base/Dream-org_Dream-v0-Base-7B_256_256_32_1_0.7_4_128_128_generations_base_native_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K base 4-shot", "Qwen-Base", "base", "results/milestone2_gsm8k_base/Qwen_Qwen2.5-7B_256_1_0.7_4_128_128_generations_base_native_vllm.json", K_128),
    MainRun("gsm8k", "GSM8K base 4-shot", "Llama-Base", "base", "results/milestone2_gsm8k_base/meta-llama_Llama-3.1-8B_256_1_0.7_4_128_128_generations_base_native_vllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct templated", "LLaDA-Instruct", "templated", "results/milestone2_gsm8k_instruct/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_4_128_128_generations_instruct_templated_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct templated", "Dream-Instruct", "templated", "results/milestone2_gsm8k_instruct/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_4_128_128_generations_instruct_templated_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct templated", "Qwen-Instruct", "templated", "results/milestone2_gsm8k_instruct/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_4_128_128_generations_instruct_templated_vllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct templated", "Llama-Instruct", "templated", "results/milestone2_gsm8k_instruct/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_4_128_128_generations_instruct_templated_vllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct flat", "LLaDA-Instruct", "flat", "results/milestone2_gsm8k_instruct/GSAI-ML_LLaDA-8B-Instruct_256_256_8_1_0.7_4_128_128_generations_instruct_flat_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct flat", "Dream-Instruct", "flat", "results/milestone2_gsm8k_instruct/Dream-org_Dream-v0-Instruct-7B_256_256_32_1_0.7_4_128_128_generations_instruct_flat_fast_dllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct flat", "Qwen-Instruct", "flat", "results/milestone2_gsm8k_instruct/Qwen_Qwen2.5-7B-Instruct_256_1_0.7_4_128_128_generations_instruct_flat_vllm.json", K_128),
    MainRun("gsm8k", "GSM8K instruct flat", "Llama-Instruct", "flat", "results/milestone2_gsm8k_instruct/meta-llama_Llama-3.1-8B-Instruct_256_1_0.7_4_128_128_generations_instruct_flat_vllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 base", "LLaDA-Base", "base", "results/milestone2_countdown_base_refresh/GSAI-ML_LLaDA-8B-Base_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 base", "Dream-Base", "base", "results/milestone2_countdown_base_refresh/Dream-org_Dream-v0-Base-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 base", "Qwen-Base", "base", "results/milestone2_countdown_base_refresh/Qwen_Qwen2.5-7B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 base", "Llama-Base", "base", "results/milestone2_countdown_base_refresh/meta-llama_Llama-3.1-8B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 instruct templated", "LLaDA-Instruct", "templated", "results/milestone2_countdown_instruct_refresh/GSAI-ML_LLaDA-8B-Instruct_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 instruct templated", "Dream-Instruct", "templated", "results/milestone2_countdown_instruct_refresh/Dream-org_Dream-v0-Instruct-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 instruct templated", "Qwen-Instruct", "templated", "results/milestone2_countdown_instruct_refresh/Qwen_Qwen2.5-7B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json", K_128),
    MainRun("countdown", "Countdown-cd4 instruct templated", "Llama-Instruct", "templated", "results/milestone2_countdown_instruct_refresh/meta-llama_Llama-3.1-8B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json", K_128),
    MainRun("trip_planning", "Trip Planning base", "LLaDA-Base", "base", "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Base_trip_planning_256_256_32_1_0.7_2_200_64_generations_base_native_fast_dllm.json", K_64),
    MainRun("trip_planning", "Trip Planning base", "Dream-Base", "base", "results/Dream-org_Dream-v0-Base-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json", K_64),
    MainRun("trip_planning", "Trip Planning base", "Qwen-Base", "base", "results/Qwen_Qwen2.5-7B_trip_planning_256_1_0.7_2_200_64_generations_vllm.json", K_64),
    MainRun("trip_planning", "Trip Planning base", "Llama-Base", "base", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B_trip_planning_256_1_0.7_2_200_64_generations_base_native_vllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct templated", "LLaDA-Instruct", "templated", "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_16_1_0.7_2_200_64_generations_instruct_templated_fast_dllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct templated", "Dream-Instruct", "templated", "results/Dream-org_Dream-v0-Instruct-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct templated", "Qwen-Instruct", "templated", "results/Qwen_Qwen2.5-7B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_vllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct templated", "Llama-Instruct", "templated", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_templated_vllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct flat", "LLaDA-Instruct", "flat", "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_16_1_0.7_2_200_64_generations_instruct_flat_fast_dllm.json", K_64),
    MainRun("trip_planning", "Trip Planning instruct flat", "Llama-Instruct", "flat", "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_flat_vllm.json", K_64),
)


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0 or n < k:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def is_correct(answer: Any, gold: Any) -> bool:
    if answer is None:
        return False
    return apf.compare_numeric(answer, gold)


def primary_correct_vector(item: dict[str, Any]) -> list[bool]:
    answers = item.get("extracted_answer", [])
    if not isinstance(answers, list):
        answers = [answers]
    return [is_correct(answer, item["ground_truth"]) for answer in answers]


def parser_stats(run: MainRun, data: dict[str, Any]) -> dict[str, float]:
    total = parse_success = parse_fail = correct = 0
    for item in data["generations"]:
        raws = item.get("raw_generations", item.get("generations", []))
        extracted = item.get("extracted_answer", [None] * len(raws))
        for idx, raw in enumerate(raws):
            existing = extracted[idx] if idx < len(extracted) else None
            result = apf.sample_analysis(
                run.benchmark,
                raw,
                item["ground_truth"],
                question=item.get("question"),
                canonical_from_file=existing,
            )
            total += 1
            parse_success += int(not result["parse_failure"])
            parse_fail += int(result["parse_failure"])
            correct += int(result["canonical_correct"])
    denom = total or 1
    return {
        "parser_failure_rate": parse_fail / denom,
        "parseable_rate": parse_success / denom,
        "correct_sample_rate": correct / denom,
        "correct_given_parseable": correct / (parse_success or 1),
    }


def per_question_scores(run: MainRun, data: dict[str, Any], ks: Iterable[int]) -> dict[int, list[float]]:
    by_k = {k: [] for k in ks}
    for item in data["generations"]:
        correct = primary_correct_vector(item)
        n = len(correct)
        c = sum(correct)
        for k in ks:
            by_k[k].append(pass_at_k(n, c, k))
    return by_k


def per_question_scores_from_raw(run: MainRun, data: dict[str, Any], ks: Iterable[int], alternate: bool = False) -> dict[int, list[float]]:
    by_k = {k: [] for k in ks}
    for item in data["generations"]:
        raws = item.get("raw_generations", item.get("generations", []))
        correct_count = 0
        for raw in raws:
            parsed = apf.sample_analysis(run.benchmark, raw, item["ground_truth"], question=item.get("question"))
            key = "alternate_correct" if alternate else "canonical_correct"
            correct_count += int(parsed[key])
        for k in ks:
            by_k[k].append(pass_at_k(len(raws), correct_count, k))
    return by_k


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    boots = []
    for _ in range(BOOTSTRAPS):
        boots.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return boots[int(0.025 * (BOOTSTRAPS - 1))], boots[int(0.975 * (BOOTSTRAPS - 1))]


def paired_bootstrap(diff_values: list[float], rng: random.Random) -> tuple[float, float, float, float]:
    observed = mean(diff_values)
    n = len(diff_values)
    boots = []
    for _ in range(BOOTSTRAPS):
        boots.append(sum(diff_values[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int(0.025 * (BOOTSTRAPS - 1))]
    hi = boots[int(0.975 * (BOOTSTRAPS - 1))]
    le_zero = sum(1 for x in boots if x <= 0) / len(boots)
    ge_zero = sum(1 for x in boots if x >= 0) / len(boots)
    p_value = min(1.0, 2 * min(le_zero, ge_zero))
    return observed, lo, hi, p_value


def fmt(x: float) -> str:
    return f"{x:.3f}"


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{fmt(point)} [{fmt(lo)}, {fmt(hi)}]"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fmt_p(x: float) -> str:
    if x < 0.001:
        return "$<0.001$"
    return f"{x:.3f}"


def esc(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def row_dicts() -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows = []
    for run in RUNS:
        data = load_json(run.path)
        scores = per_question_scores(run, data, run.ks)
        stats = parser_stats(run, data)
        row = {
            "benchmark_condition": run.condition,
            "model": run.model,
            "prompt_mode": run.prompt_mode,
            "questions": len(data["generations"]),
            "samples": len(data["generations"][0].get("raw_generations", data["generations"][0].get("generations", []))),
            "pass_max_k": max(run.ks),
            "parser_failure_rate": stats["parser_failure_rate"],
            "parseable_rate": stats["parseable_rate"],
            "correct_sample_rate": stats["correct_sample_rate"],
            "correct_given_parseable": stats["correct_given_parseable"],
            "path": run.path,
        }
        for k in run.ks:
            values = scores[k]
            lo, hi = bootstrap_ci(values, rng)
            row[f"pass@{k}"] = mean(values)
            row[f"pass@{k}_ci_low"] = lo
            row[f"pass@{k}_ci_high"] = hi
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_main_table(rows: list[dict[str, Any]]) -> None:
    def table_for(label: str, caption: str, subset: list[dict[str, Any]]) -> str:
        lines = [
            "\\begin{landscape}",
            "\\begin{table}[p]",
            "\\centering",
            f"\\caption{{{caption} Pass@1 is the estimated stochastic pass@1 from the $n$ saved samples, not greedy $n=1$ accuracy. Brackets report 95\\% bootstrap confidence intervals over questions using 1,000 resamples with fixed seed 20260523. Parser failure is the sample-level primary-parser-entry failure rate.}}",
            f"\\label{{{label}}}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{lllrrcccc}",
            "\\toprule",
            "\\textbf{Benchmark / condition} & \\textbf{Model} & \\textbf{Prompt} & \\textbf{Q} & \\textbf{$n$} & \\textbf{est. pass@1 [CI]} & \\textbf{pass@16 [CI]} & \\textbf{pass@max [CI]} & \\textbf{Parser fail. \\%} \\\\",
            "\\midrule",
        ]
        for row in subset:
            max_k = row["pass_max_k"]
            pass1 = fmt_ci(row["pass@1"], row["pass@1_ci_low"], row["pass@1_ci_high"])
            pass16 = fmt_ci(row["pass@16"], row["pass@16_ci_low"], row["pass@16_ci_high"])
            passmax = fmt_ci(row[f"pass@{max_k}"], row[f"pass@{max_k}_ci_low"], row[f"pass@{max_k}_ci_high"])
            lines.append(
                f"{esc(row['benchmark_condition'])} & {esc(row['model'])} & {esc(row['prompt_mode'])} & "
                f"{row['questions']} & {row['samples']} & {pass1} & {pass16} & {passmax} ($k={max_k}$) & {fmt_pct(row['parser_failure_rate'])} \\\\"
            )
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "\\end{landscape}",
            "",
        ])
        return "\n".join(lines)

    groups = [
        (
            "tab:main-results-gsm8k",
            "Main stochastic results for GSM8K.",
            [row for row in rows if row["benchmark_condition"].startswith("GSM8K")],
        ),
        (
            "tab:main-results-countdown",
            "Main stochastic results for Countdown-cd4.",
            [row for row in rows if row["benchmark_condition"].startswith("Countdown")],
        ),
        (
            "tab:main-results-trip",
            "Main stochastic results for Trip Planning.",
            [row for row in rows if row["benchmark_condition"].startswith("Trip Planning")],
        ),
    ]
    content = [
        "% Generated by thesis/scripts/build_main_results_tables.py.",
        *[table_for(label, caption, subset) for label, caption, subset in groups],
    ]
    (TABLES / "main_results_summary.tex").write_text("\n".join(content))


def find_run(condition: str, model: str) -> MainRun:
    for run in RUNS:
        if run.condition == condition and run.model == model:
            return run
    raise KeyError((condition, model))


@dataclass(frozen=True)
class PairSpec:
    label: str
    condition: str
    left_model: str
    right_model: str
    k: int
    direction: str


PAIR_SPECS: tuple[PairSpec, ...] = (
    PairSpec("Countdown base low-$k$ reversal", "Countdown-cd4 base", "LLaDA-Base", "Qwen-Base", 1, "LLaDA $-$ Qwen"),
    PairSpec("Countdown base high-$k$ reversal", "Countdown-cd4 base", "Qwen-Base", "LLaDA-Base", 128, "Qwen $-$ LLaDA"),
    PairSpec("Countdown instruct low-$k$", "Countdown-cd4 instruct templated", "Dream-Instruct", "Llama-Instruct", 1, "Dream $-$ Llama"),
    PairSpec("Countdown instruct high-$k$", "Countdown-cd4 instruct templated", "Llama-Instruct", "Dream-Instruct", 128, "Llama $-$ Dream"),
    PairSpec("Trip base low-$k$", "Trip Planning base", "Dream-Base", "Llama-Base", 1, "Dream $-$ Llama"),
    PairSpec("Trip base high-$k$", "Trip Planning base", "Llama-Base", "Dream-Base", 64, "Llama $-$ Dream"),
    PairSpec("Trip instruct low-$k$", "Trip Planning instruct templated", "Dream-Instruct", "Llama-Instruct", 1, "Dream $-$ Llama"),
    PairSpec("Trip instruct flat high-$k$", "Trip Planning instruct flat", "Llama-Instruct", "LLaDA-Instruct", 64, "Llama flat $-$ LLaDA flat"),
)


def pair_rows() -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows = []
    for pair in PAIR_SPECS:
        left = find_run(pair.condition, pair.left_model)
        right = find_run(pair.condition, pair.right_model)
        left_data = load_json(left.path)
        right_data = load_json(right.path)
        left_questions = [item["question"] for item in left_data["generations"]]
        right_questions = [item["question"] for item in right_data["generations"]]
        if left_questions != right_questions:
            raise ValueError(f"Question order mismatch for {pair.label}")
        left_scores = per_question_scores(left, left_data, [pair.k])[pair.k]
        right_scores = per_question_scores(right, right_data, [pair.k])[pair.k]
        diff = [a - b for a, b in zip(left_scores, right_scores)]
        observed, lo, hi, p_value = paired_bootstrap(diff, rng)
        rows.append({
            "comparison": pair.label,
            "direction": pair.direction,
            "k": pair.k,
            "difference": observed,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p_value,
            "questions": len(diff),
        })
    return rows


def write_pair_table(rows: list[dict[str, Any]]) -> None:
    lines = [
        "% Generated by thesis/scripts/build_main_results_tables.py.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Paired bootstrap comparisons for the main Countdown and Trip Planning claims. Differences are in pass@k units and are computed over matched questions; positive values favor the first model named in the direction column. The $p$ column is a two-sided bootstrap sign probability, included as a descriptive test rather than a repeated-seed estimate.}",
        "\\label{tab:paired-bootstrap-tests}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrcc}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{Direction} & \\textbf{$k$} & \\textbf{Diff.} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['comparison'])} & {row['direction']} & {row['k']} & {fmt(row['difference'])} & "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] & {fmt_p(row['p_value'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    (TABLES / "main_paired_bootstrap_tests.tex").write_text("\n".join(lines))


def write_gsm_parser_table() -> None:
    wanted = [run for run in RUNS if run.benchmark == "gsm8k" and run.condition in {"GSM8K base 4-shot", "GSM8K instruct templated", "GSM8K instruct flat"}]
    lines = [
        "% Generated by thesis/scripts/build_main_results_tables.py.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{GSM8K strict boxed-answer scoring versus permissive final-number scoring. Values are recomputed from the same saved samples. The permissive scorer is used only for parser sensitivity; because it applies a different extraction rule, it can raise or lower measured scores.}",
        "\\label{tab:gsm8k-strict-vs-permissive}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{Model} & \\textbf{Strict est. pass@1} & \\textbf{Permissive est. pass@1} & \\textbf{Strict pass@max} & \\textbf{Permissive pass@max} \\\\",
        "\\midrule",
    ]
    for run in wanted:
        data = load_json(run.path)
        strict = per_question_scores(run, data, [1, max(run.ks)])
        permissive = per_question_scores_from_raw(run, data, [1, max(run.ks)], alternate=True)
        lines.append(
            f"{esc(run.condition)} & {esc(run.model)} & {fmt(mean(strict[1]))} & {fmt(mean(permissive[1]))} & "
            f"{fmt(mean(strict[max(run.ks)]))} & {fmt(mean(permissive[max(run.ks)]))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    (TABLES / "gsm8k_strict_vs_permissive.tex").write_text("\n".join(lines))


def write_trip_parser_table(rows: list[dict[str, Any]]) -> None:
    trip_rows = [row for row in rows if row["benchmark_condition"].startswith("Trip Planning")]
    lines = [
        "% Generated by thesis/scripts/build_main_results_tables.py.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Trip Planning parseability and semantic correctness. Parseable is the share of samples that pass the same primary parser-entry checks used for the parser-failure rate in the main results table and Figure~\\ref{fig:parser-failure-rates}. Correct/all is the share of all samples that are semantically correct after parsing; correct/parsed conditions that rate on parser-entry-successful samples only.}",
        "\\label{tab:trip-parseability-semantic}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{Model} & \\textbf{Prompt} & \\textbf{Parseable \\%} & \\textbf{Correct/all \\%} & \\textbf{Correct/parsed \\%} \\\\",
        "\\midrule",
    ]
    for row in trip_rows:
        lines.append(
            f"{esc(row['benchmark_condition'])} & {esc(row['model'])} & {esc(row['prompt_mode'])} & "
            f"{fmt_pct(row['parseable_rate'])} & {fmt_pct(row['correct_sample_rate'])} & {fmt_pct(row['correct_given_parseable'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    (TABLES / "trip_parseability_semantic.tex").write_text("\n".join(lines))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = row_dicts()
    pairs = pair_rows()
    write_csv(TABLES / "main_results_summary.csv", rows)
    write_csv(TABLES / "main_paired_bootstrap_tests.csv", pairs)
    write_main_table(rows)
    write_pair_table(pairs)
    write_gsm_parser_table()
    write_trip_parser_table(rows)
    print(f"Wrote {len(rows)} main-result rows and {len(pairs)} paired tests")


if __name__ == "__main__":
    main()
