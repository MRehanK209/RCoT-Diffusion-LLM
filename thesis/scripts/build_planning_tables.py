#!/usr/bin/env python3
"""Generate appendix pass@k and bootstrap-CI tables from stored artifacts."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "thesis" / "tables"
OUT = TABLES / "planning_passk_tables.tex"

K_COUNTDOWN = [1, 2, 4, 8, 16, 32, 64, 128]
K_TRIP = [1, 2, 4, 8, 16, 32, 64]
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260523


@dataclass(frozen=True)
class RunSpec:
    condition: str
    caption: str
    label: str
    model: str
    path: str
    ks: tuple[int, ...]


RUNS = [
    RunSpec(
        "countdown_base",
        "Countdown-cd4 base main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-base-main-passk",
        "LLaDA-Base",
        "results/milestone2_countdown_base_refresh/GSAI-ML_LLaDA-8B-Base_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_base",
        "Countdown-cd4 base main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-base-main-passk",
        "Dream-Base",
        "results/milestone2_countdown_base_refresh/Dream-org_Dream-v0-Base-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_base_native_fast_dllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_base",
        "Countdown-cd4 base main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-base-main-passk",
        "Qwen-Base",
        "results/milestone2_countdown_base_refresh/Qwen_Qwen2.5-7B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_base",
        "Countdown-cd4 base main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-base-main-passk",
        "Llama-Base",
        "results/milestone2_countdown_base_refresh/meta-llama_Llama-3.1-8B_countdown_cd4_32_8_0.7_8_992_128_generations_base_native_vllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_instruct",
        "Countdown-cd4 instruct-templated main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-instruct-main-passk",
        "LLaDA-Instruct",
        "results/milestone2_countdown_instruct_refresh/GSAI-ML_LLaDA-8B-Instruct_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_instruct",
        "Countdown-cd4 instruct-templated main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-instruct-main-passk",
        "Dream-Instruct",
        "results/milestone2_countdown_instruct_refresh/Dream-org_Dream-v0-Instruct-7B_countdown_cd4_32_32_32_8_0.7_8_992_128_generations_instruct_templated_fast_dllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_instruct",
        "Countdown-cd4 instruct-templated main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-instruct-main-passk",
        "Qwen-Instruct",
        "results/milestone2_countdown_instruct_refresh/Qwen_Qwen2.5-7B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "countdown_instruct",
        "Countdown-cd4 instruct-templated main run pass@k, $n=128$, temperature $=0.7$.",
        "tab:countdown-instruct-main-passk",
        "Llama-Instruct",
        "results/milestone2_countdown_instruct_refresh/meta-llama_Llama-3.1-8B-Instruct_countdown_cd4_32_8_0.7_8_992_128_generations_instruct_templated_vllm.json",
        tuple(K_COUNTDOWN),
    ),
    RunSpec(
        "trip_base",
        "Trip Planning base stochastic run pass@k, $n=64$, temperature $=0.7$.",
        "tab:trip-base-main-passk",
        "LLaDA-Base",
        "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Base_trip_planning_256_256_32_1_0.7_2_200_64_generations_base_native_fast_dllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_base",
        "Trip Planning base stochastic run pass@k, $n=64$, temperature $=0.7$.",
        "tab:trip-base-main-passk",
        "Dream-Base",
        "results/Dream-org_Dream-v0-Base-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_base",
        "Trip Planning base stochastic run pass@k, $n=64$, temperature $=0.7$.",
        "tab:trip-base-main-passk",
        "Qwen-Base",
        "results/Qwen_Qwen2.5-7B_trip_planning_256_1_0.7_2_200_64_generations_vllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_base",
        "Trip Planning base stochastic run pass@k, $n=64$, temperature $=0.7$.",
        "tab:trip-base-main-passk",
        "Llama-Base",
        "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B_trip_planning_256_1_0.7_2_200_64_generations_base_native_vllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "LLaDA-Instruct",
        "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_16_1_0.7_2_200_64_generations_instruct_templated_fast_dllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "LLaDA-Instruct flat",
        "results/milestone2_trip_planning_llada_passk/GSAI-ML_LLaDA-8B-Instruct_trip_planning_256_256_16_1_0.7_2_200_64_generations_instruct_flat_fast_dllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "Dream-Instruct",
        "results/Dream-org_Dream-v0-Instruct-7B_trip_planning_256_256_32_1_0.7_2_200_64_generations_fast_dllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "Qwen-Instruct",
        "results/Qwen_Qwen2.5-7B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_vllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "Llama-Instruct",
        "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_templated_vllm.json",
        tuple(K_TRIP),
    ),
    RunSpec(
        "trip_instruct",
        "Trip Planning instruct stochastic run pass@k, $n=64$, temperature $=0.7$. Flat rows use flat completion-style prompts rather than chat templates.",
        "tab:trip-instruct-main-passk",
        "Llama-Instruct flat",
        "results/milestone2_trip_planning_llama_passk/meta-llama_Llama-3.1-8B-Instruct_trip_planning_256_1_0.7_2_200_64_generations_instruct_flat_vllm.json",
        tuple(K_TRIP),
    ),
]


def pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0 or n < k:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def is_correct(answer: object, gold: object) -> bool:
    if answer is None:
        return False
    try:
        return abs(float(answer) - float(gold)) < 1e-4
    except Exception:
        return str(answer).strip().lower() == str(gold).strip().lower()


def per_question_scores(path: Path, ks: Iterable[int]) -> dict[int, list[float]]:
    payload = json.loads(path.read_text())
    by_k = {k: [] for k in ks}
    for item in payload["generations"]:
        answers = item["extracted_answer"]
        if not isinstance(answers, list):
            answers = [answers]
        c = sum(1 for answer in answers if is_correct(answer, item["ground_truth"]))
        n = len(answers)
        for k in ks:
            by_k[k].append(pass_at_k(n, c, k))
    return by_k


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    boot = []
    for _ in range(BOOTSTRAPS):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boot.append(total / n)
    boot.sort()
    return boot[int(0.025 * (BOOTSTRAPS - 1))], boot[int(0.975 * (BOOTSTRAPS - 1))]


def fmt(x: float) -> str:
    return f"{x:.3f}"


def table_for_condition(condition: str, runs: list[RunSpec], with_ci: bool = False) -> str:
    ks = list(runs[0].ks)
    caption = runs[0].caption
    ci_note = " Cells report point estimate [95\\% bootstrap CI over questions]." if with_ci else ""
    label = runs[0].label + ("-ci" if with_ci else "")
    rng = random.Random(BOOTSTRAP_SEED)
    colspec = "l" + "c" * len(ks)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}{ci_note}}}",
        f"\\label{{{label}}}",
        "\\tiny" if with_ci else "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        "\\textbf{Model} & " + " & ".join(f"\\textbf{{{k}}}" for k in ks) + r" \\",
        "\\midrule",
    ]
    for spec in runs:
        scores = per_question_scores(ROOT / spec.path, ks)
        cells = []
        for k in ks:
            point = mean(scores[k])
            if with_ci:
                lo, hi = ci(scores[k], rng)
                cells.append(f"{fmt(point)} [{fmt(lo)}, {fmt(hi)}]")
            else:
                cells.append(fmt(point))
        lines.append(f"{spec.model} & " + " & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[RunSpec]] = {}
    for run in RUNS:
        grouped.setdefault(run.condition, []).append(run)

    ordered = ["countdown_base", "countdown_instruct", "trip_base", "trip_instruct"]
    content = [
        "% Generated by thesis/scripts/build_planning_tables.py.",
        "\\FloatBarrier",
        "\\subsection{Main planning pass@k tables}",
        "\\label{sec:appendix-planning-passk}",
        "The tables below are generated from the stored generation JSON artifacts. Values are exact recomputations from the saved per-sample correctness fields.",
        "",
    ]
    for condition in ordered:
        content.append(table_for_condition(condition, grouped[condition], with_ci=False))

    content.extend(
        [
            "\\FloatBarrier",
            "\\subsection{Bootstrap confidence intervals for planning pass@k}",
            "\\label{sec:appendix-bootstrap-ci}",
            f"Intervals use {BOOTSTRAPS} bootstrap resamples over questions with fixed seed {BOOTSTRAP_SEED}. They quantify uncertainty from the finite question set, not variability across repeated sampling seeds.",
            "",
        ]
    )
    for condition in ordered:
        content.append(table_for_condition(condition, grouped[condition], with_ci=True))

    OUT.write_text("\n".join(content) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
