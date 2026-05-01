#!/usr/bin/env python3
"""Analyze diversity, pass@k growth, complementarity, and qualitative examples."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
import analyze_parser_failures as apf  # noqa: E402


ROOT = apf.ROOT
THESIS = apf.THESIS
TABLES = apf.TABLES


def condition_specs() -> dict[str, list[apf.RunSpec]]:
    runs = apf.run_specs()
    wanted = {
        "gsm8k_base": [("gsm8k", "base_4shot_n128"), 4],
        "gsm8k_instruct": [("gsm8k", "instruct_templated_n128"), 4],
        "countdown_base": [("countdown", "base_refresh"), 4],
        "countdown_instruct": [("countdown", "instruct_refresh"), 4],
        "trip_planning_base": [("trip_planning", "base_main"), 4],
        "trip_planning_instruct": [("trip_planning", "instruct_main"), 4],
    }
    by_cond: dict[str, list[apf.RunSpec]] = {}
    for label, ((benchmark, condition), expected) in wanted.items():
        matched = [run for run in runs if run.benchmark == benchmark and run.condition == condition]
        if len(matched) != expected:
            raise ValueError(f"{label} expected {expected} runs, found {len(matched)}")
        by_cond[label] = matched
    return by_cond


def compare_numeric(pred: Any, gold: Any) -> bool:
    return apf.compare_numeric(pred, gold)


def normalize_answer(benchmark: str, raw: str, gold: Any, question: str | None = None) -> tuple[str | None, str | None, bool]:
    parsed = apf.sample_analysis(benchmark, raw, gold, question=question)
    token = parsed["normalized_answer"]
    valid = parsed["valid_repr"]
    correct = bool(parsed["canonical_correct"])
    return token, valid, correct


def pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0 or n < k:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def entropy_bits(tokens: list[str]) -> float:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def k_grid(n: int) -> list[int]:
    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    return [k for k in ks if k <= n]


def normalized_auc(xs: list[int], ys: list[float]) -> float:
    if len(xs) == 1:
        return ys[0]
    max_x = xs[-1]
    area = 0.0
    for i in range(len(xs) - 1):
        x0 = xs[i] / max_x
        x1 = xs[i + 1] / max_x
        area += (ys[i] + ys[i + 1]) * 0.5 * (x1 - x0)
    return area


def load_run(path: str) -> dict[str, Any]:
    with open(ROOT / path, "r") as handle:
        return json.load(handle)


def per_question_diversity(benchmark: str, item: dict[str, Any]) -> dict[str, Any]:
    raws = item.get("raw_generations", item.get("generations", []))
    existing_answers = item.get("extracted_answer", [None] * len(raws))
    n = len(raws)
    normalized_answers: list[str] = []
    valid_answers: list[str] = []
    correct_answers: list[str] = []
    correct_count = 0
    parse_valid = 0
    for idx, raw in enumerate(raws):
        existing = existing_answers[idx] if idx < len(existing_answers) else None
        parsed = apf.sample_analysis(
            benchmark,
            raw,
            item["ground_truth"],
            question=item.get("question"),
            canonical_from_file=existing,
        )
        norm = parsed["normalized_answer"]
        valid = parsed["valid_repr"]
        correct = bool(parsed["canonical_correct"])
        normalized_answers.append(norm if norm is not None else "<parse-fail>")
        if valid is not None:
            valid_answers.append(valid)
            parse_valid += 1
        if correct:
            correct_count += 1
            if norm is not None:
                correct_answers.append(norm)
    ks = k_grid(n)
    curve = [pass_at_k(n, correct_count, k) for k in ks]
    early_ref = 8 if 8 in ks else ks[-1]
    pass_map = {k: v for k, v in zip(ks, curve)}
    raw_tokens = [str(raw).strip() for raw in raws]
    return {
        "question": item["question"],
        "n_samples": n,
        "unique_raw_outputs": len(set(raw_tokens)),
        "unique_normalized_answers": len({x for x in normalized_answers if x != "<parse-fail>"}),
        "unique_valid_answers": len(set(valid_answers)),
        "unique_correct_answers": len(set(correct_answers)),
        "correct_count": correct_count,
        "duplicate_rate": 1.0 - (len(set(raw_tokens)) / n if n else 0.0),
        "parse_valid_rate": parse_valid / n if n else 0.0,
        "answer_entropy_bits": entropy_bits(normalized_answers),
        "pass_at_1": pass_map.get(1, 0.0),
        "pass_at_8": pass_map.get(early_ref, 0.0),
        "pass_at_max": curve[-1] if curve else 0.0,
        "pass_gain": (curve[-1] - pass_map.get(1, 0.0)) if curve else 0.0,
        "early_gain": pass_map.get(early_ref, 0.0) - pass_map.get(1, 0.0),
        "late_gain": (curve[-1] - pass_map.get(early_ref, 0.0)) if curve else 0.0,
        "pass_auc": normalized_auc(ks, curve) if curve else 0.0,
    }


def summarize_diversity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cond_label, specs in condition_specs().items():
        benchmark = specs[0].benchmark
        for spec in specs:
            data = load_run(spec.path)
            question_rows = [per_question_diversity(benchmark, item) for item in data["generations"]]
            agg = {
                "condition": cond_label,
                "benchmark": benchmark,
                "model": spec.model,
                "paradigm": spec.paradigm,
                "prompt_mode": spec.prompt_mode,
                "file": spec.path,
                "questions": len(question_rows),
                "samples_per_question": question_rows[0]["n_samples"],
                "mean_unique_normalized_answers": sum(r["unique_normalized_answers"] for r in question_rows) / len(question_rows),
                "mean_duplicate_rate": sum(r["duplicate_rate"] for r in question_rows) / len(question_rows),
                "mean_answer_entropy_bits": sum(r["answer_entropy_bits"] for r in question_rows) / len(question_rows),
                "mean_valid_output_diversity": sum(r["unique_valid_answers"] for r in question_rows) / len(question_rows),
                "mean_correct_output_diversity": sum(r["unique_correct_answers"] for r in question_rows) / len(question_rows),
                "mean_parse_valid_rate": sum(r["parse_valid_rate"] for r in question_rows) / len(question_rows),
                "pass_at_1": sum(r["pass_at_1"] for r in question_rows) / len(question_rows),
                "pass_at_8": sum(r["pass_at_8"] for r in question_rows) / len(question_rows),
                "pass_at_max": sum(r["pass_at_max"] for r in question_rows) / len(question_rows),
                "pass_gain": sum(r["pass_gain"] for r in question_rows) / len(question_rows),
                "early_gain": sum(r["early_gain"] for r in question_rows) / len(question_rows),
                "late_gain": sum(r["late_gain"] for r in question_rows) / len(question_rows),
                "pass_auc": sum(r["pass_auc"] for r in question_rows) / len(question_rows),
            }
            rows.append(agg)
    return rows


def solved_by_question(item: dict[str, Any]) -> bool:
    gt = item["ground_truth"]
    for answer in item["extracted_answer"]:
        if answer is not None and compare_numeric(answer, gt):
            return True
    return False


def build_condition_question_sets(specs: list[apf.RunSpec]) -> tuple[list[str], dict[str, set[int]], dict[str, dict[int, dict[str, Any]]]]:
    question_texts = None
    solved_sets: dict[str, set[int]] = {}
    payloads: dict[str, dict[int, dict[str, Any]]] = {}
    for spec in specs:
        data = load_run(spec.path)
        questions = [item["question"] for item in data["generations"]]
        if question_texts is None:
            question_texts = questions
        solved = set()
        payload_map = {}
        for idx, item in enumerate(data["generations"]):
            if solved_by_question(item):
                solved.add(idx)
            payload_map[idx] = item
        solved_sets[spec.model] = solved
        payloads[spec.model] = payload_map
    assert question_texts is not None
    return question_texts, solved_sets, payloads


def oracle_summary() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qualitative: dict[str, Any] = {}
    for cond_label, specs in condition_specs().items():
        question_texts, solved_sets, payloads = build_condition_question_sets(specs)
        n_questions = len(question_texts)
        dllm_specs = [s for s in specs if s.paradigm == "dLLM"]
        ar_specs = [s for s in specs if s.paradigm == "AR"]
        dllm_union = set().union(*(solved_sets[s.model] for s in dllm_specs))
        ar_union = set().union(*(solved_sets[s.model] for s in ar_specs))
        both = dllm_union & ar_union
        dllm_only = dllm_union - ar_union
        ar_only = ar_union - dllm_union
        neither = set(range(n_questions)) - (dllm_union | ar_union)
        best_single_model = max(specs, key=lambda s: len(solved_sets[s.model]))
        best_pair = None
        best_pair_union: set[int] = set()
        for d_spec in dllm_specs:
            for a_spec in ar_specs:
                union = solved_sets[d_spec.model] | solved_sets[a_spec.model]
                if len(union) > len(best_pair_union):
                    best_pair_union = union
                    best_pair = (d_spec.model, a_spec.model)
        all_model_union = set().union(*(solved_sets[s.model] for s in specs))
        disagreement_rate = (len(dllm_only) + len(ar_only)) / n_questions
        rows.append({
            "condition": cond_label,
            "benchmark": specs[0].benchmark,
            "questions": n_questions,
            "dllm_only_solved": len(dllm_only),
            "ar_only_solved": len(ar_only),
            "both_solved": len(both),
            "neither_solved": len(neither),
            "best_single_model": best_single_model.model,
            "best_single_solved": len(solved_sets[best_single_model.model]),
            "best_dllm_plus_best_ar_pair": " + ".join(best_pair) if best_pair else "",
            "best_dllm_plus_best_ar_union": len(best_pair_union),
            "all_model_oracle_union": len(all_model_union),
            "improvement_over_best_single": len(best_pair_union) - len(solved_sets[best_single_model.model]),
            "disagreement_rate": disagreement_rate,
            "complementarity_gain": (len(best_pair_union) - len(solved_sets[best_single_model.model])) / n_questions,
        })
        qualitative[cond_label] = {
            "question_texts": question_texts,
            "dllm_only": sorted(dllm_only),
            "ar_only": sorted(ar_only),
            "both": sorted(both),
            "neither": sorted(neither),
            "payloads": payloads,
            "specs": [spec.__dict__ for spec in specs],
        }
    return rows, qualitative


def choose_example_payload(qualitative: dict[str, Any], model_pool: list[str], question_idx: int) -> tuple[str, dict[str, Any]]:
    for model in model_pool:
        return model, qualitative["payloads"][model][question_idx]
    raise ValueError("No payload available")


def first_sample_text(item: dict[str, Any], prefer_correct: bool = False) -> str:
    raws = item.get("raw_generations", item.get("generations", []))
    if not raws:
        return ""
    if prefer_correct:
        gt = item["ground_truth"]
        for answer, raw in zip(item["extracted_answer"], raws):
            if answer is not None and compare_numeric(answer, gt):
                return str(raw).strip()[:500]
    return str(raws[0]).strip()[:500]


def tex_escape(text: str) -> str:
    compact = " ".join(str(text).split())
    return (
        compact
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("$", "\\$")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_qualitative_examples(oracle_data: dict[str, Any], parser_examples: dict[str, Any]) -> None:
    md_lines = ["# Qualitative Examples", ""]
    tex_lines = [
        "\\section{Qualitative Examples}",
        "The following compact examples are extracted from the thesis generation JSON files.",
    ]

    condition_map = {
        "countdown_base": "Countdown-cd4 Base",
        "trip_planning_base": "Trip Planning Base",
    }
    for cond_key, title in condition_map.items():
        bundle = oracle_data[cond_key]
        d_models = [spec["model"] for spec in bundle["specs"] if spec["paradigm"] == "dLLM"]
        a_models = [spec["model"] for spec in bundle["specs"] if spec["paradigm"] == "AR"]
        cases = [
            ("dLLM-only solved", bundle["dllm_only"], d_models, a_models),
            ("AR-only solved", bundle["ar_only"], a_models, d_models),
            ("Both paradigms solved", bundle["both"], d_models, a_models),
            ("Neither paradigm solved", bundle["neither"], d_models, a_models),
        ]
        md_lines.extend([f"## {title}", ""])
        tex_lines.extend([f"\\subsection{{{title}}}"])
        for label, indices, primary_models, secondary_models in cases:
            if not indices:
                continue
            q_idx = indices[0]
            q_text = bundle["question_texts"][q_idx]
            primary_model, primary_item = choose_example_payload(bundle, primary_models, q_idx)
            secondary_model, secondary_item = choose_example_payload(bundle, secondary_models, q_idx)
            q_text_tex = tex_escape(q_text[:180])
            primary_text_tex = tex_escape(first_sample_text(primary_item, prefer_correct=True)[:220])
            secondary_text_tex = tex_escape(first_sample_text(secondary_item, prefer_correct=False)[:220])
            md_lines.extend([
                f"### {label}",
                f"- Question: `{q_text}`",
                f"- {primary_model}: `{first_sample_text(primary_item, prefer_correct=True)}`",
                f"- {secondary_model}: `{first_sample_text(secondary_item, prefer_correct=False)}`",
                "",
            ])
            tex_lines.extend([
                f"\\paragraph{{{label}}}",
                f"\\textbf{{Question.}} \\texttt{{{q_text_tex}}}\\\\",
                f"\\textbf{{{primary_model}.}} \\texttt{{{primary_text_tex}}}\\\\",
                f"\\textbf{{{secondary_model}.}} \\texttt{{{secondary_text_tex}}}",
            ])

    gsm_example = None
    for key, example in parser_examples.get("gsm8k", {}).items():
        if "boxed-only miss" in key:
            gsm_example = example
            break
    if gsm_example is not None:
        gsm_q_tex = tex_escape(gsm_example["question"][:180])
        gsm_raw_tex = tex_escape(gsm_example["raw"][:260])
        md_lines.extend([
            "## GSM8K parser-sensitivity example",
            f"- Question: `{gsm_example['question']}`",
            f"- Raw sample: `{gsm_example['raw']}`",
            "",
        ])
        tex_lines.extend([
            "\\subsection{GSM8K parser-sensitivity example}",
            f"\\textbf{{Question.}} \\texttt{{{gsm_q_tex}}}\\\\",
            f"\\textbf{{Raw sample.}} \\texttt{{{gsm_raw_tex}}}",
        ])

    for benchmark in ("countdown", "trip_planning"):
        sample = next(iter(parser_examples.get(benchmark, {}).values()), None)
        if sample is None:
            continue
        sample_q_tex = tex_escape(sample["question"][:180])
        sample_raw_tex = tex_escape(sample["raw"][:260])
        md_lines.extend([
            f"## {benchmark} parser-failure example",
            f"- Question: `{sample['question']}`",
            f"- Raw sample: `{sample['raw']}`",
            "",
        ])
        tex_lines.extend([
            f"\\subsection{{{benchmark.replace('_', ' ').title()} parser-failure example}}",
            f"\\textbf{{Question.}} \\texttt{{{sample_q_tex}}}\\\\",
            f"\\textbf{{Raw sample.}} \\texttt{{{sample_raw_tex}}}",
        ])

    (TABLES / "qualitative_examples.md").write_text("\n".join(md_lines))
    appendices = THESIS / "appendices"
    appendices.mkdir(parents=True, exist_ok=True)
    (appendices / "qualitative_examples.tex").write_text("\n".join(tex_lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    diversity_rows = summarize_diversity()
    write_csv(TABLES / "diversity_summary.csv", diversity_rows)
    oracle_rows, oracle_examples = oracle_summary()
    write_csv(TABLES / "oracle_ensemble_summary.csv", oracle_rows)
    parser_examples = json.loads((TABLES / "parser_examples.json").read_text()) if (TABLES / "parser_examples.json").exists() else {}
    write_qualitative_examples(oracle_examples, parser_examples)
    print(f"Wrote {len(diversity_rows)} diversity rows")
    print(f"Wrote {len(oracle_rows)} oracle rows")
    print(f"Wrote qualitative examples to {TABLES / 'qualitative_examples.md'}")


if __name__ == "__main__":
    main()
