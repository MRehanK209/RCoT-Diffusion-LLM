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


def strict_raw_correct(benchmark: str, item: dict[str, Any], raw: str) -> bool:
    if benchmark == "countdown":
        return bool(apf.cd_score_single(str(item["question"]), raw))
    if benchmark == "trip_planning":
        cities, durations = str(item["question"]).split("||")
        if not apf.trip_score_single(cities, durations, raw):
            return False
        parsed = apf.parse_trip_response(apf.clean_trip_text(raw))
        gold_cities = [x for x in cities.split("**") if x]
        if len(parsed) != len(gold_cities):
            return False
        gold_durations = [int(x) for x in durations.split("**") if x]
        total_days = sum(gold_durations) - max(0, len(gold_durations) - 1)
        end_match = None
        for match in apf.re.finditer(r"\d+\s*-\s*(\d+)", raw):
            if int(match.group(1)) == total_days:
                end_match = match
                break
        if end_match is None:
            return False
        if apf.re.search(r"\b([A-Za-z]{3,})\b(?:\s+\1\b){3,}", raw, flags=apf.re.I):
            return False
        tail = raw[end_match.end():].lower()
        bad_tail_markers = ("**day", "task:", "your task is", "you are given", "<|endoftext|>", "<|beginoftext|>", "skip to content")
        return not any(marker in tail for marker in bad_tail_markers)
    parsed = apf.sample_analysis(
        benchmark,
        raw,
        item["ground_truth"],
        question=item.get("question"),
        canonical_from_file=None,
    )
    return bool(parsed["canonical_correct"])


def sample_text_by_strict_status(benchmark: str, item: dict[str, Any], want_correct: bool) -> str | None:
    raws = item.get("raw_generations", item.get("generations", []))
    if not raws:
        return None
    candidates: list[str] = []
    for raw in raws:
        text = str(raw)
        if strict_raw_correct(benchmark, item, text) == want_correct:
            candidates.append(text)
    if not candidates:
        return None
    if want_correct and benchmark == "trip_planning":
        def continuation_penalty(text: str) -> tuple[int, int]:
            lower = text.lower()
            has_new_task = int(
                "task:" in lower
                or "your task is" in lower
                or "you are given" in lower
                or "<|endoftext|>" in lower
                or "<|beginoftext|>" in lower
                or "skip to content" in lower
            )
            return has_new_task, len(text)

        candidates.sort(key=continuation_penalty)
        return candidates[0]
    if want_correct:
        candidates.sort(key=len)
        return candidates[0]
    return candidates[0]


def qualitative_sample_penalty(benchmark: str, text: str, want_correct: bool) -> int:
    if want_correct and benchmark == "trip_planning":
        lower = text.lower()
        has_new_task = (
            "task:" in lower
            or "your task is" in lower
            or "you are given" in lower
            or "<|endoftext|>" in lower
            or "<|beginoftext|>" in lower
            or "skip to content" in lower
        )
        return (100000 if has_new_task else 0) + len(text)
    if want_correct:
        return len(text)
    return 0


def choose_model_sample(
    qualitative: dict[str, Any],
    benchmark: str,
    model_pool: list[str],
    question_idx: int,
    want_correct: bool,
) -> tuple[str, str, int] | None:
    for model in model_pool:
        item = qualitative["payloads"][model][question_idx]
        text = sample_text_by_strict_status(benchmark, item, want_correct)
        if text is not None:
            return model, text, qualitative_sample_penalty(benchmark, text, want_correct)
    return None


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
        "\\label{sec:appendix-qualitative}",
        "The following examples list exact question strings and raw model samples copied from the recorded generation artifacts. Line wrapping is a typesetting change only.",
        "\\lstset{basicstyle=\\scriptsize\\ttfamily,breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true,frame=single}",
    ]

    def add_listing(lines: list[str], title: str, text: Any) -> None:
        lines.extend([
            f"\\textbf{{{title}.}}",
            "\\begin{lstlisting}",
            str(text),
            "\\end{lstlisting}",
        ])

    condition_map = {
        "countdown_base": "Countdown-cd4 Base",
        "trip_planning_base": "Trip Planning Base",
    }
    for cond_key, title in condition_map.items():
        bundle = oracle_data[cond_key]
        d_models = [spec["model"] for spec in bundle["specs"] if spec["paradigm"] == "dLLM"]
        a_models = [spec["model"] for spec in bundle["specs"] if spec["paradigm"] == "AR"]
        benchmark = "countdown" if cond_key == "countdown_base" else "trip_planning"
        cases = [
            ("dLLM-only solved", bundle["dllm_only"], d_models, a_models, True, False),
            ("AR-only solved", bundle["ar_only"], a_models, d_models, True, False),
            ("Both model groups solved", bundle["both"], d_models, a_models, True, True),
            ("Neither model group solved", bundle["neither"], d_models, a_models, False, False),
        ]
        md_lines.extend([f"## {title}", ""])
        tex_lines.extend([f"\\subsection{{{title}}}"])
        for label, indices, primary_models, secondary_models, primary_correct, secondary_correct in cases:
            chosen = None
            chosen_penalty = None
            for q_idx in indices:
                primary = choose_model_sample(bundle, benchmark, primary_models, q_idx, primary_correct)
                secondary = choose_model_sample(bundle, benchmark, secondary_models, q_idx, secondary_correct)
                if primary is not None and secondary is not None:
                    penalty = primary[2] + secondary[2]
                    if chosen is None or chosen_penalty is None or penalty < chosen_penalty:
                        chosen = (q_idx, primary, secondary)
                        chosen_penalty = penalty
                    if penalty == 0:
                        break
            if chosen is None:
                continue
            q_idx, primary, secondary = chosen
            q_text = bundle["question_texts"][q_idx]
            primary_model, primary_text, _ = primary
            secondary_model, secondary_text, _ = secondary
            md_lines.extend([
                f"### {label}",
                f"- Question: `{q_text}`",
                f"- {primary_model}: `{primary_text}`",
                f"- {secondary_model}: `{secondary_text}`",
                "",
            ])
            tex_lines.append(f"\\paragraph{{{label}}}")
            add_listing(tex_lines, "Question", q_text)
            add_listing(tex_lines, primary_model, primary_text)
            add_listing(tex_lines, secondary_model, secondary_text)

    gsm_example = None
    for key, example in parser_examples.get("gsm8k", {}).items():
        if "boxed-only miss" in key:
            gsm_example = example
            break
    if gsm_example is not None:
        md_lines.extend([
            "## GSM8K parser-sensitivity example",
            f"- Question: `{gsm_example['question']}`",
            f"- Raw sample: `{gsm_example['raw']}`",
            "",
        ])
        tex_lines.append("\\subsection{GSM8K parser-sensitivity example}")
        add_listing(tex_lines, "Question", gsm_example["question"])
        add_listing(tex_lines, "Raw sample", gsm_example["raw"])

    def failure_title(benchmark: str, category: str) -> str:
        if benchmark == "countdown" and category in {"wrong number use", "wrong target", "division by zero"}:
            return f"Countdown validity-failure example: {category}"
        if benchmark == "countdown":
            return f"Countdown {category} example"
        if benchmark == "trip_planning" and category in {"exact-match failure", "city sequence mismatch", "duration mismatch"}:
            return f"Trip Planning semantic-failure example: {category}"
        return f"Trip Planning parser-entry-failure example: {category}"

    for benchmark in ("countdown", "trip_planning"):
        sample_key, sample = next(iter(parser_examples.get(benchmark, {}).items()), (None, None))
        if sample is None:
            continue
        category = str(sample_key).split(":")[-1] if sample_key is not None else "failure"
        title = failure_title(benchmark, category)
        md_lines.extend([
            f"## {title}",
            f"- Question: `{sample['question']}`",
            f"- Raw sample: `{sample['raw']}`",
            "",
        ])
        tex_lines.append(f"\\subsection{{{title}}}")
        add_listing(tex_lines, "Question", sample["question"])
        add_listing(tex_lines, "Raw sample", sample["raw"])

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
