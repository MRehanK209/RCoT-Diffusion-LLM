#!/usr/bin/env python3
"""Build thesis figures from repository result JSONs and analysis CSVs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "thesis"
IMAGES = THESIS / "images"
TABLES = THESIS / "tables"


COLORS = {
    "LLaDA": "#c4473a",
    "Dream": "#2e6fbb",
    "Qwen": "#2f8f46",
    "Llama": "#8f5ab5",
    "dLLM": "#c4473a",
    "AR": "#2f8f46",
    "Cross": "#666666",
    "Flat": "#d9861f",
    "Templated": "#2e6fbb",
}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def ensure_dirs() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)


def load_passk(path: str) -> dict[str, dict[int, float]]:
    payload = json.loads((ROOT / path).read_text())
    data = payload["results"]
    if set(data.keys()) == {"trip_planning"}:
        data = data["trip_planning"]
    return {name: {int(k): float(v) for k, v in vals.items()} for name, vals in data.items()}


def short_name(name: str) -> str:
    for base in ("LLaDA", "Dream", "Qwen", "Llama"):
        if base in name:
            suffix = ""
            if "flat" in name.lower():
                suffix = " flat"
            elif "Instruct" in name or "Inst" in name:
                suffix = " inst"
            elif "Base" in name:
                suffix = " base"
            return base + suffix
    return name


def model_color(name: str) -> str:
    for key in ("LLaDA", "Dream", "Qwen", "Llama"):
        if key in name:
            return COLORS[key]
    return "#444444"


def pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def configure_line_axis(ax, title: str, y_label: str = "pass@k") -> None:
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel(y_label)
    ax.set_xscale("log", base=2)
    ax.grid(axis="y", alpha=0.25)


def plot_gsm8k_passk() -> None:
    panels = [
        ("Base, 4-shot, n=128", load_passk("results/milestone2_gsm8k_base/passk_gsm8k_comparison.json")),
        ("Base, 4-shot, n=16", load_passk("results/milestone2_gsm8k_base_4shot/passk_gsm8k_comparison.json")),
        ("Instruct, 0-shot, n=16", load_passk("results/milestone2_gsm8k_instruct_0shot/passk_gsm8k_comparison.json")),
        ("Instruct, 4-shot, n=16", load_passk("results/milestone2_gsm8k_instruct_4shot/passk_gsm8k_comparison.json")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, series) in zip(axes.flat, panels):
        for name, values in series.items():
            xs = sorted(values)
            ys = [values[k] * 100 for k in xs]
            ax.plot(xs, ys, marker="o", lw=2, ms=4, color=model_color(name), label=short_name(name))
        configure_line_axis(ax, title, "pass@k (%)")
        ax.set_xticks(sorted(next(iter(series.values())).keys()))
        ax.set_ylim(0, 102)
        pct_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], ["LLaDA", "Dream", "Qwen", "Llama"], ncol=4, loc="upper center", frameon=False)
    fig.suptitle("GSM8K pass@k across prompt and few-shot settings")
    fig.savefig(IMAGES / "gsm8k_passk_curves.pdf")
    plt.close(fig)


def plot_gsm8k_prompt_sensitivity() -> None:
    rows = list(csv.DictReader((TABLES / "main_results_summary.csv").open()))
    models = ["LLaDA", "Dream", "Qwen", "Llama"]
    templated = []
    flat = []
    for model in models:
        templated.append(100 * float(next(
            row["pass@1"]
            for row in rows
            if row["benchmark_condition"] == "GSM8K instruct templated" and model in row["model"]
        )))
        flat.append(100 * float(next(
            row["pass@1"]
            for row in rows
            if row["benchmark_condition"] == "GSM8K instruct flat" and model in row["model"]
        )))
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar([i - 0.18 for i in x], templated, width=0.36, color=COLORS["Templated"], label="Templated")
    ax.bar([i + 0.18 for i in x], flat, width=0.36, color=COLORS["Flat"], label="Flat")
    ax.set_xticks(list(x), models)
    ax.set_ylabel("pass@1 (%)")
    ax.set_ylim(0, 100)
    pct_axis(ax)
    ax.set_title("GSM8K instruct prompt sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for xi, val in enumerate(templated):
        ax.text(xi - 0.18, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, val in enumerate(flat):
        ax.text(xi + 0.18, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(IMAGES / "gsm8k_prompt_sensitivity.pdf")
    plt.close(fig)


def plot_countdown_base() -> None:
    series = load_passk("results/milestone2_countdown_base_refresh/passk_countdown_cd4_comparison.json")
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.axvspan(8, 16, color="#e6e6e6", alpha=0.8, zorder=0)
    for name, values in series.items():
        xs = sorted(values)
        ys = [values[k] * 100 for k in xs]
        ax.plot(xs, ys, marker="o", lw=2.3, ms=4.5, color=model_color(name), label=short_name(name))
    configure_line_axis(ax, "Countdown-cd4 base pass@k crossover", "pass@k (%)")
    ax.set_xticks(sorted(next(iter(series.values())).keys()))
    ax.set_ylim(0, 56)
    pct_axis(ax)
    ax.text(10.8, 52, "crossover region", ha="center", va="top", fontsize=9, color="#555555")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(IMAGES / "countdown_base_crossover.pdf")
    plt.close(fig)


def plot_countdown_instruct() -> None:
    series = load_passk("results/milestone2_countdown_instruct_refresh/passk_countdown_cd4_comparison.json")
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    for name, values in series.items():
        xs = sorted(values)
        ys = [values[k] * 100 for k in xs]
        ax.plot(xs, ys, marker="o", lw=2.3, ms=4.5, color=model_color(name), label=short_name(name))
    configure_line_axis(ax, "Countdown-cd4 instruct pass@k", "pass@k (%)")
    ax.set_xticks(sorted(next(iter(series.values())).keys()))
    ax.set_ylim(0, 50)
    pct_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(IMAGES / "countdown_instruct_passk.pdf")
    plt.close(fig)


def plot_countdown_prompt_diagnostic() -> None:
    data = load_passk("results/passk_countdown_cd4_comparison.json")
    names = [
        "LLaDA-8B-Instruct [fast-dllm]",
        "LLaDA-8B-Instruct [flat] [fast-dllm]",
        "Dream-v0-Instruct-7B [fast-dllm]",
        "Dream-v0-Instruct-7B [flat] [fast-dllm]",
        "Qwen2.5-7B-Instruct [vllm]",
        "Qwen2.5-7B-Instruct [flat] [vllm]",
        "Llama-3.1-8B-Instruct [vllm]",
        "Llama-3.1-8B-Instruct [flat] [vllm]",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    for name in names:
        values = data[name]
        base = short_name(name)
        style = "--" if "flat" in name.lower() else "-"
        ax.plot(sorted(values), [values[k] * 100 for k in sorted(values)], marker="o", lw=2, ms=4, ls=style, color=model_color(name), label=base)
    configure_line_axis(ax, "Countdown prompt-mode diagnostic", "pass@k (%)")
    ax.set_xticks(sorted(next(iter(data.values())).keys()))
    ax.set_ylim(0, 55)
    pct_axis(ax)
    legend = [
        Line2D([0], [0], color=COLORS["LLaDA"], lw=2, label="LLaDA"),
        Line2D([0], [0], color=COLORS["Dream"], lw=2, label="Dream"),
        Line2D([0], [0], color=COLORS["Qwen"], lw=2, label="Qwen"),
        Line2D([0], [0], color=COLORS["Llama"], lw=2, label="Llama"),
        Line2D([0], [0], color="#444444", lw=2, ls="-", label="templated"),
        Line2D([0], [0], color="#444444", lw=2, ls="--", label="flat"),
    ]
    ax.legend(handles=legend, ncol=3, frameon=False, loc="upper left")
    fig.savefig(IMAGES / "countdown_prompt_diagnostic.pdf")
    plt.close(fig)


def plot_trip_planning() -> None:
    legacy = load_passk("results/aime_data_analysis_large_k_comparison.json")
    llada = load_passk("results/milestone2_trip_planning_llada_passk/passk_trip_planning_comparison.json")
    llama = load_passk("results/milestone2_trip_planning_llama_passk/passk_trip_planning_comparison.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    base_names = {
        "Dream-v0-Base-7B (Base)": legacy["Dream-v0-Base-7B (Base)"],
        "LLaDA-8B-Base [fast-dllm]": llada["LLaDA-8B-Base [fast-dllm]"],
        "Qwen2.5-7B (Base)": legacy["Qwen2.5-7B (Base)"],
        "Llama-3.1-8B [vllm]": llama["Llama-3.1-8B [vllm]"],
    }
    inst_names = {
        "Dream-v0-Instruct-7B (Inst)": legacy["Dream-v0-Instruct-7B (Inst)"],
        "LLaDA-8B-Instruct [fast-dllm]": llada["LLaDA-8B-Instruct [fast-dllm]"],
        "LLaDA-8B-Instruct [flat] [fast-dllm]": llada["LLaDA-8B-Instruct [flat] [fast-dllm]"],
        "Qwen2.5-7B-Instruct (Inst)": legacy["Qwen2.5-7B-Instruct (Inst)"],
        "Llama-3.1-8B-Instruct [vllm]": llama["Llama-3.1-8B-Instruct [vllm]"],
        "Llama-3.1-8B-Instruct [flat] [vllm]": llama["Llama-3.1-8B-Instruct [flat] [vllm]"],
    }
    for ax, title, series in [(axes[0], "Trip Planning base", base_names), (axes[1], "Trip Planning instruct", inst_names)]:
        for name, values in series.items():
            style = "--" if "flat" in name.lower() else "-"
            ax.plot(sorted(values), [values[k] * 100 for k in sorted(values)], marker="o", lw=2.2, ms=4, ls=style, color=model_color(name), label=short_name(name))
        configure_line_axis(ax, title, "pass@k (%)")
        ax.set_xticks(sorted(next(iter(series.values())).keys()))
        ax.set_ylim(0, 31)
        pct_axis(ax)
        ax.grid(axis="y", alpha=0.25)
    legend = [
        Line2D([0], [0], color=COLORS["LLaDA"], lw=2, label="LLaDA"),
        Line2D([0], [0], color=COLORS["Dream"], lw=2, label="Dream"),
        Line2D([0], [0], color=COLORS["Qwen"], lw=2, label="Qwen"),
        Line2D([0], [0], color=COLORS["Llama"], lw=2, label="Llama"),
        Line2D([0], [0], color="#444444", lw=2, ls="-", label="templated/base"),
        Line2D([0], [0], color="#444444", lw=2, ls="--", label="flat"),
    ]
    axes[0].legend(handles=legend, frameon=False, ncol=2, loc="upper left")
    fig.savefig(IMAGES / "trip_planning_passk.pdf")
    plt.close(fig)


def plot_hyperparameter_summary() -> None:
    models = ["LLaDA", "Dream", "Qwen"]
    best1 = [71.01, 70.66, 72.97]
    bestk = [99.22, 99.22, 100.0]
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar([i - 0.17 for i in x], best1, width=0.34, color=COLORS["Templated"], label="Best pass@1")
    ax.bar([i + 0.17 for i in x], bestk, width=0.34, color=COLORS["AR"], label="Best high-k")
    ax.set_xticks(list(x), models)
    ax.set_ylim(0, 105)
    pct_axis(ax)
    ax.set_ylabel("Score (%)")
    ax.set_title("GSM8K sweep best observed scores")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for xi, val in enumerate(best1):
        ax.text(xi - 0.17, val + 1.2, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, val in enumerate(bestk):
        ax.text(xi + 0.17, val + 1.2, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(IMAGES / "hyperparameter_summary.pdf")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def plot_correlation_summary() -> None:
    conditions = ["Countdown Base", "Countdown Inst", "Trip Base", "Trip Inst", "GSM8K Base", "GSM8K Inst"]
    cols = ["dLLM-dLLM", "AR-AR", "Cross"]
    values = [
        [0.6382, 0.6149, 0.5718],
        [0.1475, 0.4039, 0.4345],
        [0.3451, 0.7061, 0.5394],
        [0.4622, 0.7190, 0.2484],
        [0.6291, 0.3204, 0.6189],
        [0.5672, 0.6638, 0.6550],
    ]
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_title("Per-question correlation summary")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Pearson r")
    fig.savefig(IMAGES / "correlation_summary.pdf")
    plt.close(fig)


def _box(ax, xy, text, width=0.18, height=0.10, fc="#f7f7f7", ec="#333333"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)


def _arrow(ax, start, end, color="#555555"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.3, color=color, transform=ax.transAxes))


def plot_parser_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.axis("off")
    _box(ax, (0.03, 0.72), "Raw model\ncompletions", fc="#f3efe8")
    _box(ax, (0.27, 0.72), "Task-specific\nparser", fc="#edf3fb")
    _box(ax, (0.50, 0.82), "GSM8K:\nboxed / numeric\nextraction", width=0.16, height=0.11, fc="#fbeee7")
    _box(ax, (0.50, 0.58), "Countdown:\nchain validation", width=0.16, height=0.11, fc="#eef7ea")
    _box(ax, (0.50, 0.34), "Trip Planning:\nitinerary validation", width=0.16, height=0.11, fc="#efeaf8")
    _box(ax, (0.74, 0.72), "Sample correctness\nvectors", fc="#f7f7f7")
    _box(ax, (0.88, 0.72), "Accuracy /\npass@k /\ncorrelation", width=0.10, height=0.12, fc="#fff6e6")
    _box(ax, (0.25, 0.18), "Parse failure", width=0.13, height=0.08, fc="#fdecea", ec="#aa4a44")
    _box(ax, (0.43, 0.18), "Format failure", width=0.13, height=0.08, fc="#fff3cd", ec="#a08518")
    _box(ax, (0.61, 0.18), "Validity failure", width=0.13, height=0.08, fc="#fdecea", ec="#aa4a44")
    _arrow(ax, (0.21, 0.77), (0.27, 0.77))
    _arrow(ax, (0.45, 0.77), (0.50, 0.87))
    _arrow(ax, (0.45, 0.77), (0.50, 0.63))
    _arrow(ax, (0.45, 0.77), (0.50, 0.39))
    _arrow(ax, (0.66, 0.87), (0.74, 0.77))
    _arrow(ax, (0.66, 0.63), (0.74, 0.77))
    _arrow(ax, (0.66, 0.39), (0.74, 0.77))
    _arrow(ax, (0.84, 0.77), (0.88, 0.77))
    _arrow(ax, (0.36, 0.72), (0.31, 0.26), color="#aa4a44")
    _arrow(ax, (0.48, 0.72), (0.49, 0.26), color="#a08518")
    _arrow(ax, (0.58, 0.55), (0.67, 0.26), color="#aa4a44")
    ax.set_title("Parser layer between raw generations and metrics")
    fig.savefig(IMAGES / "parser_layer_pipeline.pdf")
    plt.close(fig)


def plot_parser_failure_rates() -> None:
    rows = read_csv(TABLES / "parser_failure_summary.csv")
    key_conditions = [
        ("gsm8k", "base_4shot_n128", "GSM8K base"),
        ("gsm8k", "instruct_templated_n128", "GSM8K instruct"),
        ("countdown", "base_refresh", "Countdown base"),
        ("countdown", "instruct_refresh", "Countdown instruct"),
        ("trip_planning", "base_main", "Trip base"),
        ("trip_planning", "instruct_main", "Trip instruct"),
    ]
    models = ["LLaDA", "Dream", "Qwen", "Llama"]
    matrix = []
    labels = []
    for benchmark, condition, label in key_conditions:
        labels.append(label)
        row = []
        for model in models:
            match = next(r for r in rows if r["benchmark"] == benchmark and r["condition"] == condition and r["model"] == model)
            row.append(float(match["parse_failure_rate"]) * 100)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="OrRd", vmin=0, vmax=max(max(row) for row in matrix) or 1, aspect="auto")
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Primary parser-entry failure rates")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Parser-entry failure rate")
    fig.savefig(IMAGES / "parser_failure_rates.pdf")
    plt.close(fig)


def plot_parser_sensitivity_summary() -> None:
    gsm_rows = read_csv(TABLES / "parser_sensitivity_gsm8k.csv")
    countdown_rows = read_csv(TABLES / "parser_sensitivity_countdown.csv")
    trip_rows = read_csv(TABLES / "parser_sensitivity_trip.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)

    gsm_target = [r for r in gsm_rows if r["condition"] == "instruct_templated_n128"]
    axes[0].bar(
        [r["model"] for r in gsm_target],
        [100 * float(r["canonical_false_alternate_true"]) / float(r["total_samples"]) for r in gsm_target],
        color=[COLORS[m] for m in [r["model"] for r in gsm_target]],
    )
    axes[0].set_title("GSM8K box-only misses")
    axes[0].set_ylabel("Samples rescued by alternate parser")
    pct_axis(axes[0])

    cd_target = [r for r in countdown_rows if r["condition"] == "base_refresh"]
    axes[1].bar(
        [r["model"] for r in cd_target],
        [100 * (float(r["alternate_correct_samples"]) - float(r["correct_samples"])) / float(r["total_samples"]) for r in cd_target],
        color=[COLORS[m] for m in [r["model"] for r in cd_target]],
    )
    axes[1].set_title("Countdown alternate-parser delta")
    axes[1].set_ylabel("Change in correct-sample rate")
    axes[1].axhline(0, color="#444444", lw=1)
    pct_axis(axes[1])

    trip_target = [r for r in trip_rows if r["condition"] == "instruct_main"]
    trip_vals = []
    trip_models = []
    for row in trip_target:
        parse_fail = 100 * float(row["parse_failures"]) / float(row["total_samples"])
        trip_vals.append(parse_fail)
        trip_models.append(row["model"])
    axes[2].bar(trip_models, trip_vals, color=[COLORS[m] for m in trip_models])
    axes[2].set_title("Trip malformed-plan rate")
    axes[2].set_ylabel("Parse failure rate")
    pct_axis(axes[2])
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(IMAGES / "parser_sensitivity_summary.pdf")
    plt.close(fig)


def plot_diversity_vs_gain() -> None:
    rows = read_csv(TABLES / "diversity_summary.csv")
    bench_labels = {
        "gsm8k": "GSM8K",
        "countdown": "CD",
        "trip_planning": "Trip",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.3), constrained_layout=True)
    for row in rows:
        x = float(row["mean_unique_normalized_answers"])
        y = float(row["pass_gain"]) * 100
        name = row["model"]
        ax.scatter(x, y, s=70, color=COLORS[name], alpha=0.85)
        cond = row["condition"].replace("trip_planning_", "trip_")
        suffix = "inst" if cond.endswith("instruct") else "base"
        ax.text(x + 0.02, y + 0.12, f"{name} {bench_labels[row['benchmark']]}-{suffix}", fontsize=8)
    ax.set_xlabel("Mean unique normalized answers per question")
    ax.set_ylabel("pass@max - pass@1 (pp)")
    ax.set_title("Diversity and pass@k growth")
    ax.grid(alpha=0.25)
    fig.savefig(IMAGES / "diversity_vs_passk_gain.pdf")
    plt.close(fig)


def _grouped_metric_figure(filename: str, metric: str, title: str, ylabel: str) -> None:
    rows = read_csv(TABLES / "diversity_summary.csv")
    conditions = ["gsm8k_base", "gsm8k_instruct", "countdown_base", "countdown_instruct", "trip_planning_base", "trip_planning_instruct"]
    models = ["LLaDA", "Dream", "Qwen", "Llama"]
    x = list(range(len(conditions)))
    fig, ax = plt.subplots(figsize=(11.6, 4.8), constrained_layout=True)
    width = 0.18
    for i, model in enumerate(models):
        vals = []
        for cond in conditions:
            match = next(r for r in rows if r["condition"] == cond and r["model"] == model)
            vals.append(float(match[metric]))
        ax.bar([xi + (i - 1.5) * width for xi in x], vals, width=width, color=COLORS[model], label=model)
    labels = ["GSM base", "GSM inst", "CD base", "CD inst", "Trip base", "Trip inst"]
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    if "rate" in metric or "gain" in metric:
        pct_axis(ax)
    fig.savefig(IMAGES / filename)
    plt.close(fig)


def plot_oracle_gain() -> None:
    rows = read_csv(TABLES / "oracle_ensemble_summary.csv")
    labels = [row["condition"].replace("_", " ") for row in rows]
    vals = [100 * float(row["complementarity_gain"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.bar(labels, vals, color="#4c78a8")
    ax.set_ylabel("Gain over best single model")
    ax.set_title("Oracle complementarity gain")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    pct_axis(ax)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.savefig(IMAGES / "oracle_ensemble_gain.pdf")
    plt.close(fig)


def plot_overlap_stacked() -> None:
    rows = read_csv(TABLES / "oracle_ensemble_summary.csv")
    labels = [row["condition"].replace("_", " ") for row in rows]
    dllm_only = [100 * float(row["dllm_only_solved"]) / float(row["questions"]) for row in rows]
    both = [100 * float(row["both_solved"]) / float(row["questions"]) for row in rows]
    ar_only = [100 * float(row["ar_only_solved"]) / float(row["questions"]) for row in rows]
    neither = [100 * float(row["neither_solved"]) / float(row["questions"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.4, 4.9), constrained_layout=True)
    ax.bar(labels, dllm_only, color=COLORS["LLaDA"], label="dLLM only")
    ax.bar(labels, both, bottom=dllm_only, color="#9ecae1", label="Both")
    ax.bar(labels, ar_only, bottom=[a + b for a, b in zip(dllm_only, both)], color=COLORS["Qwen"], label="AR only")
    ax.bar(labels, neither, bottom=[a + b + c for a, b, c in zip(dllm_only, both, ar_only)], color="#d9d9d9", label="Neither")
    ax.set_ylabel("Questions")
    ax.set_title("Success-set overlap by condition")
    pct_axis(ax)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.savefig(IMAGES / "paradigm_overlap_stacked.pdf")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    plot_gsm8k_passk()
    plot_gsm8k_prompt_sensitivity()
    plot_countdown_base()
    plot_countdown_instruct()
    plot_countdown_prompt_diagnostic()
    plot_trip_planning()
    plot_hyperparameter_summary()
    plot_correlation_summary()
    plot_parser_pipeline()
    if (TABLES / "parser_failure_summary.csv").exists():
        plot_parser_failure_rates()
        plot_parser_sensitivity_summary()
    if (TABLES / "diversity_summary.csv").exists():
        plot_diversity_vs_gain()
        _grouped_metric_figure("duplicate_rate_by_model.pdf", "mean_duplicate_rate", "Duplicate rate by model", "Mean duplicate rate")
        _grouped_metric_figure("passk_gain_by_model.pdf", "pass_gain", "pass@k gain by model", "pass@k gain")
    if (TABLES / "oracle_ensemble_summary.csv").exists():
        plot_oracle_gain()
        plot_overlap_stacked()
    print(f"Wrote figures to {IMAGES}")


if __name__ == "__main__":
    main()
