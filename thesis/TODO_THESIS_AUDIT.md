# Thesis Audit Notes

## Files Inspected

- `thesis/thesis.tex`
- `thesis/chapters/body.tex`
- `thesis/chapters/01_introduction.tex`
- `thesis/chapters/02_related_work.tex`
- `thesis/chapters/03_model_background.tex`
- `thesis/chapters/04_methodology.tex`
- `thesis/chapters/05_results.tex`
- `thesis/chapters/05a_results_gsm8k.tex`
- `thesis/chapters/05b_results_countdown.tex`
- `thesis/chapters/05c_results_trip_planning.tex`
- `thesis/chapters/05d_results_hyperparameters.tex`
- `thesis/chapters/05e_results_parser_sensitivity.tex`
- `thesis/chapters/05f_results_diversity.tex`
- `thesis/chapters/05g_results_correlation_oracle.tex`
- `thesis/chapters/06_discussion.tex`
- `thesis/chapters/07_limitations.tex`
- `thesis/chapters/08_conclusion.tex`
- `thesis/images/`
- `thesis/scripts/build_thesis_figures.py`
- `thesis/literature.bib`
- `thesis/README.md`
- `results/` pass@k JSONs, generation JSONs, and HTML comparison files
- `inference_and_generation.py`
- `evaluate_pass_k.py`
- `metrics/pass_k.py`
- `metrics/parsers.py`
- `dataset/gsm8k.py`
- `dataset/countdown.py`
- `dataset/trip_planning.py`
- `utils.py`
- `metrics/parse_and_get_acc.py`
- `metrics/parser_json.py`
- `generate_comparison_html.py`

## Scripts Added or Updated

- `thesis/scripts/analyze_parser_failures.py`
- `thesis/scripts/analyze_diversity_and_ensemble.py`
- `thesis/scripts/build_thesis_figures.py`

## Analyses Completed

- Re-audited canonical parser paths for GSM8K, Countdown-cd4, and Trip Planning.
- Computed parser failure summaries and parser-sensitivity CSVs.
- Computed GSM8K alternate-parser rescue rates without changing canonical thesis scores.
- Computed diversity metrics from generation JSONs:
  unique normalized answers, duplicate rate, answer entropy, parser-valid diversity, and pass@k gain.
- Computed oracle complementarity / union upper bounds from per-question solved sets.
- Extracted compact qualitative examples from stored generations.
- Split the previous monolithic `chapters/body.tex` into focused chapter files and results-subsection files. `body.tex` now only orchestrates the includes.

## Figures Generated

- `images/gsm8k_passk_curves.pdf`
- `images/gsm8k_prompt_sensitivity.pdf`
- `images/countdown_base_crossover.pdf`
- `images/countdown_instruct_passk.pdf`
- `images/countdown_prompt_diagnostic.pdf`
- `images/trip_planning_passk.pdf`
- `images/hyperparameter_summary.pdf`
- `images/correlation_summary.pdf`
- `images/parser_layer_pipeline.pdf`
- `images/parser_failure_rates.pdf`
- `images/parser_sensitivity_summary.pdf`
- `images/diversity_vs_passk_gain.pdf`
- `images/duplicate_rate_by_model.pdf`
- `images/passk_gain_by_model.pdf`
- `images/oracle_ensemble_gain.pdf`
- `images/paradigm_overlap_stacked.pdf`

## Tables / CSVs Generated

- `tables/parser_failure_summary.csv`
- `tables/parser_sensitivity_gsm8k.csv`
- `tables/parser_sensitivity_countdown.csv`
- `tables/parser_sensitivity_trip.csv`
- `tables/diversity_summary.csv`
- `tables/oracle_ensemble_summary.csv`
- `tables/qualitative_examples.md`
- `appendices/qualitative_examples.tex`

## Key New Findings

- GSM8K parser failures are rare, but canonical boxed-only scoring misses some correct unboxed answers.
  - In the instruct-templated 128-sample run, the alternate fallback rescues 3.24 percentage points for LLaDA and 4.03 points for Qwen.
- Countdown parser crashes are negligible.
  - The dominant failure mode is invalid number use, not parser fragility.
  - The alternate single-expression parser does not rescue additional canonical correct samples.
- Trip Planning is parser-heavy.
  - Base runs have very high malformed-plan rates.
  - Llama-Instruct templated still fails to produce parseable plans on 68.0% of samples.
- Diversity metrics support the thesis interpretation on the planning tasks.
  - Countdown Base: LLaDA averages 1.31 unique normalized answers per question versus 44.65 for Qwen and 58.84 for Llama.
  - The largest pass@k gains occur in the strongest AR planning runs.
- Oracle complementarity is nontrivial on planning benchmarks.
  - Countdown Base: best single model solves 509/992, Dream+Qwen oracle union solves 541/992, all-model union solves 604/992.
  - Trip Planning Base/Instruct: Dream+Llama improves over the best single model by 13 questions in each condition.

## Result Verification Pass

- Verified all headline pass@k claims against JSON files under `results/`, including:
  - `results/milestone2_gsm8k_base/passk_gsm8k_comparison.json`
  - `results/milestone2_gsm8k_base_4shot/passk_gsm8k_comparison.json`
  - `results/milestone2_gsm8k_instruct_0shot/passk_gsm8k_comparison.json`
  - `results/milestone2_gsm8k_instruct_4shot/passk_gsm8k_comparison.json`
  - `results/milestone2_gsm8k_instruct/passk_gsm8k_comparison.json`
  - `results/milestone2_countdown_base_refresh/passk_countdown_cd4_comparison.json`
  - `results/milestone2_countdown_instruct_refresh/passk_countdown_cd4_comparison.json`
  - `results/passk_countdown_cd4_comparison.json`
  - `results/aime_data_analysis_large_k_comparison.json`
  - `results/milestone2_trip_planning_llada_passk/passk_trip_planning_comparison.json`
  - `results/milestone2_trip_planning_llama_passk/passk_trip_planning_comparison.json`
  - `results/accuracy_trip_planning_comparison.json`
- Recomputed the hyperparameter-sweep examples from the stored GSM8K generation files in `results/GSAI-ML_LLaDA-8B-Base_gsm8k/`, `results/Dream-org_Dream-v0-Base-7B_gsm8k/`, and `results/Qwen_Qwen2.5-7B_gsm8k/`.
- Recomputed per-question Pearson correlations from the generation JSON `extracted_answer` vectors; corrected small GSM8K instruct rounding differences in the appendix and figure script.
- Rechecked parser sensitivity, diversity, and oracle-union claims against the generated CSVs in `thesis/tables/`.
- Corrections made in this verification pass:
  - GSM8K high-k wording now says two, not three, main templated runs reach pass@128 = 1.000.
  - Trip Planning deterministic prose and appendix rows now reflect `accuracy_trip_planning_comparison.json`, including Llama-Instruct flat [ar] at 0.095 and [vllm] at 0.080.
  - GSM8K parser-crash wording no longer claims every diagnostic run is below 0.1%.
  - Countdown qualitative diversity example now states the verified counts: the LLaDA correct chain appears in 6/128 samples and Qwen has 94 distinct first-line candidate chains on that item.

## Remaining Weak Points

- The main text is stronger, but some detailed result tables are still verbose and could be pushed further into a dedicated appendix in a final polishing pass.
- GSM8K remains partly saturated, so it mostly supports prompt-format and parser-sensitivity claims rather than the central ranking claim.
- Trip Planning combines refreshed LLaDA/Llama runs with earlier Dream/Qwen comparison artifacts.
- Hardware metadata is not consistently stored in the JSON artifacts, so runtime comparisons remain backend-dependent.
- Oracle unions are upper bounds only; no deployable ensemble selector is implemented.

## Build Command

From `thesis/`:

```bash
env XDG_CACHE_HOME=/tmp/tectonic-cache TEXMFVAR=/tmp/texmf-var ../.tools/tectonic/tectonic thesis.tex
```

## Build Status

- `thesis.pdf` compiles successfully with the command above.
- Latest verification compile completed without TeX warnings.
