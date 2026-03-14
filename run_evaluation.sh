#!/bin/bash
#
# Unified Evaluation Script for Diffusion vs AR LLM Comparison
#
# Replaces: run_experiments.sh, run_batch_comparison.sh,
#           run_deterministic_comparison.sh, run_instruct_comparison.sh,
#           run_passk_experiments.sh, run_hard_benchmarks.sh
#
# Usage: ./run_evaluation.sh --experiment <type> [options]
#
# Experiment types:
#   accuracy     — Single-sample accuracy (temp=0, n=1)
#   passk        — Pass@k with many samples (temp=0.7, n=128)
#   batch        — Batch size comparison (bs=1 vs bs=8)
#   speed        — Fast vs slow inference path comparison
#   sweep        — Grid search over gen_length/steps/block_length
#
# See ./run_evaluation.sh --help for full option list.
# See docs/EXPERIMENTS.md for detailed documentation.
#

set -uo pipefail

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# =============================================================================
# DEFAULTS
# =============================================================================
EXPERIMENT=""
DATASET="countdown_cd4"
TARGET="all"           # all | llada | dream | qwen | llama
VARIANT="all"          # all | base | instruct
METHOD="fast"          # fast | slow | both
N_SAMPLES=""           # empty = auto per experiment
BATCH_SIZE=""          # empty = auto per experiment
TEMPERATURE=""         # empty = auto per experiment
FEW_SHOT=""            # empty = auto per dataset/variant
NUM_EVALS=""           # empty = auto per dataset
GEN_LENGTH=""          # empty = auto per dataset
STEPS=""               # empty = auto per dataset
BLOCK_LENGTH=""        # empty = auto per dataset
PROMPT_MODE=""         # empty = auto | instruct_flat
LOGITS_EOS_INF=""     # empty = auto (True for LLaDA-Instruct) | true | false
CONFIDENCE_EOS_EOT_INF=""  # empty = auto (True for LLaDA-Instruct) | true | false
RESULTS_DIR="results"

# =============================================================================
# HELP
# =============================================================================
show_help() {
    cat <<'HELP'
Unified Evaluation Script — Diffusion vs AR LLM Comparison

USAGE
  ./run_evaluation.sh --experiment <type> [options]

EXPERIMENT TYPES
  accuracy   Single-sample accuracy evaluation (deterministic).
             Defaults: temp=0, n_samples=1, batch_size=8, method=both
             Compares fast vs slow inference paths for correctness.

  passk      Pass@k evaluation with many samples per question.
             Defaults: temp=0.7, n_samples=128, batch_size=8, method=fast
             Studies how accuracy scales with repeated sampling (k=1..128).

  batch      Batch size comparison (bs=1 vs bs=8).
             Defaults: temp=0, n_samples=1, method=both
             Measures throughput speedup and accuracy consistency.

  speed      Fast vs slow inference speed comparison.
             Defaults: temp=0, n_samples=1, batch_size=8, method=both
             Times fast-dLLM/vLLM vs dLLM/AR-HF for wall-clock comparison.

  sweep      Hyperparameter grid search (gen_length × steps × block_length).
             Defaults: temp=0.7, n_samples=128, method=fast
             Only for diffusion models (LLaDA/Dream).

MODEL SELECTION
  -m, --model        llada | dream | qwen | llama | all  (default: all)
  -v, --variant      base | instruct | all | all3        (default: all)
                     all3 = base + instruct + instruct_flat (3-way comparison)
  --method           fast | slow | both                 (default: per-experiment)

PROMPT MODE
  --prompt_mode      auto | instruct_flat               (default: auto)
                     auto          — base_native for base, instruct_templated for instruct
                     instruct_flat — instruct model + flat completion prompt (ablation)
                     Only meaningful for instruct models. Base models always use base_native.

DATASET
  -d, --dataset      countdown_cd4 | countdown | countdown_cd5 |
                     gsm8k | math | math_beyond | aime | trip_planning |
                     sudoku | counting_letters           (default: countdown_cd4)

GENERATION PARAMETERS (override per-dataset defaults)
  -n, --n_samples    Samples per question
  -B, --batch_size   Batch size for inference
  -t, --temp         Sampling temperature
  -f, --few_shot     Few-shot examples (auto: 8 for countdown/sudoku, 2 trip_planning, 4 else)
  -e, --num_evals    Number of test problems (auto per dataset)
  -g, --gen_length   Max generation tokens
  -s, --steps        Diffusion steps
  -b, --block_length Block length for diffusion

EOS HANDLING (LLaDA-Instruct)
  --logits_eos_inf         true | false     (default: auto — true for LLaDA-Instruct)
                           Suppress EOS token in logits (prevents EOS from being sampled)
  --confidence_eos_eot_inf true | false     (default: auto — true for LLaDA-Instruct)
                           Suppress EOS in confidence scores (defers EOS unmasking)

OUTPUT
  -o, --output_dir   Results directory                   (default: results)

PER-DATASET DEFAULTS
  Dataset            gen_length  steps  block  few_shot  num_evals
  ─────────────────  ──────────  ─────  ─────  ────────  ─────────
  countdown_cd4           32      32     32       8        992
  countdown/cd3/cd5       24      24     32       8        992
  math (MATH500)        1024     512     32       4        200
  aime                  1024     512     32       4         60
  math_beyond           1024     512     32       4        181
  trip_planning          256     256     32       2        200
  sudoku                  24      24     32       8        256
  gsm8k                  256     256     32       4        256

LLADA-INSTRUCT OVERRIDES (auto-applied, override with -b)
  LLaDA-Instruct uses optimized semi-autoregressive block_length following
  official LLaDA EVAL.md (mitigates EOS overflow from SFT padding).
  gen_length and steps stay the same; only block_length changes:
  Dataset            block (default)  block (LLaDA-Instruct)
  ─────────────────  ──────────────   ──────────────────────
  countdown*/sudoku       32                   8
  gsm8k                   32                   8
  trip_planning            32                  16
  math/aime/math_beyond   32                  64

EXAMPLES
  # Accuracy: all models on countdown_cd4, fast vs slow
  ./run_evaluation.sh --experiment accuracy

  # Pass@k: base + instruct on countdown_cd4
  ./run_evaluation.sh --experiment passk -d countdown_cd4

  # Pass@k: only instruct models on MATH500
  ./run_evaluation.sh --experiment passk -d math -v instruct

  # Batch comparison: Dream only
  ./run_evaluation.sh --experiment batch -m dream

  # Speed comparison: LLaDA base only
  ./run_evaluation.sh --experiment speed -m llada -v base

  # Hyperparameter sweep: LLaDA on GSM8K
  ./run_evaluation.sh --experiment sweep -m llada -d gsm8k

  # Hard benchmarks: all models, all hard datasets
  ./run_evaluation.sh --experiment passk -d aime
  ./run_evaluation.sh --experiment passk -d math_beyond
  ./run_evaluation.sh --experiment passk -d trip_planning

  # 3-way prompt comparison: base + instruct_templated + instruct_flat
  ./run_evaluation.sh --experiment passk -d countdown_cd4 -v all3

  # Instruct-flat ablation only (instruct checkpoint with base prompt)
  ./run_evaluation.sh --experiment passk -d countdown_cd4 -v instruct --prompt_mode instruct_flat
HELP
}

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-E)   EXPERIMENT="$2";    shift 2 ;;
        -d|--dataset)      DATASET="$2";       shift 2 ;;
        -m|--model)        TARGET="$2";        shift 2 ;;
        -v|--variant)      VARIANT="$2";       shift 2 ;;
        --method)          METHOD="$2";        shift 2 ;;
        -n|--n_samples)    N_SAMPLES="$2";     shift 2 ;;
        -B|--batch_size)   BATCH_SIZE="$2";    shift 2 ;;
        -t|--temp)         TEMPERATURE="$2";   shift 2 ;;
        -f|--few_shot)     FEW_SHOT="$2";      shift 2 ;;
        -e|--num_evals)    NUM_EVALS="$2";     shift 2 ;;
        -g|--gen_length)   GEN_LENGTH="$2";    shift 2 ;;
        -s|--steps)        STEPS="$2";         shift 2 ;;
        -b|--block_length) BLOCK_LENGTH="$2";  shift 2 ;;
        --prompt_mode)     PROMPT_MODE="$2";   shift 2 ;;
        --logits_eos_inf)  LOGITS_EOS_INF="$2";   shift 2 ;;
        --confidence_eos_eot_inf) CONFIDENCE_EOS_EOT_INF="$2"; shift 2 ;;
        -o|--output_dir)   RESULTS_DIR="$2";   shift 2 ;;
        -h|--help)         show_help; exit 0 ;;
        *) echo "Unknown argument: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "ERROR: --experiment is required."
    echo "Valid types: accuracy, passk, batch, speed, sweep"
    echo "Run with --help for full usage."
    exit 1
fi

# =============================================================================
# APPLY EXPERIMENT-SPECIFIC DEFAULTS (only for unset values)
# =============================================================================
case $EXPERIMENT in
    accuracy)
        : "${N_SAMPLES:=1}"
        : "${BATCH_SIZE:=8}"
        : "${TEMPERATURE:=0}"
        : "${METHOD:=both}"
        ;;
    passk)
        : "${N_SAMPLES:=128}"
        : "${BATCH_SIZE:=8}"
        : "${TEMPERATURE:=0.7}"
        : "${METHOD:=fast}"
        ;;
    batch)
        : "${N_SAMPLES:=1}"
        : "${BATCH_SIZE:=8}"
        : "${TEMPERATURE:=0}"
        : "${METHOD:=both}"
        ;;
    speed)
        : "${N_SAMPLES:=1}"
        : "${BATCH_SIZE:=8}"
        : "${TEMPERATURE:=0}"
        : "${METHOD:=both}"
        ;;
    sweep)
        : "${N_SAMPLES:=128}"
        : "${BATCH_SIZE:=1}"
        : "${TEMPERATURE:=0.7}"
        : "${METHOD:=fast}"
        ;;
    *)
        echo "ERROR: Unknown experiment type '$EXPERIMENT'."
        echo "Valid types: accuracy, passk, batch, speed, sweep"
        exit 1
        ;;
esac

mkdir -p "$RESULTS_DIR"

LOG_FILE="${EXPERIMENT}_${DATASET}_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

gpu_cleanup() {
    python3 -c "import torch,gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null
    sleep 1
}

# =============================================================================
# LOCK FILE
# =============================================================================
LOCK_FILE="/tmp/run_eval_${EXPERIMENT}_${DATASET}.lock"
cleanup() {
    rm -f "$LOCK_FILE"
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ -f "$LOCK_FILE" ]; then
    OTHER_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OTHER_PID" ] && kill -0 "$OTHER_PID" 2>/dev/null; then
        echo "ERROR: Another instance is running (PID: $OTHER_PID). Remove $LOCK_FILE if stale."
        exit 1
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
declare -A BASE_MODELS
BASE_MODELS=(
    ["llada"]="GSAI-ML/LLaDA-8B-Base"
    ["dream"]="Dream-org/Dream-v0-Base-7B"
    ["qwen"]="Qwen/Qwen2.5-7B"
    ["llama"]="meta-llama/Llama-3.1-8B"
)

declare -A INST_MODELS
INST_MODELS=(
    ["llada"]="GSAI-ML/LLaDA-8B-Instruct"
    ["dream"]="Dream-org/Dream-v0-Instruct-7B"
    ["qwen"]="Qwen/Qwen2.5-7B-Instruct"
    ["llama"]="meta-llama/Llama-3.1-8B-Instruct"
)

is_diffusion_model() {
    local family=$1
    [[ "$family" == "llada" || "$family" == "dream" ]]
}

# =============================================================================
# PER-DATASET GENERATION PARAMETERS
#
#  Dataset            gl    st    bl   fs   ne    Source / Rationale
#  ─────────────────  ────  ────  ───  ──  ────  ───────────────────────────
#  countdown_cd4       32    32   32    8   992   Dream official eval
#  countdown/cd3/cd5   24    24   32    8   992   Dream official eval
#  math (MATH500)    1024   512   32    4   200   99.8% solutions fit @1024
#  aime              1024   512   32    4    60   all 60 problems
#  math_beyond       1024   512   32    4   181   all 181 problems
#  trip_planning      256   256   32    2   200   Dream official eval
#  sudoku              24    24   32    8   256   Dream official eval
#  gsm8k              256   256   32    4   256   standard
# =============================================================================

_is_llada_instruct() {
    local model_name=$1
    echo "$model_name" | grep -qi "llada" && echo "$model_name" | grep -qi "instruct"
}

_get_gen_length() {
    local dataset=$1
    local model_name=${2:-""}
    if [[ -n "$GEN_LENGTH" ]]; then echo "$GEN_LENGTH"; return; fi
    case $dataset in
        countdown_cd4)                echo 32 ;;
        countdown*|sudoku)            echo 24 ;;
        math|aime|math_beyond)        echo 1024 ;;
        trip_planning)                echo 256 ;;
        gsm8k|counting_letters)       echo 256 ;;
        *)                            echo 256 ;;
    esac
}

_get_steps() {
    local dataset=$1
    local model_name=${2:-""}
    if [[ -n "$STEPS" ]]; then echo "$STEPS"; return; fi
    case $dataset in
        countdown_cd4)                echo 32 ;;
        countdown*|sudoku)            echo 24 ;;
        math|aime|math_beyond)        echo 512 ;;
        trip_planning)                echo 256 ;;
        gsm8k|counting_letters)       echo 256 ;;
        *)                            echo 256 ;;
    esac
}

_get_block_length() {
    local dataset=$1
    local model_name=${2:-""}
    if [[ -n "$BLOCK_LENGTH" ]]; then echo "$BLOCK_LENGTH"; return; fi

    # LLaDA-Instruct: use semi-autoregressive (smaller block_length) following
    # official LLaDA EVAL.md — dramatically improves instruct model performance.
    # GSM8K: 68.8% → 78.9% with block_length=8
    # MATH:  29.6% → 42.7% with block_length=64
    if _is_llada_instruct "$model_name"; then
        case $dataset in
            countdown*|sudoku|gsm8k|counting_letters) echo 8 ;;
            trip_planning)                             echo 16 ;;
            math|aime|math_beyond)                     echo 64 ;;
            *)                                         echo 16 ;;
        esac
        return
    fi

    echo 32
}

_get_num_evals() {
    local dataset=$1
    if [[ -n "$NUM_EVALS" ]]; then echo "$NUM_EVALS"; return; fi
    case $dataset in
        countdown*)       echo 992 ;;
        math)             echo 200 ;;
        aime)             echo 60 ;;
        math_beyond)      echo 181 ;;
        trip_planning)    echo 200 ;;
        sudoku)           echo 256 ;;
        gsm8k)            echo 256 ;;
        counting_letters) echo 256 ;;
        *)                echo 256 ;;
    esac
}

_get_default_few_shot() {
    local dataset=$1
    case $dataset in
        countdown*|sudoku) echo 8 ;;
        trip_planning)     echo 2 ;;
        *)                 echo 4 ;;
    esac
}

_get_few_shot() {
    local dataset=$1
    local model_name=$2
    if [[ -n "$FEW_SHOT" ]]; then echo "$FEW_SHOT"; return; fi

    # Same few-shot for base and instruct (apples-to-apples comparison).
    # The only difference is the chat-template wrapping for instruct models.
    _get_default_few_shot "$dataset"
}

_get_k_values() {
    local n=$1
    if [ "$n" -eq 1 ]; then
        echo "[1]"
    elif [ "$n" -le 16 ]; then
        echo "[1, 2, 4, 8, 16]"
    elif [ "$n" -le 32 ]; then
        echo "[1, 2, 4, 8, 16, 32]"
    elif [ "$n" -le 64 ]; then
        echo "[1, 2, 4, 8, 16, 32, 64]"
    else
        echo "[1, 2, 4, 8, 16, 32, 64, 128]"
    fi
}

# =============================================================================
# FILENAME HELPERS
# =============================================================================
_data_tag() {
    local dataset=$1
    if [[ "$dataset" == "gsm8k" ]]; then echo ""; else echo "_${dataset}"; fi
}

_pm_tag() {
    local pm=$1
    if [[ "$pm" == "instruct_flat" ]]; then echo "_flat"; else echo ""; fi
}

fast_dllm_fn() {
    local model=$1 dataset=$2 fs=$3 bs=${4:-$BATCH_SIZE} pm=${5:-$PROMPT_MODE}
    local m=$(echo "$model" | tr '/' '_')
    local dt=$(_data_tag "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model")
    local st=$(_get_steps "$dataset" "$model")
    local bl=$(_get_block_length "$dataset" "$model")
    local ne=$(_get_num_evals "$dataset")
    local pt=$(_pm_tag "$pm")
    echo "${RESULTS_DIR}/${m}${dt}_${gl}_${st}_${bl}_${bs}_${TEMPERATURE}_${fs}_${ne}_${N_SAMPLES}_generations${pt}_fast_dllm.json"
}

dllm_fn() {
    local model=$1 dataset=$2 fs=$3 bs=${4:-$BATCH_SIZE} pm=${5:-$PROMPT_MODE}
    local m=$(echo "$model" | tr '/' '_')
    local dt=$(_data_tag "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model")
    local st=$(_get_steps "$dataset" "$model")
    local bl=$(_get_block_length "$dataset" "$model")
    local ne=$(_get_num_evals "$dataset")
    local pt=$(_pm_tag "$pm")
    echo "${RESULTS_DIR}/${m}${dt}_${gl}_${st}_${bl}_${bs}_${TEMPERATURE}_${fs}_${ne}_${N_SAMPLES}_generations${pt}_dllm.json"
}

vllm_fn() {
    local model=$1 dataset=$2 fs=$3 bs=${4:-$BATCH_SIZE} pm=${5:-$PROMPT_MODE}
    local m=$(echo "$model" | tr '/' '_')
    local dt=$(_data_tag "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model")
    local ne=$(_get_num_evals "$dataset")
    local pt=$(_pm_tag "$pm")
    echo "${RESULTS_DIR}/${m}${dt}_${gl}_${bs}_${TEMPERATURE}_${fs}_${ne}_${N_SAMPLES}_generations${pt}_vllm.json"
}

ar_fn() {
    local model=$1 dataset=$2 fs=$3 bs=${4:-$BATCH_SIZE} pm=${5:-$PROMPT_MODE}
    local m=$(echo "$model" | tr '/' '_')
    local dt=$(_data_tag "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model")
    local ne=$(_get_num_evals "$dataset")
    local pt=$(_pm_tag "$pm")
    echo "${RESULTS_DIR}/${m}${dt}_${gl}_${bs}_${TEMPERATURE}_${fs}_${ne}_${N_SAMPLES}_generations${pt}_ar.json"
}

# =============================================================================
# COMPLETION CHECK
# =============================================================================
check_complete() {
    local output_file=$1
    local expected_evals=$2
    if [ -f "$output_file" ]; then
        local completed
        completed=$(python3 -c "
import json
try:
    with open('$output_file') as f:
        d = json.load(f)
    gens = d.get('generations', [])
    if gens:
        ea = gens[0].get('extracted_answer')
        expected_n = ${N_SAMPLES}
        if isinstance(ea, list) and len(ea) >= expected_n:
            print(len(gens))
        elif not isinstance(ea, list) and expected_n == 1:
            print(len(gens))
        else:
            print(0)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
        if [ "$completed" -ge "$expected_evals" ]; then
            return 0
        fi
    fi
    return 1
}

# =============================================================================
# RUNNERS
# =============================================================================

run_fast_dllm() {
    local model_name=$1 dataset=$2
    local bs=${3:-$BATCH_SIZE}
    local pm=${4:-$PROMPT_MODE}
    local fs=$(_get_few_shot "$dataset" "$model_name")
    local ne=$(_get_num_evals "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model_name")
    local st=$(_get_steps "$dataset" "$model_name")
    local bl=$(_get_block_length "$dataset" "$model_name")
    local output_file
    output_file=$(fast_dllm_fn "$model_name" "$dataset" "$fs" "$bs" "$pm")

    # Resolve prompt_mode Python value: empty/auto → None
    local pm_py="None"
    [[ -n "$pm" && "$pm" != "auto" ]] && pm_py="'${pm}'"

    if check_complete "$output_file" "$ne"; then
        log "SKIP  [fast-dLLM] $model_name on $dataset (bs=$bs, pm=${pm:-auto}) — already complete"
        return 0
    fi

    local alg="entropy"
    local dual_cache="False"
    local cache_refresh_steps=0

    # Resolve EOS handling flags (auto-enable for LLaDA-Instruct)
    local eos_logits="False"
    local eos_confidence="False"
    if [[ -n "$LOGITS_EOS_INF" ]]; then
        [[ "$LOGITS_EOS_INF" == "true" ]] && eos_logits="True"
    elif _is_llada_instruct "$model_name"; then
        eos_logits="True"
    fi
    if [[ -n "$CONFIDENCE_EOS_EOT_INF" ]]; then
        [[ "$CONFIDENCE_EOS_EOT_INF" == "true" ]] && eos_confidence="True"
    elif _is_llada_instruct "$model_name"; then
        eos_confidence="True"
    fi

    if echo "$model_name" | grep -qi "dream"; then
        alg="confidence_threshold"
        dual_cache="True"
        cache_refresh_steps=4
    elif echo "$model_name" | grep -qi "llada"; then
        alg="entropy"
        dual_cache="True"
        cache_refresh_steps=0
    fi

    log "START [fast-dLLM] $model_name on $dataset (n=$N_SAMPLES, temp=$TEMPERATURE, bs=$bs, fs=$fs, gl=$gl, st=$st, bl=$bl, evals=$ne, pm=${pm:-auto}, eos_logits=$eos_logits, eos_conf=$eos_confidence)"
    gpu_cleanup

    python3 -u -c "
import sys; sys.path.insert(0, '.')
from inference_and_generation import evaluate_fast_dllm
from evaluate_pass_k import compute_metrics

filename = evaluate_fast_dllm(
    diffusion_model_name='${model_name}',
    data='${dataset}',
    num_evals_to_use=${ne},
    few_shot=${fs},
    batch_size=${bs},
    gen_length=${gl},
    temperature=${TEMPERATURE},
    steps=${st},
    block_length=${bl},
    remasking='low_confidence',
    use_cache=True,
    factor=None,
    alg='${alg}',
    alg_temp=0.0,
    top_p=1.0,
    top_k=None,
    dual_cache=${dual_cache},
    threshold=0.9,
    cache_refresh_steps=${cache_refresh_steps},
    n_samples=${N_SAMPLES},
    output_dir='${RESULTS_DIR}',
    prompt_mode=${pm_py},
    logits_eos_inf=${eos_logits},
    confidence_eos_eot_inf=${eos_confidence},
)

compute_metrics(results_file=filename, samples_per_problem=${N_SAMPLES}, k_values=$(_get_k_values "$N_SAMPLES"))
" 2>&1 | tee -a "$LOG_FILE"

    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && log "DONE  [fast-dLLM] $model_name on $dataset (pm=${pm:-auto})" \
                   || log "FAIL  [fast-dLLM] $model_name on $dataset (pm=${pm:-auto}, exit $rc)"
    gpu_cleanup
    return $rc
}

run_dllm() {
    local model_name=$1 dataset=$2
    local bs=${3:-$BATCH_SIZE}
    local pm=${4:-$PROMPT_MODE}
    local fs=$(_get_few_shot "$dataset" "$model_name")
    local ne=$(_get_num_evals "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model_name")
    local st=$(_get_steps "$dataset" "$model_name")
    local bl=$(_get_block_length "$dataset" "$model_name")
    local output_file
    output_file=$(dllm_fn "$model_name" "$dataset" "$fs" "$bs" "$pm")

    local pm_py="None"
    [[ -n "$pm" && "$pm" != "auto" ]] && pm_py="'${pm}'"

    if check_complete "$output_file" "$ne"; then
        log "SKIP  [dLLM] $model_name on $dataset (bs=$bs, pm=${pm:-auto}) — already complete"
        return 0
    fi

    log "START [dLLM] $model_name on $dataset (n=$N_SAMPLES, temp=$TEMPERATURE, bs=$bs, fs=$fs, gl=$gl, st=$st, bl=$bl, evals=$ne, pm=${pm:-auto})"
    gpu_cleanup

    python3 -u -c "
import sys; sys.path.insert(0, '.')
from inference_and_generation import evaluate_dllm
from evaluate_pass_k import compute_metrics

filename = evaluate_dllm(
    diffusion_model_name='${model_name}',
    data='${dataset}',
    num_evals_to_use=${ne},
    few_shot=${fs},
    batch_size=${bs},
    gen_length=${gl},
    steps=${st},
    temperature=${TEMPERATURE},
    block_length=${bl},
    cfg_scale=0.0,
    remasking='low_confidence',
    alg='entropy',
    alg_temp=0.0,
    top_p=1.0,
    top_k=None,
    n_samples=${N_SAMPLES},
    output_dir='${RESULTS_DIR}',
    prompt_mode=${pm_py},
)

compute_metrics(results_file=filename, samples_per_problem=${N_SAMPLES}, k_values=$(_get_k_values "$N_SAMPLES"))
" 2>&1 | tee -a "$LOG_FILE"

    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && log "DONE  [dLLM] $model_name on $dataset (pm=${pm:-auto})" \
                   || log "FAIL  [dLLM] $model_name on $dataset (pm=${pm:-auto}, exit $rc)"
    gpu_cleanup
    return $rc
}

run_vllm() {
    local model_name=$1 dataset=$2
    local bs=${3:-$BATCH_SIZE}
    local pm=${4:-$PROMPT_MODE}
    local fs=$(_get_few_shot "$dataset" "$model_name")
    local ne=$(_get_num_evals "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model_name")
    local output_file
    output_file=$(vllm_fn "$model_name" "$dataset" "$fs" "$bs" "$pm")

    local pm_py="None"
    [[ -n "$pm" && "$pm" != "auto" ]] && pm_py="'${pm}'"

    if check_complete "$output_file" "$ne"; then
        log "SKIP  [vLLM] $model_name on $dataset (bs=$bs, pm=${pm:-auto}) — already complete"
        return 0
    fi

    log "START [vLLM] $model_name on $dataset (n=$N_SAMPLES, temp=$TEMPERATURE, bs=$bs, fs=$fs, gl=$gl, evals=$ne, pm=${pm:-auto})"
    gpu_cleanup

    python3 -u -c "
import sys; sys.path.insert(0, '.')
from inference_and_generation import evaluate_vllm_model
from evaluate_pass_k import compute_metrics

filename = evaluate_vllm_model(
    model_name='${model_name}',
    data='${dataset}',
    num_evals_to_use=${ne},
    few_shot=${fs},
    batch_size=${bs},
    gen_length=${gl},
    temperature=${TEMPERATURE},
    n_samples=${N_SAMPLES},
    output_dir='${RESULTS_DIR}',
    prompt_mode=${pm_py},
)

compute_metrics(results_file=filename, samples_per_problem=${N_SAMPLES}, k_values=$(_get_k_values "$N_SAMPLES"))
" 2>&1 | tee -a "$LOG_FILE"

    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && log "DONE  [vLLM] $model_name on $dataset (pm=${pm:-auto})" \
                   || log "FAIL  [vLLM] $model_name on $dataset (pm=${pm:-auto}, exit $rc)"
    gpu_cleanup
    return $rc
}

run_ar() {
    local model_name=$1 dataset=$2
    local bs=${3:-$BATCH_SIZE}
    local pm=${4:-$PROMPT_MODE}
    local fs=$(_get_few_shot "$dataset" "$model_name")
    local ne=$(_get_num_evals "$dataset")
    local gl=$(_get_gen_length "$dataset" "$model_name")
    local output_file
    output_file=$(ar_fn "$model_name" "$dataset" "$fs" "$bs" "$pm")

    local pm_py="None"
    [[ -n "$pm" && "$pm" != "auto" ]] && pm_py="'${pm}'"

    if check_complete "$output_file" "$ne"; then
        log "SKIP  [AR] $model_name on $dataset (bs=$bs, pm=${pm:-auto}) — already complete"
        return 0
    fi

    log "START [AR-HF] $model_name on $dataset (n=$N_SAMPLES, temp=$TEMPERATURE, bs=$bs, fs=$fs, gl=$gl, evals=$ne, pm=${pm:-auto})"
    gpu_cleanup

    python3 -u -c "
import sys; sys.path.insert(0, '.')
from inference_and_generation import evaluate_auto_regressive_model
from evaluate_pass_k import compute_metrics

filename = evaluate_auto_regressive_model(
    model_name='${model_name}',
    data='${dataset}',
    num_evals_to_use=${ne},
    few_shot=${fs},
    batch_size=${bs},
    gen_length=${gl},
    temperature=${TEMPERATURE},
    top_p=1.0,
    n_samples=${N_SAMPLES},
    output_dir='${RESULTS_DIR}',
    prompt_mode=${pm_py},
)

compute_metrics(results_file=filename, samples_per_problem=${N_SAMPLES}, k_values=$(_get_k_values "$N_SAMPLES"))
" 2>&1 | tee -a "$LOG_FILE"

    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && log "DONE  [AR-HF] $model_name on $dataset (pm=${pm:-auto})" \
                   || log "FAIL  [AR-HF] $model_name on $dataset (pm=${pm:-auto}, exit $rc)"
    gpu_cleanup
    return $rc
}

# =============================================================================
# DISPATCH: run a single model on a single dataset with the chosen method(s)
# =============================================================================
run_one() {
    local family=$1 model_name=$2 dataset=$3 bs=${4:-$BATCH_SIZE} pm=${5:-$PROMPT_MODE}

    if is_diffusion_model "$family"; then
        if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
            run_fast_dllm "$model_name" "$dataset" "$bs" "$pm" || ((FAILED++))
        fi
        if [[ "$METHOD" == "slow" || "$METHOD" == "both" ]]; then
            run_dllm "$model_name" "$dataset" "$bs" "$pm" || ((FAILED++))
        fi
    else
        if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
            run_vllm "$model_name" "$dataset" "$bs" "$pm" || ((FAILED++))
        fi
        if [[ "$METHOD" == "slow" || "$METHOD" == "both" ]]; then
            run_ar "$model_name" "$dataset" "$bs" "$pm" || ((FAILED++))
        fi
    fi
}

# =============================================================================
# RESOLVE FAMILIES AND VARIANTS
# =============================================================================
FAMILIES=()
case $TARGET in
    all)   FAMILIES=(llada dream qwen llama) ;;
    llada) FAMILIES=(llada) ;;
    dream) FAMILIES=(dream) ;;
    qwen)  FAMILIES=(qwen) ;;
    llama) FAMILIES=(llama) ;;
    *)     echo "ERROR: Unknown model '$TARGET' (use llada, dream, qwen, llama, or all)"; exit 1 ;;
esac

get_models_for_variant() {
    local family=$1
    case $VARIANT in
        base)          echo "${BASE_MODELS[$family]}" ;;
        instruct)      echo "${INST_MODELS[$family]}" ;;
        all|all3)      echo "${BASE_MODELS[$family]} ${INST_MODELS[$family]}" ;;
    esac
}

# =============================================================================
# EXPERIMENT: accuracy / passk / speed
# =============================================================================
run_standard_experiment() {
    for family in "${FAMILIES[@]}"; do
        for model_name in $(get_models_for_variant "$family"); do
            run_one "$family" "$model_name" "$DATASET" "$BATCH_SIZE" "$PROMPT_MODE"
            ((RUN++))
            log "  Progress: $RUN runs completed"
        done
        # all3: also run instruct models with instruct_flat prompt
        if [[ "$VARIANT" == "all3" ]]; then
            local inst_model="${INST_MODELS[$family]}"
            log "--- instruct_flat ablation: $inst_model ---"
            run_one "$family" "$inst_model" "$DATASET" "$BATCH_SIZE" "instruct_flat"
            ((RUN++))
            log "  Progress: $RUN runs completed"
        fi
    done
}

# =============================================================================
# EXPERIMENT: batch (runs bs=1 then bs=8)
# =============================================================================
run_batch_experiment() {
    for bs in 1 8; do
        log ""
        log "############################################################"
        log "# BATCH SIZE = $bs"
        log "############################################################"
        for family in "${FAMILIES[@]}"; do
            for model_name in $(get_models_for_variant "$family"); do
                run_one "$family" "$model_name" "$DATASET" "$bs"
                ((RUN++))
                log "  Progress: $RUN runs completed"
            done
        done
    done
}

# =============================================================================
# EXPERIMENT: sweep (gen_length × steps × block_length grid)
# =============================================================================
run_sweep_experiment() {
    if [[ "$TARGET" == "qwen" ]]; then
        echo "ERROR: sweep experiment is only for diffusion models (llada, dream)"
        exit 1
    fi

    local sweep_families=()
    for f in "${FAMILIES[@]}"; do
        if is_diffusion_model "$f"; then
            sweep_families+=("$f")
        fi
    done

    if [ ${#sweep_families[@]} -eq 0 ]; then
        echo "ERROR: No diffusion models selected for sweep."
        exit 1
    fi

    # Build experiment grid
    local EXPERIMENTS=()
    for gl in 128 256; do
        if [ "$gl" -eq 128 ]; then
            local steps_list=(32 64 128)
            local block_list=(32 64 128)
        else
            local steps_list=(32 64 128 256)
            local block_list=(32 64 128 256)
        fi
        for st in "${steps_list[@]}"; do
            for bl in "${block_list[@]}"; do
                EXPERIMENTS+=("${gl}:${st}:${bl}")
            done
        done
    done

    local total_exps=${#EXPERIMENTS[@]}
    log "Sweep: ${#sweep_families[@]} families × $total_exps configs = $((${#sweep_families[@]} * total_exps)) runs"

    local ne=$(_get_num_evals "$DATASET")

    for family in "${sweep_families[@]}"; do
        local model_name="${BASE_MODELS[$family]}"
        log ""
        log "=== SWEEP: $model_name ==="

        local exp_num=0
        for exp in "${EXPERIMENTS[@]}"; do
            ((exp_num++))
            IFS=':' read -r gl st bl <<< "$exp"

            local m_clean=$(echo "$model_name" | tr '/' '_')
            local dt=$(_data_tag "$DATASET")
            local fs=$(_get_few_shot "$DATASET" "$model_name")
            local output_file="${RESULTS_DIR}/${m_clean}${dt}_${gl}_${st}_${bl}_${BATCH_SIZE}_${TEMPERATURE}_${fs}_${ne}_${N_SAMPLES}_generations_fast_dllm.json"

            if check_complete "$output_file" "$ne"; then
                log "[$exp_num/$total_exps] SKIP: gl=$gl st=$st bl=$bl — complete"
                continue
            fi

            log "[$exp_num/$total_exps] START: gl=$gl st=$st bl=$bl"
            gpu_cleanup

            local alg="entropy"
            local dual_cache="True"
            local cache_refresh_steps=0
            if echo "$model_name" | grep -qi "dream"; then
                alg="confidence_threshold"
                cache_refresh_steps=4
            fi

            python3 -u -c "
import sys; sys.path.insert(0, '.')
from inference_and_generation import evaluate_fast_dllm
from evaluate_pass_k import compute_metrics

filename = evaluate_fast_dllm(
    diffusion_model_name='${model_name}',
    data='${DATASET}',
    num_evals_to_use=${ne},
    few_shot=${fs},
    batch_size=${BATCH_SIZE},
    gen_length=${gl},
    temperature=${TEMPERATURE},
    steps=${st},
    block_length=${bl},
    remasking='low_confidence',
    use_cache=True,
    factor=None,
    alg='${alg}',
    alg_temp=0.0,
    top_p=1.0,
    top_k=None,
    dual_cache=${dual_cache},
    threshold=0.9,
    cache_refresh_steps=${cache_refresh_steps},
    n_samples=${N_SAMPLES},
    output_dir='${RESULTS_DIR}',
    prompt_mode=None,
)
compute_metrics(results_file=filename, samples_per_problem=${N_SAMPLES}, k_values=$(_get_k_values "$N_SAMPLES"))
" 2>&1 | tee -a "$LOG_FILE"

            local rc=${PIPESTATUS[0]}
            [ $rc -eq 0 ] && log "[$exp_num/$total_exps] DONE: gl=$gl st=$st bl=$bl" \
                           || { log "[$exp_num/$total_exps] FAIL: gl=$gl st=$st bl=$bl (exit $rc)"; ((FAILED++)); }
            gpu_cleanup
        done
    done
}

# =============================================================================
# COMPARISON TABLE
# =============================================================================
generate_comparison_table() {
    log ""
    log "Generating comparison table..."

    # Build the file list in bash using the same filename helpers
    # that were used to create the results, then pass to Python.
    local file_list_json="{"
    local first=true

    _add_entry() {
        local label=$1 filepath=$2
        if [[ -n "$filepath" ]]; then
            $first || file_list_json+=","
            first=false
            file_list_json+="\"${label}\":\"${filepath}\""
        fi
    }

    for family in "${FAMILIES[@]}"; do
        local models_to_check=()
        case $VARIANT in
            base)     models_to_check=("${BASE_MODELS[$family]}") ;;
            instruct) models_to_check=("${INST_MODELS[$family]}") ;;
            all|all3) models_to_check=("${BASE_MODELS[$family]}" "${INST_MODELS[$family]}") ;;
        esac

        for model_name in "${models_to_check[@]}"; do
            local batch_sizes=("$BATCH_SIZE")
            [[ "$EXPERIMENT" == "batch" ]] && batch_sizes=(1 8)

            for bs in "${batch_sizes[@]}"; do
                local bs_label=""
                [[ "$EXPERIMENT" == "batch" ]] && bs_label=" (bs=${bs})"
                local short_name
                short_name=$(echo "$model_name" | sed 's|.*/||')
                local label="${short_name}${bs_label}"
                local fs=$(_get_few_shot "$DATASET" "$model_name")

                local filepath=""
                if is_diffusion_model "$family"; then
                    if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
                        filepath=$(fast_dllm_fn "$model_name" "$DATASET" "$fs" "$bs" "$PROMPT_MODE")
                    fi
                else
                    if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
                        filepath=$(vllm_fn "$model_name" "$DATASET" "$fs" "$bs" "$PROMPT_MODE")
                    fi
                fi

                _add_entry "$label" "$filepath"
            done
        done

        # all3: add instruct_flat entries
        if [[ "$VARIANT" == "all3" ]]; then
            local inst_model="${INST_MODELS[$family]}"
            local short_name
            short_name=$(echo "$inst_model" | sed 's|.*/||')
            local label="${short_name} [flat]"
            local fs=$(_get_few_shot "$DATASET" "$inst_model")
            local filepath=""

            if is_diffusion_model "$family"; then
                if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
                    filepath=$(fast_dllm_fn "$inst_model" "$DATASET" "$fs" "$BATCH_SIZE" "instruct_flat")
                fi
            else
                if [[ "$METHOD" == "fast" || "$METHOD" == "both" ]]; then
                    filepath=$(vllm_fn "$inst_model" "$DATASET" "$fs" "$BATCH_SIZE" "instruct_flat")
                fi
            fi

            _add_entry "$label" "$filepath"
        fi
    done
    file_list_json+="}"

    python3 -u -c "
import sys, json, os
sys.path.insert(0, '.')
from metrics.pass_k import pass_at_k, compute_pass_at_k
from evaluate_pass_k import prepare_pass_k_data, load_generation_results

n_samples = ${N_SAMPLES}
k_values = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= n_samples]
dataset = '${DATASET}'
experiment = '${EXPERIMENT}'

models = json.loads('${file_list_json}')

print()
header = f\"{'Model':<35}\" + ''.join(f'pass@{k:>3}  ' for k in k_values)
print(header)
print('=' * len(header))

all_results = {}
for label, filepath in models.items():
    if not os.path.exists(filepath):
        print(f'{label:<35} (file not found: {os.path.basename(filepath)})')
        continue
    try:
        gens = load_generation_results(filepath)
        pass_k_data = prepare_pass_k_data(gens)
        scores = compute_pass_at_k(pass_k_data, k_values)
        row = f'{label:<35}'
        model_scores = {}
        for k in k_values:
            if k in scores:
                row += f'{scores[k]*100:>6.2f}%  '
                model_scores[k] = round(scores[k], 6)
            else:
                row += f'{\"N/A\":>7}  '
        print(row)
        all_results[label] = model_scores
    except Exception as e:
        print(f'{label:<35} ERROR: {e}')
print()

output = {
    'experiment': experiment,
    'dataset': dataset,
    'n_samples': n_samples,
    'temperature': ${TEMPERATURE},
    'k_values': k_values,
    'results': all_results,
}
outfile = f'${RESULTS_DIR}/{experiment}_{dataset}_comparison.json'
with open(outfile, 'w') as f:
    json.dump(output, f, indent=2)
print(f'Saved to: {outfile}')
" 2>&1 | tee -a "$LOG_FILE"
}

# =============================================================================
# MAIN
# =============================================================================
FAILED=0
RUN=0

log "================================================================"
log "EVALUATION — experiment=$EXPERIMENT"
log "================================================================"
log "  Dataset:      $DATASET"
log "  Models:       ${FAMILIES[*]}"
log "  Variant:      $VARIANT"
log "  Method:       $METHOD"
log "  N_Samples:    $N_SAMPLES"
log "  Batch Size:   $BATCH_SIZE"
log "  Temperature:  $TEMPERATURE"
log "  Few-shot:     ${FEW_SHOT:-auto}"
log "  Prompt mode:  ${PROMPT_MODE:-auto}"
log "  Gen params (base defaults, LLaDA-Instruct may auto-override):"
log "    gen_length:   $(_get_gen_length "$DATASET")"
log "    steps:        $(_get_steps "$DATASET")"
log "    block_length: $(_get_block_length "$DATASET")"
log "    num_evals:    $(_get_num_evals "$DATASET")"
log "  EOS handling:  logits_eos_inf=${LOGITS_EOS_INF:-auto}, confidence_eos_eot_inf=${CONFIDENCE_EOS_EOT_INF:-auto}"
log "  Log file:     $LOG_FILE"
log "================================================================"

case $EXPERIMENT in
    accuracy|passk|speed)
        run_standard_experiment
        ;;
    batch)
        run_batch_experiment
        ;;
    sweep)
        run_sweep_experiment
        ;;
esac

generate_comparison_table

log ""
log "================================================================"
log "COMPLETE — experiment=$EXPERIMENT, dataset=$DATASET, failed=$FAILED"
log "================================================================"

[ $FAILED -eq 0 ] && exit 0 || exit 1
