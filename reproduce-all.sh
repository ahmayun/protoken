#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally hardcoded to reproduce exactly the figures
# needed by the paper (see 2601.19672v2.pdf):
# - Figure 2 & 3 (--rq1): all 4 models x all 4 datasets
# - Figure 4 (--rq2): all 4 models x all 4 datasets
# - Figure 5 (--rq3): coding dataset, all 4 models
# - Figures 6 & 7 (--rq4): coding dataset, gemma and qwen only

RESULTS_ROOT="results"
PASSTHRU_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXTRA_ARGS...]"
            echo ""
            echo "Runs reproduce.sh only for the paper-needed model/dataset combos."
            echo ""
            echo "Options:"
            echo "  --results-root DIR  Root directory for per-combo outputs (default: results)"
            echo ""
            echo "Extra args:"
            echo "  Any other args are forwarded to reproduce.sh (e.g., --cache DIR, --rounds N)."
            exit 0
            ;;
        *)
            PASSTHRU_ARGS+=("$1")
            shift 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Figure 2 & 3: Main results (RQ1) — all models x all datasets
# ------------------------------------------------------------------------------
MODELS=("gemma" "smollm" "llama" "qwen")
DATASETS=("medical" "finance" "math" "coding")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        combo_dir="${RESULTS_ROOT}/${model}-${dataset}"
        echo "=================================================="
        echo "Figure 2&3 (RQ1): model=${model} dataset=${dataset}"
        echo "Output: ${combo_dir}"
        echo "=================================================="
        ./reproduce.sh \
            "${PASSTHRU_ARGS[@]}" \
            --model "${model}" \
            --dataset "${dataset}" \
            --results "${combo_dir}" \
            --rq1
    done
done

# ------------------------------------------------------------------------------
# Figure 4: Gradient enable/disable (RQ2) — all models x all datasets
# ------------------------------------------------------------------------------
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        combo_dir="${RESULTS_ROOT}/${model}-${dataset}"
        echo "=================================================="
        echo "Figure 4 (RQ2): model=${model} dataset=${dataset}"
        echo "Output: ${combo_dir}"
        echo "=================================================="
        ./reproduce.sh \
            "${PASSTHRU_ARGS[@]}" \
            --model "${model}" \
            --dataset "${dataset}" \
            --results "${combo_dir}" \
            --rq2
    done
done

# ------------------------------------------------------------------------------
# Figure 5: Overhead / tractability (RQ3) — coding dataset, all models
# ------------------------------------------------------------------------------
combo_dir="${RESULTS_ROOT}/gemma-coding"
echo "=================================================="
echo "Figure 5 (RQ3): model=gemma dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model gemma \
    --dataset coding \
    --results "${combo_dir}" \
    --rq3

combo_dir="${RESULTS_ROOT}/smollm-coding"
echo "=================================================="
echo "Figure 5 (RQ3): model=smollm dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model smollm \
    --dataset coding \
    --results "${combo_dir}" \
    --rq3

combo_dir="${RESULTS_ROOT}/llama-coding"
echo "=================================================="
echo "Figure 5 (RQ3): model=llama dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model llama \
    --dataset coding \
    --results "${combo_dir}" \
    --rq3

combo_dir="${RESULTS_ROOT}/qwen-coding"
echo "=================================================="
echo "Figure 5 (RQ3): model=qwen dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model qwen \
    --dataset coding \
    --results "${combo_dir}" \
    --rq3

# ------------------------------------------------------------------------------
# Figures 6 & 7: Scalability (RQ4) — coding dataset, gemma + qwen only
# ------------------------------------------------------------------------------
combo_dir="${RESULTS_ROOT}/gemma-coding"
echo "=================================================="
echo "Figures 6&7 (RQ4): model=gemma dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model gemma \
    --dataset coding \
    --results "${combo_dir}" \
    --rq4

combo_dir="${RESULTS_ROOT}/qwen-coding"
echo "=================================================="
echo "Figures 6&7 (RQ4): model=qwen dataset=coding"
echo "Output: ${combo_dir}"
echo "=================================================="
./reproduce.sh \
    "${PASSTHRU_ARGS[@]}" \
    --model qwen \
    --dataset coding \
    --results "${combo_dir}" \
    --rq4

