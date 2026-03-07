#!/usr/bin/env bash
# ==============================================================================
# ProToken: Top-level script to reproduce all paper figures
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# TOP-LEVEL CONFIGURATION — edit these before running
# ------------------------------------------------------------------------------

# Number of federated rounds to train
ROUNDS=10
SCALABILITY_ROUNDS=16

# Model and dataset combo — used for all figures including scalability (Fig 6 & 7)
MODEL="qwen"       # e.g. gemma | smollm | llama | qwen
DATASET="coding"   # e.g. medical | finance | math | coding

# Number of samples for RQ2 (Fig 4) and RQ3 (Fig 5) evaluations
RQ2_SAMPLES=20
RQ3_SAMPLES=20

# Experiment cache directory (used by CacheManager; default if unset)
export GENFL_EXPERIMENT_CACHE="${GENFL_EXPERIMENT_CACHE:-/scratch/ahmad35/_storage/caches/complete_experiment_cache-4}"

# Figure groups to run — set to 0 to skip a group
RUN_FIG_2_3=0   # Main accuracy results (requires training + provenance)
RUN_FIG_4=0     # Gradient enable/disable (RQ2)
RUN_FIG_5=0     # Overhead / tractability (RQ3)
RUN_FIG_6_7=1   # Scalability (requires separate training + provenance)

# ------------------------------------------------------------------------------
# DIRECTORY SETUP
# Timestamped root keeps runs from clobbering each other.
# Each figure group gets its own results subdir.
# All plots land in a matching timestamped subdir under paper/graphs/.
# ------------------------------------------------------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_ROOT="results/${TIMESTAMP}"
GRAPHS_DIR="results/${TIMESTAMP}/graphs"

RESULTS_FIG23="${RESULTS_ROOT}/rq1-fig2-fig3"
RESULTS_FIG4="${RESULTS_ROOT}/rq2-fig4"
RESULTS_FIG5="${RESULTS_ROOT}/rq3-fig5"
RESULTS_FIG67="${RESULTS_ROOT}/rq4-fig6-fig7"

mkdir -p \
    "${RESULTS_FIG23}" \
    "${RESULTS_FIG4}" \
    "${RESULTS_FIG5}" \
    "${RESULTS_FIG67}" \
    "${GRAPHS_DIR}"

echo "=================================================="
echo "  ProToken reproduction run: ${TIMESTAMP}"
echo "  Model: ${MODEL}  |  Dataset: ${DATASET}  |  Rounds: ${ROUNDS}"
echo "  Results root : ${RESULTS_ROOT}"
echo "  Graphs dir   : ${GRAPHS_DIR}"
echo "=================================================="

# ------------------------------------------------------------------------------
# Fig 2, Fig 3 — Main accuracy + client contribution distributions
# Requires: training with backdoor, provenance run, then plotting.
# edit file: rounds = 3, pick model, pick dataset
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_2_3}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 2 & 3: Main results (train + provenance) ---"

    uv run python -m src.run_train_backdoor \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --rounds "${ROUNDS}" \
        --output_dir "${RESULTS_FIG23}/train/backdoor"

    uv run python -m src.run_provenance \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --rounds "${ROUNDS}" \
        --results_dir "${RESULTS_FIG23}/prov/backdoor"

    uv run python -m plotting.plot_eval_main_results \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --results_dir "${RESULTS_FIG23}/prov/backdoor" \
        --output_dir "${GRAPHS_DIR}/rq1"
fi

# ------------------------------------------------------------------------------
# Fig 4 — Gradient weighting enable/disable (RQ2: Relevance Filtering)
# change output dir and line 117/118 in fl_prov.py
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_4}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 4: Gradient enable/disable (RQ2) ---"

    uv run python -m src.run_RQ2_layers \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --round_num "${ROUNDS}" \
        --num_samples "${RQ2_SAMPLES}" \
        --output_dir "${RESULTS_FIG4}"

    uv run python -m plotting.plot_grad_enable_disable \
        --results_dir "${RESULTS_FIG4}" \
        --output_dir "${GRAPHS_DIR}/rq2"
fi

# ------------------------------------------------------------------------------
# Fig 5 — Computational overhead vs. layer count (RQ3: Tractability)
# By default plots for code dataset, change in file
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_5}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 5: Overhead (RQ3) ---"

    uv run python -m src.run_RQ3_overhead \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --round_num "${ROUNDS}" \
        --num_samples "${RQ3_SAMPLES}" \
        --output_dir "${RESULTS_FIG5}"

    uv run python -m plotting.plot_overhead \
        --round_num "${ROUNDS}" \
        --results_dir "${RESULTS_FIG5}" \
        --output_dir "${GRAPHS_DIR}/rq3"
fi

# ------------------------------------------------------------------------------
# Fig 6, Fig 7 — Scalability (55 clients)
# Comment in/out relevant models and datasets
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_6_7}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 6 & 7: Scalability (train + provenance) ---"

    uv run python -m src.run_train_scalabiliity_backdoor \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --rounds "${SCALABILITY_ROUNDS}" \
        --output_dir "${RESULTS_FIG67}/train/backdoor"


    uv run python -m src.run_provenance \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --rounds "${SCALABILITY_ROUNDS}" \
        --results_dir "${RESULTS_FIG67}/train/backdoor" \

    uv run python -m plotting.plot_scalability \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --rounds "${SCALABILITY_ROUNDS}" \
        --results_dir "${RESULTS_FIG67}/train/backdoor" \
        --output_dir "${GRAPHS_DIR}/rq4"
fi

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  All done. Outputs:"
echo "  Results : ${RESULTS_ROOT}"
echo "  Graphs  : ${GRAPHS_DIR}"
echo "=================================================="