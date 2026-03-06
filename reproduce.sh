#!/usr/bin/env bash
# ==============================================================================
# ProToken: Top-level script to reproduce all paper figures
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# TOP-LEVEL CONFIGURATION — edit these before running
# ------------------------------------------------------------------------------

# Number of federated rounds to train
ROUNDS=3

# Model and dataset to use for Fig 2, 3, 4, 5
# These are passed into the training/provenance/plotting scripts
MODEL="gemma"       # e.g. gemma | smollm | llama | qwen
DATASET="coding"   # e.g. medical | finance | math | coding

# Models to use for Fig 6 & 7 (scalability). Paper uses gemma and qwen.
# Must be a space-separated list; only gemma/qwen are supported by the
# scalability training script without modification.
SCALABILITY_MODELS="gemma qwen"

# Number of samples for RQ2 (Fig 4) and RQ3 (Fig 5) evaluations
RQ2_SAMPLES=20
RQ3_SAMPLES=20

# Figure groups to run — set to 0 to skip a group
RUN_FIG_2_3=1   # Main accuracy results (requires training + provenance)
RUN_FIG_4=1     # Gradient enable/disable (RQ2)
RUN_FIG_5=1     # Overhead / tractability (RQ3)
RUN_FIG_6_7=1   # Scalability (requires separate training + provenance)

# ------------------------------------------------------------------------------
# DIRECTORY SETUP
# Timestamped root keeps runs from clobbering each other.
# Each figure group gets its own results subdir.
# All plots land in a matching timestamped subdir under paper/graphs/.
# ------------------------------------------------------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_ROOT="results/${TIMESTAMP}"
GRAPHS_DIR="paper/graphs/${TIMESTAMP}"

RESULTS_FIG23="${RESULTS_ROOT}/fig2_3_main"
RESULTS_FIG4="${RESULTS_ROOT}/fig4_rq2_grad"
RESULTS_FIG5="${RESULTS_ROOT}/fig5_rq3_overhead"
RESULTS_FIG67="${RESULTS_ROOT}/fig6_7_scalability"

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
        --output_dir "${RESULTS_FIG23}"

    uv run python -m src.run_provenance \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --results_dir "${RESULTS_FIG23}"

    # edit file plotting/common.py, pick model and dataset
    uv run python -m plotting.plot_eval_main_results \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --results_dir "${RESULTS_FIG23}" \
        --output_dir "${GRAPHS_DIR}"
fi

# ------------------------------------------------------------------------------
# Fig 4 — Gradient weighting enable/disable (RQ2: Relevance Filtering)
# change output dir and line 117/118 in fl_prov.py
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_4}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 4: Gradient enable/disable (RQ2) ---"

    # Had to tinker with file pattern matching, might need generalizing
    uv run python -m src.run_RQ2_layers \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --round_num "${ROUNDS}" \
        --num_samples "${RQ2_SAMPLES}" \
        --output_dir "${RESULTS_FIG4}"

    uv run python -m plotting.plot_grad_enable_disable \
        --results_dir "${RESULTS_FIG4}" \
        --output_dir "${GRAPHS_DIR}"
fi

# ------------------------------------------------------------------------------
# Fig 5 — Computational overhead vs. layer count (RQ3: Tractability)
# By default plots for code dataset, change in file
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_5}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 5: Overhead (RQ3) ---"

    # had to add the round num arg, was hardcoded to 10
    uv run python -m src.run_RQ3_overhead \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --round_num "${ROUNDS}" \
        --num_samples "${RQ3_SAMPLES}" \
        --output_dir "${RESULTS_FIG5}"

    uv run python -m plotting.plot_overhead \
        --round_num "${ROUNDS}" \
        --results_dir "${RESULTS_FIG5}" \
        --output_dir "${GRAPHS_DIR}"
fi

# ------------------------------------------------------------------------------
# Fig 6, Fig 7 — Scalability (55 clients)
# Comment in/out relevant models and datasets
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_6_7}" -eq 1 ]]; then
    echo ""
    echo "--- Fig 6 & 7: Scalability (train + provenance) ---"

    uv run python -m src.run_train_scalabiliity_backdoor \
        --models ${SCALABILITY_MODELS} \
        --dataset "${DATASET}" \
        --rounds "${ROUNDS}" \
        --output_dir "${RESULTS_FIG67}"

    #          * UNCOMMENT: full_cache_provenance(results_dir)
    #          * COMMENT OUT: single_key_provenance(debug_dir)
    uv run python -m src.run_provenance \
        --models ${SCALABILITY_MODELS} \
        --dataset "${DATASET}" \
        --results_dir "${RESULTS_FIG67}" \
        --scalability

    # hard coded to gemma and qwen, added SmolLM myself
    uv run python -m plotting.plot_scalability \
        --models ${SCALABILITY_MODELS} \
        --results_dir "${RESULTS_FIG67}" \
        --output_dir "${GRAPHS_DIR}"
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