#!/usr/bin/env bash
# ==============================================================================
# ProToken: Top-level script to reproduce all paper figures
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# DEFAULTS
# ------------------------------------------------------------------------------
ROUNDS=10
SCALABILITY_ROUNDS=16
MODEL="qwen"
DATASET="coding"
RQ2_SAMPLES=20
RQ3_SAMPLES=20
RUN_FIG_2_3=0
RUN_FIG_4=0
RUN_FIG_5=0
RUN_FIG_6_7=0

# ------------------------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --scalability-rounds)
            SCALABILITY_ROUNDS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --rq2-samples)
            RQ2_SAMPLES="$2"
            shift 2
            ;;
        --rq3-samples)
            RQ3_SAMPLES="$2"
            shift 2
            ;;
        --cache)
            export GENFL_EXPERIMENT_CACHE="$2"
            shift 2
            ;;
        --rq1)
            RUN_FIG_2_3=1
            shift
            ;;
        --rq2)
            RUN_FIG_4=1
            shift
            ;;
        --rq3)
            RUN_FIG_5=1
            shift
            ;;
        --rq4)
            RUN_FIG_6_7=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [FLAGS]"
            echo ""
            echo "Options:"
            echo "  --rounds N              Federated rounds (default: 10)"
            echo "  --scalability-rounds N  Rounds for scalability (default: 16)"
            echo "  --model NAME            Model: gemma|smollm|llama|qwen (default: qwen)"
            echo "  --dataset NAME          Dataset: medical|finance|math|coding (default: coding)"
            echo "  --rq2-samples N         Samples for RQ2/Fig4 (default: 20)"
            echo "  --rq3-samples N         Samples for RQ3/Fig5 (default: 20)"
            echo "  --cache DIR             Experiment cache (default: /tmp/genfl_experiment_cache)"
            echo ""
            echo "Flags (which RQs to run):"
            echo "  --rq1   Fig 2 & 3 — Main accuracy + provenance"
            echo "  --rq2   Fig 4 — Gradient enable/disable"
            echo "  --rq3   Fig 5 — Overhead / tractability"
            echo "  --rq4   Fig 6 & 7 — Scalability"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

export GENFL_EXPERIMENT_CACHE="${GENFL_EXPERIMENT_CACHE:-/tmp/genfl_experiment_cache}"

GRAPHS_DIR="results/graphs"

echo "=================================================="
echo "  ProToken reproduction"
echo "  Model: ${MODEL}  |  Dataset: ${DATASET}  |  Rounds: ${ROUNDS}"
echo "  RQ1: ${RUN_FIG_2_3}  RQ2: ${RUN_FIG_4}  RQ3: ${RUN_FIG_5}  RQ4: ${RUN_FIG_6_7}"
echo "=================================================="

# ------------------------------------------------------------------------------
# Fig 2, Fig 3 — Main accuracy + client contribution distributions
# Requires: training with backdoor, provenance run, then plotting.
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_2_3}" -eq 1 ]]; then
    RESULTS_FIG23="results/rq1-fig2-fig3"
    rm -rf "${RESULTS_FIG23}"
    mkdir -p "${RESULTS_FIG23}"
    mkdir -p "${GRAPHS_DIR}/rq1"

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
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_4}" -eq 1 ]]; then
    RESULTS_FIG4="results/rq2-fig4"
    rm -rf "${RESULTS_FIG4}"
    mkdir -p "${RESULTS_FIG4}"
    mkdir -p "${GRAPHS_DIR}/rq2"

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
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_5}" -eq 1 ]]; then
    RESULTS_FIG5="results/rq3-fig5"
    rm -rf "${RESULTS_FIG5}"
    mkdir -p "${RESULTS_FIG5}"
    mkdir -p "${GRAPHS_DIR}/rq3"

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
# ------------------------------------------------------------------------------
if [[ "${RUN_FIG_6_7}" -eq 1 ]]; then
    RESULTS_FIG67="results/rq4-fig6-fig7"
    rm -rf "${RESULTS_FIG67}"
    mkdir -p "${RESULTS_FIG67}"
    mkdir -p "${GRAPHS_DIR}/rq4"

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
        --results_dir "${RESULTS_FIG67}/train/backdoor"

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
echo "  All done. Results: results/rq*-*  |  Graphs: ${GRAPHS_DIR}"
echo "=================================================="