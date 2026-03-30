#!/usr/bin/env bash
# Run benchmarks with frozen encoder + MLP/classifier head (NOT linear probe).
# Encoder stays frozen for all models; only the classification head trains.
#
# Models: labram, cbramod, luna, reve
# All clinical + BCI tasks, 100% data, seeds 100-500
#
# Usage:
#   bash run_mlp_benchmark.sh              # full run
#   bash run_mlp_benchmark.sh --dry-run    # preview commands
#   bash run_mlp_benchmark.sh --resume     # skip completed

set -euo pipefail

MODELS="labram cbramod luna" # reve
SEEDS="100 200 300" # 400 500
PERCENTAGES="1.0"
TASKS="right_feet left_right_feet_tongue abnormal seizure binary_artifact"
GPUS=3
WORKERS=1  # conservative to avoid h5 lock issues

echo "============================================"
echo "MLP Benchmark (frozen encoder, NO linear probe)"
echo "============================================"
echo "Models: ${MODELS}"
echo "Tasks:  ${TASKS}"
echo "Seeds:  ${SEEDS}"
echo "GPUs:   ${GPUS}, Workers/GPU: ${WORKERS}"
echo "============================================"

python run_experiments.py \
    --models ${MODELS} \
    --tasks ${TASKS} \
    --percentages ${PERCENTAGES} \
    --seeds ${SEEDS} \
    --gpus ${GPUS} \
    --workers-per-gpu ${WORKERS} \
    --no-linear-probe \
    --resume \
    --log-dir "logs/mlp_benchmark" \
    "$@"
