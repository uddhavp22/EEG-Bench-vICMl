#!/usr/bin/env bash
set -euo pipefail

# Full benchmark run: LUNA, LaBraM, CBraMod
# All 14 tasks
# Phase 1: data efficiency curve (0.1-0.75) with single seed
# Phase 2: full data (1.0) with 5 seeds for variance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments_${TIMESTAMP}"
SUMMARY_LOG="logs/run_summary_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" results/raw

echo "============================================================"
echo "EEG-Bench Full Benchmark Run"
echo "Started: $(date)"
echo "Log dir: $LOG_DIR"
echo "============================================================"

# echo ""
# echo "--- Phase 1: Data efficiency (single seed) ---"
# python run_experiments.py \
#     --models luna labram cbramod\ 
#     --percentages 0.1 0.25 0.5 0.75 \
#     --seeds 100 \
#     --gpus 3 \
#     --workers-per-gpu 1 \
#     --log-dir "$LOG_DIR" \
#     --resume \
#     2>&1 | tee "$SUMMARY_LOG"

echo ""
echo "--- Phase 2: Full data, 3 seeds for variance ---"
python run_experiments.py \
    --models reve\
    --percentages 1.0 \
    --seeds 100 200 300 \
    --gpus 3 \
    --workers-per-gpu 1 \
    --log-dir "$LOG_DIR" \
    --resume \
    2>&1 | tee -a "$SUMMARY_LOG"

echo ""
echo "============================================================"
echo "Run complete: $(date)"
echo "============================================================"

# Summarize failures from logs
FAILED=$(grep -rl "Return code: [^0]" "$LOG_DIR" 2>/dev/null || true)
if [ -n "$FAILED" ]; then
    echo ""
    echo "FAILED EXPERIMENTS:"
    echo "$FAILED" | while read -r logfile; do
        echo "  - $(basename "$logfile")"
        grep "^Command:" "$logfile" | head -1 | sed 's/^/    /'
        grep "Return code:" "$logfile" | tail -1 | sed 's/^/    /'
    done
    echo ""
    echo "Failed log files saved in: $LOG_DIR"
else
    echo "All experiments completed successfully!"
fi

# Count results
N_RESULTS=$(find results/raw -name "*.json" -newer "$SUMMARY_LOG" 2>/dev/null | wc -l | tr -d ' ')
echo "New result files: $N_RESULTS"
echo "Summary log: $SUMMARY_LOG"
