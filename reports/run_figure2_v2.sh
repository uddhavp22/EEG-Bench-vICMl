#!/usr/bin/env bash
# Figure 2 quantification, v2. Supersedes every eta2 margin reported before
# 2026-07-30: those used the uniform `_relocate` null, which manufactures a
# margin of +0.080 on a centred-boundary clock and -0.019 on an off-centre one
# (see calib_null_bias.py). Both published margins are void.
#
# Run from EEG-Bench-vICMl/ on the GPU box. Each run writes its own log.
# Runs A/B are the ones that describe the PUBLISHED figure and are the priority.
set -euo pipefail
# Lives at <repo>/reports/, so this works from anywhere:
#     bash reports/run_figure2_v2.sh
cd "$(dirname "$0")/.."
OUT="reports/f2v2"
mkdir -p "$OUT"

COMMON="--h5-tag LaBraMModel --pc-norm plot --null chunk --guard 0.5 \
        --n-null 40 --tracking --state-auc --reversal --decoder"

# --- A. artifact, sliding LaBraM. This is the configuration Figure 2 shows. ---
python quantify_pca_changepoint.py \
  --task binary_artifact_clinical --split train \
  $COMMON --labram-mode sliding --auc-max-chunks 1500 \
  --out figures/cp_artifact_v2_sliding \
  2>&1 | tee "$OUT/artifact_sliding.log"

# --- B. seizure, sliding LaBraM. n=216, so it fits entirely in RAM. ---
python quantify_pca_changepoint.py \
  --task seizure_clinical --split train \
  $COMMON --labram-mode sliding --auc-max-chunks 2000 \
  --out figures/cp_seizure_v2_sliding \
  2>&1 | tee "$OUT/seizure_sliding.log"

# --- C/D. native LaBraM: one 16 s pass, full across-time attention, 1 Hz. ---
# This is the FAIR arm for any claim about LaBraM as a model. It is NOT the
# figure. If LaBraM's jitter collapses here, the published contrast is a
# statement about our analysis protocol, not about the two models.
python quantify_pca_changepoint.py \
  --task binary_artifact_clinical --split train \
  $COMMON --labram-mode native --auc-max-chunks 1500 \
  --out figures/cp_artifact_v2_native \
  2>&1 | tee "$OUT/artifact_native.log"

python quantify_pca_changepoint.py \
  --task seizure_clinical --split train \
  $COMMON --labram-mode native --auc-max-chunks 2000 \
  --out figures/cp_seizure_v2_native \
  2>&1 | tee "$OUT/seizure_native.log"

echo "done. logs in $OUT/"
