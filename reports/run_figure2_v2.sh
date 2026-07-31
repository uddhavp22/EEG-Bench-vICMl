#!/usr/bin/env bash
# Figure 2 quantification, v2. Supersedes every eta2 margin reported before
# 2026-07-30: those used the uniform `_relocate` null, which manufactures a
# margin of +0.080 on a centred-boundary clock and -0.019 on an off-centre one
# (see calib_null_bias.py). Both published margins are void.
#
# Lives at <repo>/reports/, so this works from the repo root:
#     bash reports/run_figure2_v2.sh
# Runs A/B are the ones that describe the PUBLISHED figure and are the priority.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="reports/f2v2"
mkdir -p "$OUT"

# H5 PATHS ARE PINNED ON PURPOSE. The repo-local ./data/make_dataset holds only
# LeJEPAClinical builds, which are defossez-scaled and CLIPPED TO +/-20
# (utils_2.py:516). That clipping flattens exactly the high-amplitude ictal and
# artifact segments these metrics are about, and it is what made every LaBraM
# number before 2026-07-30 wrong. The LaBraMModel builds are unclipped
# microvolts and live on radraid. Do not let --h5-tag fall back to a lookup.
DATA="${DATA:-/radraid2/spanchavati/EEGBench/data/make_dataset}"
H5_ARTIFACT="$DATA/binary_artifact_clinical_LaBraMModel_TUAR_16_True_243.h5"
H5_SEIZURE="$DATA/seizure_clinical_LaBraMModel_CHB-MIT_16_True_526.h5"

for f in "$H5_ARTIFACT" "$H5_SEIZURE"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f" >&2
    echo "Set DATA=/path/to/make_dataset, or find the build with:" >&2
    echo "  ls $DATA/*_LaBraMModel_*.h5" >&2
    echo "Do NOT substitute a LeJEPAClinical build; it is clipped to +/-20." >&2
    exit 1
  fi
done

COMMON="--h5-tag LaBraMModel --pc-norm plot --null chunk --guard 0.5 \
        --n-null 40 --tracking --state-auc --reversal --decoder"

# --- A. artifact, sliding LaBraM. This is the configuration Figure 2 shows. ---
python quantify_pca_changepoint.py \
  --task binary_artifact_clinical --split train --h5 "$H5_ARTIFACT" \
  $COMMON --labram-mode sliding --auc-max-chunks 1500 \
  --out figures/cp_artifact_v2_sliding \
  2>&1 | tee "$OUT/artifact_sliding.log"

# --- B. seizure, sliding LaBraM. n=216, so it fits entirely in RAM. ---
python quantify_pca_changepoint.py \
  --task seizure_clinical --split train --h5 "$H5_SEIZURE" \
  $COMMON --labram-mode sliding --auc-max-chunks 2000 \
  --out figures/cp_seizure_v2_sliding \
  2>&1 | tee "$OUT/seizure_sliding.log"

# --- C/D. native LaBraM: one 16 s pass, full across-time attention, 1 Hz. ---
# This is the FAIR arm for any claim about LaBraM as a model. It is NOT the
# figure. If LaBraM's jitter collapses here, the published contrast is a
# statement about our analysis protocol, not about the two models.
python quantify_pca_changepoint.py \
  --task binary_artifact_clinical --split train --h5 "$H5_ARTIFACT" \
  $COMMON --labram-mode native --auc-max-chunks 1500 \
  --out figures/cp_artifact_v2_native \
  2>&1 | tee "$OUT/artifact_native.log"

python quantify_pca_changepoint.py \
  --task seizure_clinical --split train --h5 "$H5_SEIZURE" \
  $COMMON --labram-mode native --auc-max-chunks 2000 \
  --out figures/cp_seizure_v2_native \
  2>&1 | tee "$OUT/seizure_native.log"

echo "done. logs in $OUT/"
