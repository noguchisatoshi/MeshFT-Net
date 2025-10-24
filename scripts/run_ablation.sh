#!/usr/bin/env bash
# Run MeshFT ablation over multiple seeds and aggregate into a single CSV

set -Eeuo pipefail

# ====== user knobs ============================================================
PYTHON=${PYTHON:-python3}
SCRIPT=${SCRIPT:-meshft_ablation.py}

OUT_DIR=${OUT_DIR:-runs/meshft_ablation}
DEVICE=${DEVICE:-cuda}             # cuda or cpu
GRID_X=${GRID_X:-32}
GRID_Y=${GRID_Y:-32}
DT=${DT:-0.002}
C_SPEED=${C_SPEED:-1.0}
KMAX=${KMAX:-4}
TRAIN_PAIRS=${TRAIN_PAIRS:-2000}
VAL_PAIRS=${VAL_PAIRS:-256}
BATCH=${BATCH:-8}
EPOCHS=${EPOCHS:-10}
ROLLOUT_T=${ROLLOUT_T:-200}

# Comma-separated seeds (example: "0,1,2,3,4")
SEEDS=${SEEDS:-0,1,2,3,4}

# Aggregate CSV (created if missing)
CSV_PATH=${CSV_PATH:-${OUT_DIR}/ablation_results.csv}
# ============================================================================

mkdir -p "${OUT_DIR}"

echo "== MeshFT ablation (multi-seed) =="
echo "out_dir: ${OUT_DIR}"
echo "seeds  : ${SEEDS}"
echo "csv    : ${CSV_PATH}"
echo "device : ${DEVICE}"

"${PYTHON}" "${SCRIPT}" \
  --device "${DEVICE}" \
  --out_dir "${OUT_DIR}" \
  --grid "${GRID_X}" "${GRID_Y}" \
  --dt "${DT}" \
  --c "${C_SPEED}" \
  --kmax "${KMAX}" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --rollout_T "${ROLLOUT_T}" \
  --seeds "${SEEDS}" \
  --csv_path "${CSV_PATH}" \
  |& tee "${OUT_DIR}/ablation_seeds.log"

echo "✅ Done. CSV -> ${CSV_PATH}"