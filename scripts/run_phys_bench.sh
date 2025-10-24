#!/usr/bin/env bash
# run_phys_bench.sh

set -euo pipefail

# --------------------------- configurable defaults ---------------------------
# Path to the Python benchmark file
PYFILE="${PYFILE:-phys_consistency_bench.py}"

# Output
OUT_DIR="${OUT_DIR:-runs/phys_bench}"
OUT_CSV="${OUT_CSV:-$OUT_DIR/results.csv}"

# Core setup
SEED="${SEED:-0}"
GRID_NX="${GRID_NX:-32}"
GRID_NY="${GRID_NY:-32}"
DT="${DT:-0.002}"
KMAX="${KMAX:-6}"              # plane-wave kmax (dataset)
C_SPEED="${C_SPEED:-1.0}"      # speed used by theory Hodge for evaluation
C_WAVE="${C_WAVE:-$C_SPEED}"   # analytic wave speed; default matches c_speed

# Training / rollout
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_SIZE="${TRAIN_SIZE:-4000}"
VAL_SIZE="${VAL_SIZE:-256}"
ROLLOUT_T="${ROLLOUT_T:-200}"

# Models
MeshFT_HODGE_MODE="${MeshFT_HODGE_MODE:-learn_geom}"  # {theory,learn_geom}
MGN_HIDDEN="${MGN_HIDDEN:-64}"
MGN_LAYERS="${MGN_LAYERS:-4}"
LAM_HAM="${LAM_HAM:-0.00001}"
HNN_HIDDEN="${HNN_HIDDEN:-64}"
HNN_LAYERS="${HNN_LAYERS:-4}"

# Device: "auto" picks CUDA if available, otherwise CPU. Override via DEVICE=cpu/cuda
DEVICE="${DEVICE:-auto}"

# --------------------------- pre-flight checks -------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

if [ ! -f "$PYFILE" ]; then
  echo "ERROR: '$PYFILE' not found. Place 'phys_consistency_bench.py' next to this script or set \$PYFILE." >&2
  exit 1
fi

# Basic dependency check (soft fail with hint)
python3 - <<'PY' >/dev/null 2>&1 || true
try:
    import torch, numpy, matplotlib  # noqa: F401
except Exception as e:
    import sys
    sys.stderr.write("[WARN] Missing dependencies detected. You will need: torch, numpy, matplotlib.\n")
PY

# Decide device automatically if requested
if [ "$DEVICE" = "auto" ]; then
  DEVICE=$(python3 - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
)
fi

mkdir -p "$OUT_DIR"

# ------------------------------- run command --------------------------------
CMD=(python3 "$PYFILE"
  --out_dir "$OUT_DIR"
  --out_csv "$OUT_CSV"
  --device "$DEVICE"
  --seed "$SEED"
  --mesh grid
  --grid "$GRID_NX" "$GRID_NY"
  --Lx 1.0 --Ly 1.0
  --dt "$DT"
  --kmax "$KMAX"
  --c_speed "$C_SPEED"
  --c_wave "$C_WAVE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --train_size "$TRAIN_SIZE"
  --val_size "$VAL_SIZE"
  --meshft_hodge_mode "$MeshFT_HODGE_MODE"
  --mgn_hidden "$MGN_HIDDEN"
  --mgn_layers "$MGN_LAYERS"
  --lam_ham "$LAM_HAM"
  --hnn_hidden "$HNN_HIDDEN"
  --hnn_layers "$HNN_LAYERS"
  --rollout_T "$ROLLOUT_T"
)

# Forward any extra CLI args to the Python script
if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "[run_phys_bench] Device: $DEVICE"
echo "[run_phys_bench] Output: $OUT_DIR"
echo "[run_phys_bench] Executing:"
printf '  %q ' "${CMD[@]}"; echo

"${CMD[@]}"