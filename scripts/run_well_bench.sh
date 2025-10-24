#!/usr/bin/env bash
# Run all Well-benchmark experiments

set -Eeuo pipefail

# ----------------------- User-tunable knobs -----------------------
# Python binary and script path
PYTHON="${PYTHON:-python3}"
PY_SCRIPT="${PY_SCRIPT:-the_well_bench.py}"

# The Well base path (local folder or hf:// path)
BASE_PATH="${BASE_PATH:-hf://datasets/polymathic-ai/}"
SPLIT="${SPLIT:-train}"

# Training settings
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-2}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-800}"
VAL_STEPS="${VAL_STEPS:-50}"
DEVICE="${DEVICE:-cuda}"       # "cuda" or "cpu"

# MeshFTNet Hodge learning flags (new)
MeshFTNet_LEARN_HODGE="${MeshFTNet_LEARN_HODGE:-1}"  # 1: learn Hodge in MeshFTNet, 0: fix to theory
MeshFTNet_LEARN_M="${MeshFTNet_LEARN_M:-1}"          # learn node mass M
MeshFTNet_LEARN_W="${MeshFTNet_LEARN_W:-1}"          # learn edge weight W
HODGE_REG="${HODGE_REG:-0.0}"           # small L2 reg on Hodge log-params

# Saving & benchmarking (requires the Python script to support these flags)
SAVE_MODELS="${SAVE_MODELS:-1}"          # 1: save checkpoints
SAVE_EVERY="${SAVE_EVERY:-1}"            # save every N epochs (use 0 to save only final/best if supported)
# Seeds (space-separated)
SEEDS_STR="${SEEDS:-0}"

# Output root for logs & artifacts
LOG_ROOT="${LOG_ROOT:-./runs/well_bench}"
mkdir -p "$LOG_ROOT"

# -----------------------------------------------------------------

# Small helper to run one configuration
run_one () {
  local dataset="$1"     # dataset slug in The Well
  local regime="$2"      # human-readable regime tag for folder names
  local extra="$3"       # extra CLI args to pass to the_well_bench.py
  local seed="$4"

  local tag="${dataset}_${regime}_seed${seed}"
  local outdir="$LOG_ROOT/$dataset/$regime/$tag"
  mkdir -p "$outdir"

  echo "----------------------------------------------------------------"
  echo "[RUN] dataset=${dataset} regime=${regime} seed=${seed}"
  echo "      run_dir -> $outdir"
  echo "      log     -> $outdir/train.log"
  echo "----------------------------------------------------------------"

  # Common arguments passed to every run
  # Note: --save_dir / --bench_csv / --save_models / --save_every must exist in your Python script.
  #       They were added in the previous step where saving & standard evaluation were implemented.
  "$PYTHON" "$PY_SCRIPT" \
    --dataset "$dataset" \
    --split "$SPLIT" \
    --base_path "$BASE_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --val_steps "$VAL_STEPS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --meshft_learn_hodge "$MeshFTNet_LEARN_HODGE" \
    --meshft_learn_M "$MeshFTNet_LEARN_M" \
    --meshft_learn_W "$MeshFTNet_LEARN_W" \
    --hodge_reg "$HODGE_REG" \
    --save_dir "$outdir" \
    --save_models "$SAVE_MODELS" \
    --save_every "$SAVE_EVERY" \
    --bench_csv "$outdir/bench_summary.csv" \
    $extra \
    |& tee "$outdir/train.log"
}

# Iterate seeds into an array
read -r -a SEEDS <<<"$SEEDS_STR"

# ======================= Acoustic scattering (2D) =======================
# Dataset: acoustic_scattering_discontinuous
# Dissipation is learned inside MeshFTNet; rayleigh_gamma flags are kept for backward-compat (no-op if unused).
for s in "${SEEDS[@]}"; do
  run_one "acoustic_scattering_discontinuous" "conservative" "--rayleigh_gamma 0.0" "$s"
done
echo "✅ All Well benchmark runs finished. See $LOG_ROOT for logs, CSV summaries, and saved models."