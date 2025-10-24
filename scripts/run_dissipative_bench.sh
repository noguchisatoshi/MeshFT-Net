#!/usr/bin/env bash
# Run the dissipative benchmark (MeshFT-Net, MGN, MGN-HP, HNN).


set -euo pipefail

# ---- configurable defaults ----
PYTHON_BIN="${PYTHON:-python3}"
SCRIPT_PATH="${SCRIPT:-dissipative_bench.py}"

OUT_ROOT="${OUT_ROOT:-runs/dissipative_bench_kdk}"
SEEDS_STR="${SEEDS:-"0 1 2"}"

GRID_NX="${GRID_NX:-32}"
GRID_NY="${GRID_NY:-32}"
LX="${LX:-1.0}"
LY="${LY:-1.0}"

DT="${DT:-0.002}"
KMAX="${KMAX:-6}"
C_SPEED="${C_SPEED:-1.0}"
# Leave C_WAVE empty to default to C_SPEED inside the script
C_WAVE="${C_WAVE:-}"

GAMMA_MIN="${GAMMA_MIN:-0.01}"
GAMMA_MAX="${GAMMA_MAX:-0.1}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_SIZE="${TRAIN_SIZE:-4000}"
VAL_SIZE="${VAL_SIZE:-256}"

LAM_HAM="${LAM_HAM:-0.05}"          # HP weight for MGN-HP
ROLL_T="${ROLL_T:-200}"

USE_SN="${USE_SN:-0}"               # 0/1 spectral_norm for Linear layers
CFL_GATE="${CFL_GATE:-1}"           # 0/1
CFL_SAFETY="${CFL_SAFETY:-1.0}"

DEVICE="${DEVICE:-cuda}"            # "cuda" or "cpu"

# ---- sanity checks ----
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "ERROR: ${PYTHON_BIN} not found."; exit 1; }
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: Python script not found at ${SCRIPT_PATH}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

# Optional flag for c_wave (only if provided)
C_WAVE_FLAG=()
if [[ -n "${C_WAVE}" ]]; then
  C_WAVE_FLAG=(--c_wave "${C_WAVE}")
fi

# ---- run for each seed ----
for seed in ${SEEDS_STR}; do
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${OUT_ROOT}/grid${GRID_NX}x${GRID_NY}_s${seed}_${ts}"
  LOG_DIR="${OUT_DIR}/logs"
  mkdir -p "${OUT_DIR}" "${LOG_DIR}"

  echo "[RUN] seed=${seed} -> ${OUT_DIR}"

  "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --out_dir "${OUT_DIR}" \
    --out_csv "${OUT_ROOT}/results.csv" \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --grid "${GRID_NX}" "${GRID_NY}" \
    --Lx "${LX}" --Ly "${LY}" \
    --dt "${DT}" \
    --kmax "${KMAX}" \
    --c_speed "${C_SPEED}" \
    "${C_WAVE_FLAG[@]}" \
    --gamma_min "${GAMMA_MIN}" \
    --gamma_max "${GAMMA_MAX}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --train_size "${TRAIN_SIZE}" \
    --val_size "${VAL_SIZE}" \
    --damp_mode "learn_diag_latent" \
    --mgn_hidden 64 \
    --mgn_layers 4 \
    --use_sn "${USE_SN}" \
    --lam_ham "${LAM_HAM}" \
    --hnn_hidden 64 \
    --hnn_layers 4 \
    --rollout_T "${ROLL_T}" \
    --save_details 1 \
    2>&1 | tee "${LOG_DIR}/seed_${seed}.log"
done

echo "[DONE] All runs finished. Aggregated CSV: ${OUT_ROOT}/results.csv"