#!/usr/bin/env bash
# Extrapolation suite runner for 4 models (MeshFT-Net / MGN / MGN-HP / HNN).
# - Trains ALL models (including MeshFT-Net with learnable geometry-conditioned Hodge)
# - Evaluates extrapolation on four scenarios:
#   (1) Frequency OOD, (2) Resolution OOD, (3) Parameter OOD, (4) Long-horizon OOD

set -Eeuo pipefail

# ----------------------- configurable knobs (override via env) -----------------------
OUT_ROOT="${OUT_ROOT:-runs/extrapolation_test_kdk}"
PY="${PYTHON_BIN:-python3}"

# Training distribution
TRAIN_GRID_X="${TRAIN_GRID_X:-32}"
TRAIN_GRID_Y="${TRAIN_GRID_Y:-32}"
TRAIN_DT="${TRAIN_DT:-0.004}"
TRAIN_KMAX="${TRAIN_KMAX:-3}"
TRAIN_C="${TRAIN_C:-1.0}"
TRAIN_PAIRS="${TRAIN_PAIRS:-4000}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-10}"
Q_ONLY="${Q_ONLY:-0}"              # 1 = supervise q only (p is unobserved)

# MeshFT-Net Hodge mode: use learnable geometry-conditioned Hodge to ensure "training" also for MeshFT-Net
MeshFT_HODGE_MODE="${MeshFT_HODGE_MODE:-learn_geom}"

# MGN-HP penalty weight
LAM_HAM="${LAM_HAM:-0.05}"

# Seeds
SEEDS=(${SEEDS:-0 1 2})

# Test defaults (each scenario may overwrite)
TEST_DT_DEFAULT="${TEST_DT_DEFAULT:-0.004}"
ROLLOUT_T_DEFAULT="${ROLLOUT_T_DEFAULT:-200}"

# ----------------------- helpers -----------------------

detect_device() {
  # Ask Python whether CUDA is available; fall back to CPU.
  ${PY} - <<'PY'
import torch, sys
print("cuda" if torch.cuda.is_available() else "cpu")
PY
}

DEVICE="$(detect_device)"

assert_py_script() {
  if [[ ! -f "extrapolation_bench.py" ]]; then
    echo "[ERROR] Missing extrapolation_bench.py in current directory." >&2
    exit 1
  fi
}

run_one_case() {
  # Args:
  #   $1 = scenario name (freq|reso|param|long)
  #   $2 = seed
  local SCEN="$1"
  local SEED="$2"

  # Defaults (override per scenario below)
  local TEST_GRID_X TEST_GRID_Y TEST_DT TEST_KMAX TEST_C ROLLOUT_T
  TEST_GRID_X="$TRAIN_GRID_X"
  TEST_GRID_Y="$TRAIN_GRID_Y"
  TEST_DT="$TEST_DT_DEFAULT"
  TEST_KMAX="$TRAIN_KMAX"
  TEST_C="$TRAIN_C"
  ROLLOUT_T="$ROLLOUT_T_DEFAULT"

  case "${SCEN}" in
    freq)
      # Frequency extrapolation: higher kmax on test
      TEST_KMAX=$(( TRAIN_KMAX * 2 ))
      ;;
    reso)
      # Resolution extrapolation: finer mesh
      TEST_GRID_X=$(( TRAIN_GRID_X * 2 ))
      TEST_GRID_Y=$(( TRAIN_GRID_Y * 2 ))
      ;;
    param)
      # Parameter extrapolation: different wave speed at test
      TEST_C="1.4"
      ;;
    long)
      # Long-horizon extrapolation: much longer rollout
      ROLLOUT_T=$(( ROLLOUT_T_DEFAULT * 3 ))
      ;;
    *)
      echo "[ERROR] Unknown scenario: ${SCEN}" >&2
      exit 1
      ;;
  esac

  local OUT_DIR="${OUT_ROOT}/${SCEN}/seed-${SEED}"
  mkdir -p "${OUT_DIR}"

  echo "--------------------------------------------------------------------------------"
  echo "[RUN] scenario=${SCEN}  seed=${SEED}  device=${DEVICE}"
  echo "      train: grid=${TRAIN_GRID_X}x${TRAIN_GRID_Y}, kmax=${TRAIN_KMAX}, c=${TRAIN_C}, dt=${TRAIN_DT}"
  echo "      test : grid=${TEST_GRID_X}x${TEST_GRID_Y}, kmax=${TEST_KMAX}, c=${TEST_C}, dt=${TEST_DT}, T=${ROLLOUT_T}"
  echo "--------------------------------------------------------------------------------"

  ${PY} extrapolation_bench.py \
    --out_dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --train_grid "${TRAIN_GRID_X}" "${TRAIN_GRID_Y}" \
    --train_dt "${TRAIN_DT}" \
    --train_kmax "${TRAIN_KMAX}" \
    --train_c "${TRAIN_C}" \
    --train_pairs "${TRAIN_PAIRS}" \
    --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" \
    --q_only_supervision "${Q_ONLY}" \
    --test_grid "${TEST_GRID_X}" "${TEST_GRID_Y}" \
    --test_dt "${TEST_DT}" \
    --test_kmax "${TEST_KMAX}" \
    --test_c "${TEST_C}" \
    --test_pairs 512 \
    --rollout_T "${ROLLOUT_T}" \
    --meshft_hodge_mode "${MeshFT_HODGE_MODE}" \
    --mgn_hidden 64 \
    --mgn_layers 4 \
    --lam_ham "${LAM_HAM}" \
    --hnn_hidden 64 \
    --hnn_layers 4
}

aggregate_csv() {
  # Aggregate all JSON summaries into a single CSV (pure stdlib Python).
  local OUT_DIR="${OUT_ROOT}/_compiled"
  mkdir -p "${OUT_DIR}"
  local CSV="${OUT_DIR}/summary_compiled.csv"

  ${PY} - "${OUT_ROOT}" "${CSV}" <<'PY'
import os, json, csv, sys, glob

root = sys.argv[1]
csv_path = sys.argv[2]
rows = []
for path in glob.glob(os.path.join(root, "*", "seed-*", "extrapolation_summary.json")):
    try:
        with open(path, "r") as f:
            j = json.load(f)
        parts = path.replace(root + os.sep, "").split(os.sep)
        scenario = parts[0]
        seed = parts[1].split("-")[-1]
        train = j["train"]; test = j["test"]
        mse = j["one_step_mse"]; ro = j["rollout"]
        rows.append({
            "scenario": scenario,
            "seed": int(seed),
            "train_grid": f"{train['grid'][0]}x{train['grid'][1]}",
            "test_grid": f"{test['grid'][0]}x{test['grid'][1]}",
            "train_kmax": train["kmax"],
            "test_kmax": test["kmax"],
            "train_c": train["c"],
            "test_c": test["c"],
            "train_dt": train["dt"],
            "test_dt": test["dt"],
            "epochs": train["epochs"],
            "train_pairs": train["pairs"],
            "rollout_T": test["rollout_T"],
            "q_only": train["q_only"],
            "meshft_hodge_mode": train["meshft_hodge_mode"],
            "MSE_MeshFT-Net": float(mse["MeshFT-Net"]),
            "MSE_MGN": float(mse["MGN"]),
            "MSE_MGNHP": float(mse["MGN-HP"]),
            "MSE_HNN": float(mse["HNN"]),
            "RelFinal_MeshFT-Net": float(ro["MeshFT-Net"]["rel_final"]),
            "RelFinal_MGN": float(ro["MGN"]["rel_final"]),
            "RelFinal_MGNHP": float(ro["MGN-HP"]["rel_final"]),
            "RelFinal_HNN": float(ro["HNN"]["rel_final"]),
            "Drift_MeshFT-Net": float(ro["MeshFT-Net"]["drift_mean"]),
            "Drift_MGN": float(ro["MGN"]["drift_mean"]),
            "Drift_MGNHP": float(ro["MGN-HP"]["drift_mean"]),
            "Drift_HNN": float(ro["HNN"]["drift_mean"]),
        })
    except Exception as e:
        print(f"[WARN] skip {path}: {e}", file=sys.stderr)

rows.sort(key=lambda r: (r["scenario"], r["seed"]))
fields = [
    "scenario","seed","train_grid","test_grid","train_kmax","test_kmax","train_c","test_c",
    "train_dt","test_dt","epochs","train_pairs","rollout_T","q_only","meshft_hodge_mode",
    "MSE_MeshFT-Net","MSE_MGN","MSE_MGNHP","MSE_HNN",
    "RelFinal_MeshFT-Net","RelFinal_MGN","RelFinal_MGNHP","RelFinal_HNN",
    "Drift_MeshFT-Net","Drift_MGN","Drift_MGNHP","Drift_HNN"
]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(csv_path)
PY

  echo "[OK] Compiled CSV -> ${CSV}"
}

# ----------------------- main flow -----------------------
assert_py_script
mkdir -p "${OUT_ROOT}"

SCENARIOS=(freq reso param long)

for scen in "${SCENARIOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one_case "${scen}" "${seed}"
  done
done

aggregate_csv

echo "All done. Results under: ${OUT_ROOT}"