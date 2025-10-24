#!/usr/bin/env bash
# Sweep data-size vs accuracy (grid / Delaunay, canonical momentum dataset)


set -Eeuo pipefail

# ---------- Path to your Python benchmark script ----------
SCRIPT="${SCRIPT:-analytic_wave_bench.py}"  # Match the file name of the benchmark script
[[ -f "$SCRIPT" ]] || { echo "ERROR: SCRIPT not found: $SCRIPT"; exit 1; }

# ---------- General knobs ----------
PYTHON="${PYTHON:-python3}"
PROGRESS="${PROGRESS:-bars}"           # bars | none
DEVICE="${DEVICE:-auto}"               # auto | cuda | cpu
OUTROOT="${OUTROOT:-runs/analytic_bench}"

STAMP="$(date +%Y%m%d_%H%M%S)"

# ---------- Mesh & resolution ----------
MESHES=(${MESHES:-grid delaunay})      # "grid" and/or "delaunay"
GRIDS=(${GRIDS:-"32 32"})              # for grid: items are "NX NY"
NPOINTS_LIST=(${NPOINTS_LIST:-1024})   # for delaunay: number of points (defaults to NX*NY if <=0)

Lx="${Lx:-1.0}"
Ly="${Ly:-1.0}"

# ---------- Dynamics ----------
DTS=(${DTS:-0.002})                    # supports multiple dt values
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-8}"
VAL_SIZE="${VAL_SIZE:-256}"
KMAX="${KMAX:-4}"

# ---------- Data sweeps ----------
TRAIN_SIZES="${TRAIN_SIZES:-32,128,256,512,1000,2000}"   # comma-separated
MISS_RATIOS="${MISS_RATIOS:-0.0}"          # comma-separated
SEEDS=(${SEEDS:-0 1 2 3 4})
MISS_MODE="${MISS_MODE:-random}"                         # random | grid
GRID_STRIDE="${GRID_STRIDE:-2}"

# ---------- Model / physics ----------
STATE_MODE="${STATE_MODE:-canonical}"                 # canonical | velocity (model side)
DATA_STATE_MODE="${DATA_STATE_MODE:-canonical}"       # canonical | velocity (data side)
C_SPEED="${C_SPEED:-1.0}"                             # theory Hodge uses W = (c_speed^2) * V1inv (for evaluation fairness)
C_WAVE="${C_WAVE:-}"                                  # analytic wave speed (if empty, equals C_SPEED)
NORMALIZE_HODGE="${NORMALIZE_HODGE:-0}"

# Hodge for MeshFT-Net only (MGN-HP no longer uses any Hodge)
MESHFT_HODGE_MODE="${MESHFT_HODGE_MODE:-learn_geom}"  # learn | learn_geom | theory
MESHFT_W_STRUCTURE="${MESHFT_W_STRUCTURE:-diag}"      # diag | offdiag
MESHFT_USE_SPEED_SCALAR="${MESHFT_USE_SPEED_SCALAR:-0}"
OFFDIAG_INIT="${OFFDIAG_INIT:--6.0}"                  # initial value for offdiag (log-gamma)

# Hamiltonian penalty for MGN-HP (EnergyNet-based; no Hodge/DEC required)
LAMBDA_HAM="${LAMBDA_HAM:-0.1}"

# MGN capacity
MGN_HIDDEN="${MGN_HIDDEN:-64}"
MGN_LAYERS="${MGN_LAYERS:-4}"

# ---------- HNN toggle/capacity ----------
HNN_ENABLE="${HNN_ENABLE:-1}"          # 1: run HNN branch (canonical only), 0: skip
HNN_HIDDEN="${HNN_HIDDEN:-64}"
HNN_LAYERS="${HNN_LAYERS:-4}"

# ---------- Plotting ----------
MAKE_PLOTS="${MAKE_PLOTS:-1}"                          # save figures when 1
PLOT_EXT="${PLOT_EXT:-pdf}"                            # pdf | svg | png

# ---------- Energy trace (post-run, representative settings) ----------
ENERGY_RUN="${ENERGY_RUN:-1}"                  # when 1, do the extra energy-trace runs at the end
ENERGY_TRAIN_SIZE="${ENERGY_TRAIN_SIZE:-8000}" # "data abundant"
ENERGY_ROLLOUT_T="${ENERGY_ROLLOUT_T:-500}"    # long-enough to see drift trend
ENERGY_SEED="${ENERGY_SEED:-0}"                # single seed
ENERGY_EPOCHS="${ENERGY_EPOCHS:-$EPOCHS}"      # reuse EPOCHS unless overridden

# ---------- Device autodetect ----------
if [[ "${DEVICE}" == "auto" ]]; then
  if "${PYTHON}" - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then DEVFLAG=(--device cuda)
  else DEVFLAG=(--device cpu)
  fi
else
  DEVFLAG=(--device "${DEVICE}")
fi

# tiny helper
_contains() { local x="$1"; shift; for e in "$@"; do [[ "$e" == "$x" ]] && return 0; done; return 1; }

# ---------- Helper: run one sweep ----------
run_sweep() {
  local mesh="$1" nx="$2" ny="$3" npts="$4" dt="$5"
  local outdir="${OUTROOT}/wave_analytic_test_${mesh}_g${nx}x${ny}_dt${dt}_${STAMP}"
  local csv="${outdir}/results.csv"
  mkdir -p "${outdir}"

  echo ""
  echo ">>> RUN mesh=${mesh} grid=${nx}x${ny} npts=${npts} dt=${dt}"
  echo "    outdir=${outdir}"
  echo "    seeds: ${SEEDS[*]}"
  echo "    train_sizes: ${TRAIN_SIZES} ; miss_ratios: ${MISS_RATIOS}"
  echo "    HNN: enable=${HNN_ENABLE} (hidden=${HNN_HIDDEN}, layers=${HNN_LAYERS})"

  # Common arguments (kept in an array)
  common=(
    --out_dir "${outdir}"
    --out_csv "${csv}"
    --Lx "${Lx}" --Ly "${Ly}"
    --dt "${dt}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH}"
    --val_size "${VAL_SIZE}"
    --kmax "${KMAX}"
    --progress "${PROGRESS}"
    --sweep_train_sizes "${TRAIN_SIZES}"
    --sweep_miss_ratios "${MISS_RATIOS}"
    --miss_mode "${MISS_MODE}"
    --grid_stride "${GRID_STRIDE}"
    --mgn_hidden "${MGN_HIDDEN}"
    --mgn_layers "${MGN_LAYERS}"
    --lam_ham "${LAMBDA_HAM}"
    --use_weighted_loss 1
    --normalize_hodge "${NORMALIZE_HODGE}"
    --state_mode "${STATE_MODE}"
    --data_state_mode "${DATA_STATE_MODE}"
    --meshft_hodge_mode "${MESHFT_HODGE_MODE}"
    --meshft_w_structure "${MESHFT_W_STRUCTURE}"
    --offdiag_init "${OFFDIAG_INIT}"
    --c_speed "${C_SPEED}"
    --rollout_T 100
    # --- HNN branch (canonical only) ---
    --hnn_enable "${HNN_ENABLE}"
    --hnn_hidden "${HNN_HIDDEN}"
    --hnn_layers "${HNN_LAYERS}"
  )

  # seeds are variable-length
  common+=( --seeds "${SEEDS[@]}" )

  # Analytic wave speed (only if explicitly provided)
  if [[ -n "${C_WAVE}" ]]; then
    common+=( --c_wave "${C_WAVE}" )
  fi

  # Mesh selection
  if [[ "${mesh}" == "grid" ]]; then
    common+=( --mesh grid --grid "${nx}" "${ny}" )
  else
    common+=( --mesh delaunay --npoints "${npts}" )
  fi

  # Save plots
  if [[ "${MAKE_PLOTS}" == "1" ]]; then
    common+=( --make_plots --plot_ext "${PLOT_EXT}" )
  fi

  # Run
  "${PYTHON}" "${SCRIPT}" "${common[@]}" "${DEVFLAG[@]}"

  echo "==> Done: CSV ${csv}"
  if [[ "${MAKE_PLOTS}" == "1" ]]; then
    echo "==> Plots saved under: ${outdir}"
  fi
}

# ---------- Helper: representative energy-trace run (no missing, large train) ----------
run_energy_trace() {
  local mesh="$1" nx="$2" ny="$3" npts="$4" dt="$5"
  local outdir="${OUTROOT}/energy_${mesh}_g${nx}x${ny}_dt${dt}_${STAMP}"
  local csv="${outdir}/results.csv"
  mkdir -p "${outdir}"

  echo ""
  echo ">>> ENERGY TRACE (representative) mesh=${mesh} grid=${nx}x${ny} npts=${npts} dt=${dt}"
  echo "    train_size=${ENERGY_TRAIN_SIZE}, miss_ratio=0.0, seed=${ENERGY_SEED}, rollout_T=${ENERGY_ROLLOUT_T}"
  echo "    HNN: enable=${HNN_ENABLE} (hidden=${HNN_HIDDEN}, layers=${HNN_LAYERS})"
  echo "    outdir=${outdir}"

  args=(
    --out_dir "${outdir}"
    --out_csv "${csv}"
    --Lx "${Lx}" --Ly "${Ly}"
    --dt "${dt}"
    --epochs "${ENERGY_EPOCHS}"
    --batch_size "${BATCH}"
    --train_size "${ENERGY_TRAIN_SIZE}"
    --val_size "${VAL_SIZE}"
    --kmax "${KMAX}"
    --progress "${PROGRESS}"
    --miss_ratio 0.0
    --mgn_hidden "${MGN_HIDDEN}"
    --mgn_layers "${MGN_LAYERS}"
    --lam_ham "${LAMBDA_HAM}"
    --use_weighted_loss 1
    --normalize_hodge "${NORMALIZE_HODGE}"
    --state_mode "${STATE_MODE}"
    --data_state_mode "${DATA_STATE_MODE}"
    --meshft_hodge_mode "${MESHFT_HODGE_MODE}"
    --meshft_w_structure "${MESHFT_W_STRUCTURE}"
    --offdiag_init "${OFFDIAG_INIT}"
    --c_speed "${C_SPEED}"
    --rollout_T "${ENERGY_ROLLOUT_T}"
    --save_energy_csv 1
    --energy_csv_dir "${outdir}/energy_traces"
    --seed "${ENERGY_SEED}"
    # --- HNN branch (canonical only) ---
    --hnn_enable "${HNN_ENABLE}"
    --hnn_hidden "${HNN_HIDDEN}"
    --hnn_layers "${HNN_LAYERS}"
  )
  if [[ -n "${C_WAVE}" ]]; then args+=( --c_wave "${C_WAVE}" ); fi
  if [[ "${mesh}" == "grid" ]]; then
    args+=( --mesh grid --grid "${nx}" "${ny}" )
  else
    args+=( --mesh delaunay --npoints "${npts}" )
  fi

  "${PYTHON}" "${SCRIPT}" "${args[@]}" "${DEVFLAG[@]}"

  echo "==> Energy time-series CSVs saved under: ${outdir}/energy_traces"
}

# ---------- Main sweep loop ----------
for mesh in "${MESHES[@]}"; do
  for grid_pair in "${GRIDS[@]}"; do
    read -r NX NY <<< "${grid_pair}"
    local_default_npts=$(( NX * NY ))

    for dt in "${DTS[@]}"; do
      if [[ "${mesh}" == "grid" ]]; then
        run_sweep "grid" "${NX}" "${NY}" "${local_default_npts}" "${dt}"
      else
        for npts in "${NPOINTS_LIST[@]}"; do
          if [[ "${npts}" -le 0 ]]; then npts="${local_default_npts}"; fi
          run_sweep "delaunay" "${NX}" "${NY}" "${npts}" "${dt}"
        done
      fi
    done
  done
done

echo "All sweeps finished."

# ---------- Post: representative energy-trace runs (once per mesh) ----------
if [[ "${ENERGY_RUN}" == "1" ]]; then
  echo ""
  echo ">>> Starting representative energy-trace runs..."
  # Use the first grid pair & first dt as "representative"
  first_grid_pair="${GRIDS[0]}"
  read -r ENX ENY <<< "${first_grid_pair}"
  EDT="${DTS[0]}"

  # Default npts = NX*NY if not provided or <=0
  ENPTS="${NPOINTS_LIST[0]:-0}"
  if [[ -z "${ENPTS}" || "${ENPTS}" -le 0 ]]; then ENPTS=$(( ENX * ENY )); fi

  if _contains "grid" "${MESHES[@]}"; then
    run_energy_trace "grid" "${ENX}" "${ENY}" "$(( ENX * ENY ))" "${EDT}"
  fi
  if _contains "delaunay" "${MESHES[@]}"; then
    run_energy_trace "delaunay" "${ENX}" "${ENY}" "${ENPTS}" "${EDT}"
  fi
  echo "Representative energy-trace runs finished."
fi