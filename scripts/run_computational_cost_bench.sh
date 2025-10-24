#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- User-config ----------
PYTHON="${PYTHON:-python3}"
COST_BENCH="${COST_BENCH:-cost_bench.py}"        # cost benchmark script (separate file)
AB_MODULE="${AB_MODULE:-analytic_wave_bench.py}"      # original long benchmark script (path ok)

# Sweep settings (edit as needed)
SEEDS=(0 1 2)                     # e.g., (0 1 2)
BATCH=8
WARMUP=20
ITERS=80
GRID_SIZES=("32x32")          # e.g., ("32x32" "64x64")
DELAUNAY_NPOINTS=(1024)       # e.g., (1024 2048)
DT=0.002
C_SPEED=1.0
STATE_MODE="canonical"        # canonical / velocity
DATA_STATE_MODE="canonical"
MISS_RATIO=0.0
MISS_MODE="random"            # random / grid
GRID_STRIDE=2
MGN_HIDDEN=64
MGN_LAYERS=4
LAM_HAM=0.1                   # Used for MGN-HP cost (J∇H penalty on)
HNN_ENABLE=1                  # 0 to skip HNN

# Output locations
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-runs/cost_bench/cost_sweep_stab_${STAMP}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Device auto-detection (can override: DEVICE=cuda ./run_cost_bench.sh)
if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$("$PYTHON" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi

echo "== Running cost benchmarks =="
echo "  Python     : ${PYTHON}"
echo "  cost_bench : ${COST_BENCH}"
echo "  ab-module  : ${AB_MODULE}"
echo "  device     : ${DEVICE}"
echo "  out dir    : ${OUT_DIR}"
echo

# Pre-flight checks
[[ -f "${COST_BENCH}" ]] || { echo "ERROR: ${COST_BENCH} not found"; exit 1; }
[[ -f "${AB_MODULE}" ]]  || { echo "ERROR: ${AB_MODULE} not found"; exit 1; }

# ---------- Aggregated CSV across runs (with config columns) ----------
ALL_CSV="${OUT_DIR}/ALL_cost_results.csv"
HEADER_WRITTEN=0

# Stats CSV (seed-aggregated)
STATS_CSV="${OUT_DIR}/ALL_cost_stats_by_model.csv"

# Header for ALL_CSV (we extend cost_bench header by config columns)
ALL_HDR="mesh,config,seed,batch,dt,device,model,params,infer_ms,train_ms,samples_per_s,nodes_per_s,inf_peakMB,train_peakMB,gate_alpha"

ensure_all_header () {
  if [[ ${HEADER_WRITTEN} -eq 0 ]]; then
    echo "${ALL_HDR}" > "${ALL_CSV}"
    HEADER_WRITTEN=1
  fi
}

# ---------- Runner (one config) with collection ----------
# Usage: run_and_collect <mesh> <config_label> <out_csv> -- <args passed to cost_bench>
run_and_collect () {
  local mesh="$1"; shift
  local config="$1"; shift
  local out_csv="$1"; shift

  # Make a log filename based on out_csv stem
  local log_file="${LOG_DIR}/$(basename "${out_csv%.csv}").log"

  # Execute cost_bench
  set -x
  "${PYTHON}" "${COST_BENCH}" \
    --ab-module "${AB_MODULE}" \
    --device "${DEVICE}" \
    --mesh "${mesh}" \
    --dt "${DT}" \
    --batch_size "${BATCH}" \
    --c_speed "${C_SPEED}" \
    --state_mode "${STATE_MODE}" \
    --data_state_mode "${DATA_STATE_MODE}" \
    --miss_ratio "${MISS_RATIO}" \
    --miss_mode "${MISS_MODE}" \
    --grid_stride "${GRID_STRIDE}" \
    --mgn_hidden "${MGN_HIDDEN}" \
    --mgn_layers "${MGN_LAYERS}" \
    --lam_ham "${LAM_HAM}" \
    --hnn_enable "${HNN_ENABLE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --seed "${SEED}" \
    --out_csv "${out_csv}" \
    "$@" 2>&1 | tee "${log_file}"
  { set +x; } 2>/dev/null

  # Append to the global ALL_CSV with extra config columns
  if [[ -f "${out_csv}" ]]; then
    ensure_all_header
    # Prepend config columns and skip the header of each per-run CSV
    awk -F, -v OFS=',' \
        -v mesh="${mesh}" -v config="${config}" -v seed="${SEED}" \
        -v batch="${BATCH}" -v dt="${DT}" -v device="${DEVICE}" \
        'NR>1 { print mesh,config,seed,batch,dt,device,$0 }' \
        "${out_csv}" >> "${ALL_CSV}"
  fi
}

# ---------- Seed-wise sweeps ----------
# GRID runs
for SEED in "${SEEDS[@]}"; do
  for SZ in "${GRID_SIZES[@]}"; do
    IFS=x read -r NX NY <<< "${SZ}"
    OUT_CSV="${OUT_DIR}/cost_grid_${NX}x${NY}_B${BATCH}_seed${SEED}.csv"
    CFG="grid_${NX}x${NY}"
    run_and_collect "grid" "${CFG}" "${OUT_CSV}" --grid "${NX}" "${NY}"
  done
done

# DELAUNAY runs
for SEED in "${SEEDS[@]}"; do
  for NPTS in "${DELAUNAY_NPOINTS[@]}"; do
    OUT_CSV="${OUT_DIR}/cost_delaunay_${NPTS}_B${BATCH}_seed${SEED}.csv"
    CFG="delaunay_${NPTS}"
    run_and_collect "delaunay" "${CFG}" "${OUT_CSV}" --npoints "${NPTS}"
  done
done

# ---------- Aggregate across seeds into stats CSV ----------
# Groups: (mesh, config, model). Metrics: mean, std, min, max, count for each numeric column.
if [[ -f "${ALL_CSV}" ]]; then
  awk -F, -v OFS=',' '
    NR==1 { next }  # skip header

    # Columns in ALL_CSV:
    # 1 mesh, 2 config, 3 seed, 4 batch, 5 dt, 6 device,
    # 7 model, 8 params,
    # 9 infer_ms, 10 train_ms, 11 samples_per_s, 12 nodes_per_s,
    # 13 inf_peakMB, 14 train_peakMB, 15 gate_alpha

    {
      key = $1 "|" $2 "|" $7  # mesh|config|model
      cnt[key]++

      # params is constant per model/config; take the first seen
      if (!(key in params)) { params[key]=$8 }

      # init mins/max if first time
      if (cnt[key]==1) {
        min_i[key]=$9;  max_i[key]=$9;   mean_i[key]=$9;   m2_i[key]=0
        min_t[key]=$10; max_t[key]=$10;  mean_t[key]=$10;  m2_t[key]=0
        min_s[key]=$11; max_s[key]=$11;  mean_s[key]=$11;  m2_s[key]=0
        min_n[key]=$12; max_n[key]=$12;  mean_n[key]=$12;  m2_n[key]=0
        min_pi[key]=$13; max_pi[key]=$13; mean_pi[key]=$13; m2_pi[key]=0
        min_pt[key]=$14; max_pt[key]=$14; mean_pt[key]=$14; m2_pt[key]=0
        min_ga[key]=$15; max_ga[key]=$15; mean_ga[key]=$15; m2_ga[key]=0
      } else {
        # infer_ms
        vi=$9;  if (vi<min_i[key]) min_i[key]=vi; if (vi>max_i[key]) max_i[key]=vi
        di=vi-mean_i[key]; mean_i[key]+=di/cnt[key]; m2_i[key]+=di*(vi-mean_i[key])
        # train_ms
        vt=$10; if (vt<min_t[key]) min_t[key]=vt; if (vt>max_t[key]) max_t[key]=vt
        dt2=vt-mean_t[key]; mean_t[key]+=dt2/cnt[key]; m2_t[key]+=dt2*(vt-mean_t[key])
        # samples/s
        vs=$11; if (vs<min_s[key]) min_s[key]=vs; if (vs>max_s[key]) max_s[key]=vs
        ds=vs-mean_s[key]; mean_s[key]+=ds/cnt[key]; m2_s[key]+=ds*(vs-mean_s[key])
        # nodes/s
        vn=$12; if (vn<min_n[key]) min_n[key]=vn; if (vn>max_n[key]) max_n[key]=vn
        dn=vn-mean_n[key]; mean_n[key]+=dn/cnt[key]; m2_n[key]+=dn*(vn-mean_n[key])
        # inf_peakMB
        vpi=$13; if (vpi<min_pi[key]) min_pi[key]=vpi; if (vpi>max_pi[key]) max_pi[key]=vpi
        dpi=vpi-mean_pi[key]; mean_pi[key]+=dpi/cnt[key]; m2_pi[key]+=dpi*(vpi-mean_pi[key])
        # train_peakMB
        vpt=$14; if (vpt<min_pt[key]) min_pt[key]=vpt; if (vpt>max_pt[key]) max_pt[key]=vpt
        dpt=vpt-mean_pt[key]; mean_pt[key]+=dpt/cnt[key]; m2_pt[key]+=dpt*(vpt-mean_pt[key])
        # gate_alpha
        vga=$15; if (vga<min_ga[key]) min_ga[key]=vga; if (vga>max_ga[key]) max_ga[key]=vga
        dga=vga-mean_ga[key]; mean_ga[key]+=dga/cnt[key]; m2_ga[key]+=dga*(vga-mean_ga[key])
      }
    }

    END {
      print "mesh,config,model,count,params," \
            "infer_ms_mean,infer_ms_std,infer_ms_min,infer_ms_max," \
            "train_ms_mean,train_ms_std,train_ms_min,train_ms_max," \
            "samples_per_s_mean,samples_per_s_std," \
            "nodes_per_s_mean,nodes_per_s_std," \
            "inf_peakMB_mean,inf_peakMB_std," \
            "train_peakMB_mean,train_peakMB_std," \
            "gate_alpha_mean,gate_alpha_std"

      for (k in cnt) {
        n=cnt[k]
        split(k, parts, /\|/)
        mesh=parts[1]; config=parts[2]; model=parts[3]

        # standard deviations (sample)
        si=(n>1)?sqrt(m2_i[k]/(n-1)):0
        st=(n>1)?sqrt(m2_t[k]/(n-1)):0
        ss=(n>1)?sqrt(m2_s[k]/(n-1)):0
        sn=(n>1)?sqrt(m2_n[k]/(n-1)):0
        spi=(n>1)?sqrt(m2_pi[k]/(n-1)):0
        spt=(n>1)?sqrt(m2_pt[k]/(n-1)):0
        sga=(n>1)?sqrt(m2_ga[k]/(n-1)):0

        print mesh,config,model,n,params[k], \
              mean_i[k],si,min_i[k],max_i[k], \
              mean_t[k],st,min_t[k],max_t[k], \
              mean_s[k],ss, \
              mean_n[k],sn, \
              mean_pi[k],spi, \
              mean_pt[k],spt, \
              mean_ga[k],sga
      }
    }
  ' "${ALL_CSV}" > "${STATS_CSV}"
fi

echo
echo "== Done =="
echo "  Row-level CSV (all runs): ${ALL_CSV}"
echo "  Seed-aggregated stats CSV: ${STATS_CSV}"
echo "  Logs directory: ${LOG_DIR}"

# ----------- Usage examples -----------
# chmod +x run_cost_bench.sh
# ./run_cost_bench.sh
# Customize:
#   SEEDS=(0 1 2) GRID_SIZES=("32x32" "64x64") DELAUNAY_NPOINTS=(1024 2048) DEVICE=cuda ./run_cost_bench.sh