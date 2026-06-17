#!/usr/bin/env bash
# MPS + NCU on no-MIG — L1 kernel DRAM/L2/SM under CUDA MPS (concurrent exec).
# Same conditions as default NCU (04) so MPS can be compared 1:1. Under MPS the
# AI runs concurrently during ncu replay, so L1's measured DRAM throughput
# reflects real co-execution contention (vs default time-slice serialization).
set -uo pipefail
IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"; HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"; AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/MPS_ncu}"
CELLS="${CELLS:-4}"; ITERS="${ITERS:-3}"; AI_DUR="${AI_DUR:-1800}"; NEURALRX_WAIT="${NEURALRX_WAIT:-75}"
NCU_CONDS="${NCU_CONDS:-alone neuralrx resnet chanpred sat_hbm sat_compute}"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor"
mkdir -p "$RESULTS_DIR"; LOG="$RESULTS_DIR/MPS_ncu.log"
ts(){ date '+%H:%M:%S'; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"; want(){ [[ " $NCU_CONDS " == *" $1 "* ]]; }
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$HANDOFF/mps_pipe_${SLURM_JOB_ID:-$$}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$HANDOFF/mps_log_${SLURM_JOB_ID:-$$}}"
start_mps(){ log "MPS start"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 3; }
stop_mps(){ log "MPS stop"; echo quit | nvidia-cuda-mps-control 2>/dev/null||true; sleep 2; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; }

profile_l1_ncu(){
  local label=$1; log "  NCU [$label]"
  shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
    --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" --workdir=/scripts \
    bash -c "ncu --target-processes all --replay-mode kernel --clock-control none --metrics $METRICS \
             --csv --log-file /out/${label}.csv python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -2 | tee -a "$LOG"
}
ai_bg_venv(){ local tag=$1 script=$2; shift 2
  ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" --env=HF_HOME="$HF_HOME" \
      --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      "$VENV/bin/python" "/experiments/$script" 0 "$AI_DUR" "$@" ) > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
  echo $! >> "$PIDS"; log "  AI(venv) [$tag] pid=$!"; }
ai_bg_base(){ local tag=$1 script=$2; shift 2
  ( shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
      --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB \
      --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      python3 "/experiments/$script" 0 "$AI_DUR" "$@" ) > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
  echo $! >> "$PIDS"; log "  AI(base) [$tag] pid=$!"; }
kill_all_ai(){ while read -r p; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null; done < "$PIDS"; : > "$PIDS"; sleep 6; }

trap 'stop_mps; kill_all_ai' EXIT INT TERM
start_mps
log "===== MPS+NCU no-MIG CELLS=$CELLS ITERS=$ITERS conds='$NCU_CONDS' ====="
if want alone;       then log "=== alone ===";       profile_l1_ncu "MPSncu_alone"; fi
if want neuralrx;    then log "=== neuralrx ===";    ai_bg_base neuralrx run_neural_rx_stress.py; sleep "$NEURALRX_WAIT"; profile_l1_ncu "MPSncu_neuralrx"; kill_all_ai; fi
if want resnet;      then log "=== resnet ===";      ai_bg_venv resnet run_resnet_stress.py 64 fp16; sleep 10; profile_l1_ncu "MPSncu_resnet"; kill_all_ai; fi
if want chanpred;    then log "=== chanpred ===";    ai_bg_venv chanpred run_channel_prediction.py; sleep 10; profile_l1_ncu "MPSncu_chanpred"; kill_all_ai; fi
if want sat_hbm;     then log "=== sat_hbm ===";     ai_bg_venv sat_hbm run_hbm_stress.py 16; sleep 10; profile_l1_ncu "MPSncu_sat_hbm"; kill_all_ai; fi
if want sat_compute; then log "=== sat_compute ==="; ai_bg_venv sat_compute run_realistic_ai_stress.py matmul 0.8; sleep 10; profile_l1_ncu "MPSncu_sat_compute"; kill_all_ai; fi
stop_mps; trap - EXIT INT TERM
log "===== DONE: $(ls "$RESULTS_DIR"/MPSncu_*.csv 2>/dev/null|wc -l) CSV ====="
