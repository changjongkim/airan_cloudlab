#!/usr/bin/env bash
# MU-MIMO weak cells (two UEs, 16QAM, 273 PRBs) with the public NVlabs neural receiver as
# NeuralRx.  Antiphase vs fixed GPU shares, same NeuralRx rules, AI units, admission and
# global dispatch.  Bounds come from mu_cal.sh:
#   NRX_BOUND  NeuralRx run bound without AI      TABLE  co-run bound per AI unit size (70% share)
#   KFAIL      most failed code blocks (both UEs) that still get a NeuralRx
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
ENGINE=${ENGINE:-/softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt}
DATA=${DATA:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab/run_state/backstop_slot/dataset_nv}
ITER=${ITER:-20}        # LDPC iterations of both receivers
go() { SLOT_TAG=$1 SLOT_PERIODS=${PERIODS:-4000} SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=${CELLS:-24}; d2=${D2:-6.5}; b=${NRX_BOUND:?}; table=${TABLE:?}; gating=${GATING:-conv=128,512,1024+yield}
bc=${table#128:}; bc=${bc%%,*}
for seed in ${SEEDS:-1 2}; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --weak-profile nv_mu2 --dataset $DATA --engine $ENGINE --conv-ldpc-iterations $ITER --nrx-ldpc-iterations $ITER --ai-admission 1 --controller ${CONTROLLER:-controller3.py} ${EXTRA:-} --lanes-per-gpu ${LANES:-1} --nrx-bound-ms $b --nrx-bound-corun-ms $bc --nrx-max-cb-fail ${KFAIL:?} ${NRXFLAGS:-} --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global"
  for ref in ${REFS:-n m}; do go ${TAG:-u}${seed}${ref} 4 "$common" "c$cells:rescue_value:none"; done
  for rate in ${RATES:-16 32}; do
    go ${TAG:-u}${seed}r${rate}v4 $rate "$common --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms ${PIECE:-4.0} --ai-chunk-choice sustained --unit-gating $table/$gating ${BUDGET:-}" "c$cells:rescue_value:backstop_units"
    for pct in ${SHARES:-30 50 70}; do
      go ${TAG:-u}${seed}r${rate}s$pct $rate "$common --static-mps-pct $pct" "c$cells:rescue_value:static"
    done
    # Dynamic-share baseline: DYNS="low,high:lag_periods ..." (one pre-loaded AI worker per share).
    for dyn in ${DYNS:-}; do
      shares=${dyn%%:*}; lag=${dyn##*:}
      go ${TAG:-u}${seed}r${rate}d${shares/,/x}l$lag $rate "$common --static-mps-pct 100 --dynamic-shares $shares --dynamic-lag $lag" "c$cells:rescue_value:static"
    done
  done
done
echo MU_MAIN_DONE
