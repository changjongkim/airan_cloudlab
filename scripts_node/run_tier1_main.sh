#!/usr/bin/env bash
# Main Tier 1 orchestrator — sequential L1+AI experiments on GPU 0.
# Run AFTER L1 baselines (run_fullgpu_baseline.sh, run_mig_baselines.sh) finish.
#
# Sequence:
#   Phase 1: 4 Qwen variants on split-50-50 (3g+3g, L1+AI co-tenant)
#   Phase 4: 3 Real AI-RAN workloads on split-50-50
#   Phase 2: 4 Multi-partition layouts (M1-M4)
#   Phase 3: D1 = split-40-60 (2g L1, 3g AI) — L1-starved
#   D2 (extra): split-60-40 (4g L1, 3g AI) — L1-boosted
#
# Each leg = N=20 with DMON=1 DURATION=30 → ~100 sec/leg
# Total ≈ 13 legs × 100s + MIG reconfig overhead ≈ 25-30 min

set -uo pipefail
cd "$HOME/cloudlab_aerial"

N="${N:-20}"
DURATION="${DURATION:-30}"
CELLS="${CELLS:-20}"
GPU="${GPU:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export N DURATION CELLS GPU DATE_DIR

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

LEGS=(
  # Phase 1 — Qwen variants on 3g+3g (5/24 baseline matched)
  "split-50-50 qwen7b           phase1_qwen7b_stress"
  "split-50-50 qwen7b_prefill   phase1_qwen7b_prefill"
  "split-50-50 qwen7b_decode    phase1_qwen7b_decode"
  "split-50-50 qwen_small       phase1_qwen_small"
  # Phase 4 — Real AI-RAN
  "split-50-50 neuralrx         phase4_neuralrx"
  "split-50-50 chanpred         phase4_chanpred"
  "split-50-50 xapp_anomaly     phase4_xapp"
  # Phase 2 — Multi-partition
  "3way-balanced qwen_small     phase2_M1_3way_balanced"
  "3way-L1small  qwen_small     phase2_M2_3way_L1small"
  "3way-asym     qwen_small     phase2_M3_3way_asym"
  "4way-1L1+3AI  qwen_small     phase2_M4_4way_1L1_3AI"
  # Phase 3 — D1 (L1 starved, AI on 3g)
  "split-40-60   qwen_small     phase3_D1_L1_starved"
  # D2 — L1 boosted (4g L1, 3g AI)
  "split-60-40   qwen_small     phase3_D2_L1_boosted"
)

START_TS=$(date +%s)
for leg in "${LEGS[@]}"; do
  read -r preset ai tag <<< "$leg"
  log "===== LEG: preset=$preset AI=$ai TAG=$tag ====="
  if env GPU=$GPU N=$N DMON=1 DURATION=$DURATION CELLS=$CELLS \
       PRESET=$preset AI=$ai TAG=$tag DATE_DIR=$DATE_DIR \
       bash ./run_n20.sh; then
    log "  ✓ leg done: $tag"
  else
    log "  ✗ leg FAILED: $tag (continuing)"
  fi
done
END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
log "tier1_main DONE — elapsed ${ELAPSED}s = $((ELAPSED/60))m"
