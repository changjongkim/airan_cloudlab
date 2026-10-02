#!/usr/bin/env bash
# Re-validation after the NeuralRx readback fix: capacity/latency, pass rates,
# value-rule generality, co-run bounds per AI chunk.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
echo "### capacity $(date +%T)"
bash run_state/backstop_slot/p0_nrx_capacity.sh 2>&1 | grep -E "lanes=|stage_ms"
echo "### components $(date +%T)"
shifter_slot env CUDA_VISIBLE_DEVICES=0 python3 /backstop_slot/probe_components.py --dataset $SLOT_DATA \
  --engines /softwall_runtime/engines/neural_rx_fp16_full.trt --ring 256 --repeats 2 \
  --output $SLOT_RESULTS/raw/components_fixed_j${SLURM_JOB_ID}.json > $SLOT_RESULTS/raw/components_fixed_j${SLURM_JOB_ID}.log 2>&1
echo "### value rule $(date +%T)"
i=0
for prof in weak_rank1 weak_cdl_e weak_wide_snr weak_mcs4; do
  count=512; [[ $prof == weak_rank1 ]] && count=1024
  shifter_slot env CUDA_VISIBLE_DEVICES=$i python3 "$SOFTWALL_ROOT/run_state/backstop_slot/probes/probe_cbcrc.py" \
    "$SLOT_RESULTS/raw/cbcrc_fixed_${prof}_j${SLURM_JOB_ID}.json" $prof $count 2>&1 | grep -E "weak TBs|failed CBs|Error" | sed "s/^/$prof: /" &
  i=$((i + 1))
done
wait
echo "### co-run bounds $(date +%T)"
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for cells in 16 32; do
  if [ $cells -eq 16 ]; then lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.8"; else lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.6"; fi
  base="--seed 82000 $lanes --rescue-deadline-ms 41.5 --ai-admission 0 --static-mps-pct 100 --ai-chunks 128,512,1024"
  SLOT_TAG=r0n SLOT_PERIODS=2400 SLOT_AI_RATE=12 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "FAILED"
  for c in 128 512 1024; do
    SLOT_TAG=r0f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk $c" SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "FAILED"
  done
done
echo "REVALIDATE_DONE $(date +%T)"
