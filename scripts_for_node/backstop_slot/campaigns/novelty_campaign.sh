#!/usr/bin/env bash
# Novelty campaign: value-rule generality, ablations, AI-load and weak-fraction sweeps.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
seedx() { echo "--seed $((81000 + 1000 * $1)) --ring-offset $((50 * ($1 - 1)))"; }

echo "### G generality $(date +%T)"
i=0
for prof in weak_cdl_e weak_wide_snr weak_mcs4; do
  shifter_slot env CUDA_VISIBLE_DEVICES=$i \
    PYTHONPATH=/softwall_runtime/sionna_deps:/backstop_slot:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    python3 /backstop_slot/build_ul_dataset.py --profile $prof --count 512 \
    --payload-seed $((73001 + i)) --channel-seed $((74001 + i)) --out-dir "$SLOT_DATA" > "$SLOT_DATA/$prof.log" 2>&1 &
  i=$((i + 1))
done
wait
i=0
for prof in weak_rank1 weak_cdl_e weak_wide_snr weak_mcs4; do
  count=512; [[ $prof == weak_rank1 ]] && count=1024
  [[ $prof == weak_rank1 && -f "$SLOT_RESULTS/raw/cbcrc_value_j59103692.json" ]] && { i=$((i + 1)); continue; }
  shifter_slot env CUDA_VISIBLE_DEVICES=$i python3 "$SOFTWALL_ROOT/run_state/backstop_slot/probes/probe_cbcrc.py" \
    "$SLOT_RESULTS/raw/cbcrc_${prof}_j${SLURM_JOB_ID}.json" $prof $count 2>&1 | grep -E "weak TBs|failed CBs|Error" | sed "s/^/$prof: /" &
  i=$((i + 1))
done
wait

echo "### A ablation $(date +%T)"
for seed in 1 2; do
  for rate in 4 8; do
    base="$(seedx $seed) --static-mps-pct 50"
    for v in "noval:value=0" "noadmit:admit=0" "partner:lanes=partner" "failonly:start=fail_only" "arrival:start=arrival"; do
      IFS=: read -r name flags <<<"$v"
      SLOT_TAG=a${seed}r${rate}$name SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$base --nrx-flags $flags" \
        SLOT_MATRIX="c16:rescue_value:backstop_corun" $M 2>&1 | grep -E "^===|FAILED"
    done
    SLOT_TAG=a${seed}r${rate}strict SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$base" \
      SLOT_MATRIX="c16:rescue_value:backstop" $M 2>&1 | grep -E "^===|FAILED"
  done
done

echo "### L load $(date +%T)"
for seed in 1 2; do
  SLOT_TAG=l${seed}r12 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$(seedx $seed) --static-mps-pct 50" \
    SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  for rate in 4 8 12; do
    for pct in 30 70; do
      SLOT_TAG=l${seed}r${rate}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$(seedx $seed) --static-mps-pct $pct" \
        SLOT_MATRIX="c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    done
  done
done

echo "### W weak fraction $(date +%T)"
for seed in 1 2; do
  for wf in 0.25 0.75; do
    tagwf=$(echo $wf | tr -d .)
    SLOT_TAG=w${seed}f$tagwf SLOT_PERIODS=4000 SLOT_AI_RATE=8 SLOT_EXTRA="$(seedx $seed) --static-mps-pct 50 --weak-fraction $wf" \
      SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static c16:parallel_admit:static c16:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo "NOVELTY_DONE $(date +%T)"
