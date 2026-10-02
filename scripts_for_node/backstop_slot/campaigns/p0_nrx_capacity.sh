#!/usr/bin/env bash
# P0: NeuralRx stage profile and lane concurrency on one GPU (under MPS).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
out="$SLOT_RESULTS/raw/p0"
mkdir -p "$out"
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start
trap softwall_mps_stop EXIT
shifter_slot env CUDA_VISIBLE_DEVICES=0 python3 /backstop_slot/probe_nrx_capacity.py profile --output "$out/nrx_profile_j${SLURM_JOB_ID}.json" 2>&1 | grep -E "stage_ms|Error|Trace"
for k in 1 2 3 4 6; do
  sync="$SOFTWALL_ROOT/run_state/backstop_slot/sync_$k"; rm -rf "$sync"; mkdir -p "$sync"
  pids=()
  for i in $(seq 0 $((k - 1))); do
    shifter_slot env CUDA_VISIBLE_DEVICES=0 python3 /backstop_slot/probe_nrx_capacity.py worker --index $i --sync "$sync" --seconds 5 \
      --output "$out/nrx_lanes${k}_w${i}_j${SLURM_JOB_ID}.json" > /dev/null 2>&1 &
    pids+=($!)
  done
  until [ "$(ls $sync | grep -c ready_)" -ge "$k" ]; do sleep 0.5; done
  python3 -c "import time; print(int(time.monotonic() * 1e9) + 2000000000)" > "$sync/go.tmp" && mv "$sync/go.tmp" "$sync/go"
  wait "${pids[@]}"
  python3 - "$out" "$k" "$SLURM_JOB_ID" <<'PY'
import json, sys, glob
out, k, job = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.load(open(f)) for f in glob.glob(f"{out}/nrx_lanes{k}_w*_j{job}.json")]
tput = sum(r["count"] for r in rows) / rows[0]["seconds"]
print(f"lanes={k}: throughput {tput:.0f} TB/s, latency p50 {max(r['latency_ms_p50'] for r in rows):.2f} ms, p99 {max(r['latency_ms_p99'] for r in rows):.2f} ms")
PY
done
