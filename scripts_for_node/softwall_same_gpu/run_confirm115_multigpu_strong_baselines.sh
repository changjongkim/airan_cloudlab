#!/usr/bin/env bash

# Frozen same-budget multi-GPU strong-baseline campaign.

set -euo pipefail

cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab

job=${SLURM_JOB_ID:?run inside an allocation}
protocol="results/softwall_multigpu/confirm115_multigpu_strong_baselines_protocol.json"

python3 - "$protocol" "$job" \
    scripts_for_node/softwall_same_gpu/run_confirm115_multigpu_strong_baselines.sh \
    scripts_for_node/softwall_same_gpu/run_multigpu_softwall_arm.sh \
    scripts_for_node/softwall_same_gpu/four_cell_trace_baseline_controller.py \
    scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py \
    scripts_for_node/softwall_same_gpu/multigpu_p2p_nrx_worker.py \
    scripts_for_node/softwall_same_gpu/trace_qwen_worker.py \
    scripts_for_node/softwall_same_gpu/analyze_confirm115_multigpu_strong_baselines.py \
    scripts_for_node/task1/isca_v2/dart_runtime.py \
    scripts_for_node/task1/isca_v2/cuda_ipc_channel.py <<'PY'
import hashlib, json, sys
from pathlib import Path

out=Path(sys.argv[1]); job=sys.argv[2]; paths=[Path(value) for value in sys.argv[3:]]
root=Path.cwd()
arms=[
 {"name":"static_s1","system":"static","payload_seed":22400051,"channel_seed":22401000},
 {"name":"wc_s1_a","system":"work_conserving","payload_seed":22400051,"channel_seed":22401000},
 {"name":"sw_s1_a","system":"softwall","payload_seed":22400051,"channel_seed":22401000},
 {"name":"sw_s1_b","system":"softwall","payload_seed":22400051,"channel_seed":22401000},
 {"name":"wc_s1_b","system":"work_conserving","payload_seed":22400051,"channel_seed":22401000},
 {"name":"static_s2","system":"static","payload_seed":22500132,"channel_seed":23120000},
 {"name":"sw_s2_a","system":"softwall","payload_seed":22500132,"channel_seed":23120000},
 {"name":"wc_s2_a","system":"work_conserving","payload_seed":22500132,"channel_seed":23120000},
 {"name":"wc_s2_b","system":"work_conserving","payload_seed":22500132,"channel_seed":23120000},
 {"name":"sw_s2_b","system":"softwall","payload_seed":22500132,"channel_seed":23120000},
]
for arm in arms:
    arm["label"]=f"c115_{arm['name']}_j{job}"
value={
 "schema":"softwall-confirm115-multigpu-strong-baselines-protocol-v1",
 "status":"frozen-before-run", "job":job,
 "question":"With the same two-GPU placement, does atomic recovery-credit exchange improve timely Qwen value over strong safe baselines while preserving radio parity?",
 "placement":{"gpu0":"four-cell Aerial + local IPC NRx + Qwen","gpu1":"remote P2P NRx"},
 "iterations":340, "warmup":20,
 "mode":{"period_ms":180,"deadline_ms":155,"nrx_bound_ms":45,"conv_bound_ms":12,"guard_ms":2,"fault_every":5,"fault_pattern":"mixed"},
 "trace":{"path":"data/public/burstgpt/softwall_burst60_prefill_trace.json","sha256":"da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d"},
 "arms":arms,
 "radio_parity_groups":{"seed1":[x["name"] for x in arms[:5]],"seed2":[x["name"] for x in arms[5:]]},
 "paired_comparisons":[
   {"name":"seed1_work_vs_softwall","softwall":["sw_s1_a","sw_s1_b"],"work_conserving":["wc_s1_a","wc_s1_b"]},
   {"name":"seed2_work_vs_softwall","softwall":["sw_s2_a","sw_s2_b"],"work_conserving":["wc_s2_a","wc_s2_b"]},
 ],
 "primary_gate":"Every arm passes safety and provenance, radio structural signatures match within seed, and both paired comparisons have >=2% timely-token gain with source-second bootstrap 95% CI lower bound >0.",
 "source_sha256_before_run":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
}
trace=root/value["trace"]["path"]
if hashlib.sha256(trace.read_bytes()).hexdigest()!=value["trace"]["sha256"]:
    raise SystemExit("trace hash mismatch before freeze")
out.parent.mkdir(parents=True,exist_ok=True)
tmp=out.with_suffix(out.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(out)
PY

run_arm() {
    local name=$1 system=$2 payload_seed=$3 channel_seed=$4
    local label="c115_${name}_j${job}"
    echo "confirm115 arm begin: $name ($system)"
    SOFTWALL_MGPU_LABEL="$label" \
    SOFTWALL_MGPU_SYSTEM="$system" \
    SOFTWALL_MGPU_ITERATIONS=340 \
    SOFTWALL_MGPU_WARMUP=20 \
    SOFTWALL_MGPU_PAYLOAD_SEED="$payload_seed" \
    SOFTWALL_MGPU_CHANNEL_SEED="$channel_seed" \
        bash scripts_for_node/softwall_same_gpu/run_multigpu_softwall_arm.sh
    echo "confirm115 arm end: $name"
}

run_arm static_s1 static 22400051 22401000
run_arm wc_s1_a work_conserving 22400051 22401000
run_arm sw_s1_a softwall 22400051 22401000
run_arm sw_s1_b softwall 22400051 22401000
run_arm wc_s1_b work_conserving 22400051 22401000

run_arm static_s2 static 22500132 23120000
run_arm sw_s2_a softwall 22500132 23120000
run_arm wc_s2_a work_conserving 22500132 23120000
run_arm wc_s2_b work_conserving 22500132 23120000
run_arm sw_s2_b softwall 22500132 23120000

python3 scripts_for_node/softwall_same_gpu/analyze_confirm115_multigpu_strong_baselines.py

