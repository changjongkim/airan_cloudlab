#!/usr/bin/env bash
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation
iterations=${SOFTWALL_HOST_ITERATIONS:-10000}
label=${SOFTWALL_HOST_LABEL:-g1b_host_staging_job${SLURM_JOB_ID}}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"; state="$SOFTWALL_ROOT/run_state/softwall_multigpu/${label}"
script="$SOFTWALL_SCRIPTS/multigpu_host_staging_gate.py"; analyzer="$SOFTWALL_SCRIPTS/analyze_multigpu_host_staging_gate.py"; channel="$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py"
protocol="$result_root/${label}_protocol.json"; owner="$result_root/${label}_owner.json"; worker="$result_root/${label}_worker.json"; result="$result_root/${label}_result.json"
tag="${label}_${BASHPID}"; mkdir -p "$result_root" "$state"; rm -f "$state"/cuda_ipc_* "$owner" "$worker" "$result"
python3 - "$protocol" "$script" "$channel" "$iterations" "$label" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,script,channel=map(Path,sys.argv[1:4]); n=int(sys.argv[4]); label=sys.argv[5]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v={"schema":"softwall-multigpu-host-staging-protocol-v1","status":"frozen-before-run","label":label,"iterations":n,"source_device":0,"destination_device":1,"mps":False,"ring_depth":1,"timing_warmup_units":min(20,n),"forward_bytes":1415232,"backward_bytes":314496,"source_sha256":{"/softwall/multigpu_host_staging_gate.py":h(script),"/softwall_task1/isca_v2/cuda_ipc_channel.py":h(channel)}}
tmp=out.with_suffix(out.suffix+".tmp");tmp.write_text(json.dumps(v,indent=2));tmp.replace(out)
PY
shifter_gpu(){ shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_SCRIPTS:/softwall" --volume="$SOFTWALL_TASK1:/softwall_task1" --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONPATH=/softwall:/softwall_task1 --env=CUDA_VISIBLE_DEVICES=0,1 "$@"; }
owner_pid=""; worker_pid=""; cleanup(){ [[ -z "$owner_pid" ]]||kill "$owner_pid" 2>/dev/null||true; [[ -z "$worker_pid" ]]||kill "$worker_pid" 2>/dev/null||true; }; trap cleanup EXIT INT TERM
shifter_gpu python3 /softwall/multigpu_host_staging_gate.py --role owner --tag "$tag" --ipc-dir "$state" --iterations "$iterations" --output "$owner" & owner_pid=$!
info="$state/cuda_ipc_${tag}.info"; for _ in {1..2400}; do [[ -f "$info" ]]&&break; kill -0 "$owner_pid" 2>/dev/null||{ echo "owner exited" >&2; exit 1; }; sleep .05; done; [[ -f "$info" ]]||exit 1
shifter_gpu python3 /softwall/multigpu_host_staging_gate.py --role worker --tag "$tag" --ipc-dir "$state" --iterations "$iterations" --output "$worker" & worker_pid=$!
wait "$owner_pid"; owner_pid=""; wait "$worker_pid"; worker_pid=""
shifter_gpu python3 /softwall/analyze_multigpu_host_staging_gate.py --protocol "$protocol" --owner "$owner" --worker "$worker" --output "$result"
trap - EXIT INT TERM
echo "host-staging gate complete: $result"
