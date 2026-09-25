#!/usr/bin/env bash

# Run the complete SoftWall Python validation in the same GPU-enabled
# environments used by the physical campaigns.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

raw="$SOFTWALL_ROOT/results/softwall_multigpu/raw"
mkdir -p "$raw"
job=${SLURM_JOB_ID}

modules=()
for path in "$SOFTWALL_SCRIPTS"/test_*.py; do
    name=$(basename "$path" .py)
    [[ "$name" == "test_trace_qwen_worker" ]] && continue
    modules+=("$name")
done

srun --overlap --gpus=4 shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_TASK1:/softwall_task1" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    --env=CUDA_VISIBLE_DEVICES=0 bash -c \
    "cd /softwall && python3 -m unittest -v ${modules[*]}" \
    2>&1 | tee "$raw/envelope_v11_aerial_tests_job${job}_compute.log"

srun --overlap --gpus=4 shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONNOUSERSITE=1 \
    --env=PYTHONPATH=/softwall_runtime/python:/softwall \
    --env=HF_HOME=/softwall_runtime/cache/huggingface \
    --env=TMPDIR=/softwall_runtime/tmp \
    --env=HF_HUB_DISABLE_XET=1 \
    --env=CUDA_VISIBLE_DEVICES=0 bash -c \
    "cd /softwall && python3 -m unittest -v test_trace_qwen_worker" \
    2>&1 | tee "$raw/envelope_v11_qwen_tests_job${job}_compute.log"

srun --overlap --gpus=4 shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_TASK1:/softwall_task1" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    --env=CUDA_VISIBLE_DEVICES=0 bash -c \
    "cd /softwall_task1/isca_v2 && python3 -m unittest discover -s . -p 'test_*.py' -v" \
    2>&1 | tee "$raw/envelope_v11_dart_tests_job${job}_compute.log"

python3 - "$raw/envelope_v11_validation_environment_job${job}.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
value = {
    "schema": "softwall-envelope-v11-validation-environment-v1",
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "status": "all-three-suites-completed",
}
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n")
temporary.replace(output)
PY
