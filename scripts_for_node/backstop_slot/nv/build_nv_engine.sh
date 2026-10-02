#!/usr/bin/env bash
# usage: build_nv_engine.sh <stem>   (expects <stem>.onnx and <stem>.json in engines/nv)
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
E=/softwall_runtime/engines/nv
stem=$1
read prbs ntx < <(python3 -c "import json;d=json.load(open('$SOFTWALL_RUNTIME/engines/nv/$stem.json'));print(d['prbs'],d['num_tx'])")
sc=$((prbs*12)); pil=$((prbs*6*2))
shifter_slot trtexec --onnx=$E/$stem.onnx --saveEngine=$E/$stem.trt --skipInference --fp16 --memPoolSize=workspace:4096 \
  --shapes=rx_slot_real:1x${sc}x14x4,rx_slot_imag:1x${sc}x14x4,h_hat_real:1x${pil}x${ntx}x4,h_hat_imag:1x${pil}x${ntx}x4 \
  > $SOFTWALL_RUNTIME/engines/nv/$stem.build.log 2>&1
tail -3 $SOFTWALL_RUNTIME/engines/nv/$stem.build.log
shifter_slot bash -c "PYTHONPATH=/backstop_slot/nv:/backstop_slot:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 python3 /backstop_slot/nv/nv_validate_trt.py --engine $E/$stem.trt" 2>&1 | grep -E '^\{|rror|Traceback' | tail -5
