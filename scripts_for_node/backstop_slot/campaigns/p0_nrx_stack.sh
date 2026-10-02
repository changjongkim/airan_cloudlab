#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
for s in 2 4 8; do
  sc=$((3276 * s)); pl=$((4914 * s))
  eng=/softwall_runtime/engines/neural_rx_fp16_stack${s}.trt
  [ -f "$SOFTWALL_RUNTIME/engines/neural_rx_fp16_stack${s}.trt" ] && continue
  shifter_slot env CUDA_VISIBLE_DEVICES=$((s / 2 - 1)) /usr/src/tensorrt/bin/trtexec --onnx=/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx \
    --saveEngine=$eng --skipInference --fp16 --memPoolSize=workspace:4096 \
    --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw --outputIOFormats=fp32:chw,fp32:chw \
    --shapes=rx_slot_real:1x${sc}x12x4,rx_slot_imag:1x${sc}x12x4,h_hat_real:1x${pl}x1x4,h_hat_imag:1x${pl}x1x4 \
    > "$SOFTWALL_RUNTIME/engines/neural_rx_fp16_stack${s}.build.log" 2>&1 &
done
wait
ls -la $SOFTWALL_RUNTIME/engines/ | grep stack
shifter_slot env CUDA_VISIBLE_DEVICES=0 python3 /backstop_slot/probe_nrx_stack.py --tbs 256 \
  --engines 1=/softwall_runtime/engines/neural_rx_fp16_full.trt 2=/softwall_runtime/engines/neural_rx_fp16_stack2.trt \
  4=/softwall_runtime/engines/neural_rx_fp16_stack4.trt 8=/softwall_runtime/engines/neural_rx_fp16_stack8.trt \
  --output "$SLOT_RESULTS/raw/p0/nrx_stack_j${SLURM_JOB_ID}.json" 2>&1 | grep -E '^[0-9] |Error|Trace|error'
