#!/usr/bin/env bash
# Why does the run-time neural chain decode fewer TBs than NVlabs' TensorFlow evaluation?
# Variants: LDPC iterations after the neural receiver, and an FP32 build of the engine.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
E=/softwall_runtime/engines/nv
DATA=$SOFTWALL_ROOT/run_state/backstop_slot/dataset_nv
if [ ! -f $SOFTWALL_RUNTIME/engines/nv/nrx_rt_273prb_2ue_fp32.trt ]; then
  cp $SOFTWALL_RUNTIME/engines/nv/nrx_rt_273prb_2ue.json $SOFTWALL_RUNTIME/engines/nv/nrx_rt_273prb_2ue_fp32.json
  CUDA_VISIBLE_DEVICES=0 shifter_slot trtexec --onnx=$E/nrx_rt_273prb_2ue.onnx --saveEngine=$E/nrx_rt_273prb_2ue_fp32.trt --skipInference --memPoolSize=workspace:4096 \
    --shapes=rx_slot_real:1x3276x14x4,rx_slot_imag:1x3276x14x4,h_hat_real:1x3276x2x4,h_hat_imag:1x3276x2x4 \
    > $SOFTWALL_RUNTIME/engines/nv/nrx_rt_273prb_2ue_fp32.build.log 2>&1
  tail -2 $SOFTWALL_RUNTIME/engines/nv/nrx_rt_273prb_2ue_fp32.build.log
fi
for variant in "nrx_rt_273prb_2ue 10" "nrx_rt_273prb_2ue 20" "nrx_rt_273prb_2ue 40" "nrx_rt_273prb_2ue_fp32 10" "nrx_rt_273prb_2ue_fp32 20"; do
  set -- $variant
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu.py --dataset $DATA --engine $E/$1.trt --ldpc-iterations $2 --passes 2 \
    --output $SLOT_RESULTS/raw/mu_probe_${1}_it${2}_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror|Traceback' | tail -3 \
    | python3 -c "
import sys, json
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: print(line.strip()[:300]); continue
    print('$1 it$2', {k: d[k] for k in ('conventional', 'neural', 'conventional_only', 'neural_only', 'neither', 'nvlabs_tensorflow')}, d['runtime_vs_tensorflow'], 'nrx_ms', d['nrx_ms'])
"
done
echo MU_CHECK_DONE
