#!/usr/bin/env bash
# Green contexts on a compute node: do they hold under MPS and across processes, and what does the neural
# receiver lose when AI runs on other SMs?  One GPU; the receivers are gc_receivers.py, the AI stand-in gc_load.py.
#   usage: bash gc_node_probe.sh       -> results/backstop_slot/gc_probe_j<job>.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot; G=/backstop_slot/gc
LARGE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
OUT=$SLOT_RESULTS/gc_probe_j${SLURM_JOB_ID}.txt; : > $OUT
say() { echo "$*" | tee -a $OUT; }
recv() { shifter_slot "${@:2}" python3 $G/gc_receivers.py $1 $R/dataset_la_smoke $LARGE 2>&1 | grep -E "RESULT|rror|Traceback" | cut -c1-200 | tee -a $OUT; }
load() {   # load SPEC SECONDS [shifter --env options]: in the background; waits until the load is on the GPU
  rm -f $R/logs/gc_load.log
  shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONPATH=/softwall_runtime/python --env=CUDA_VISIBLE_DEVICES=0 "${@:3}" \
    python3 $G/gc_load.py $1 $2 > $R/logs/gc_load.log 2>&1 &
  LOAD_PID=$!
  for i in $(seq 1 600); do grep -q "LOAD READY" $R/logs/gc_load.log 2>/dev/null && break; sleep 0.1; done
}
endload() { wait $LOAD_PID; grep -E "RESULT|rror|Traceback" $R/logs/gc_load.log | cut -c1-200 | tee -a $OUT; }
say "## A. no MPS, one process"
for s in 0 96 80 54; do recv $s; done
export SOFTWALL_MPS_GPU=0
softwall_mps_start; trap softwall_mps_stop EXIT
say "## B. under MPS, one client"
for s in 0 80 54; do recv $s; done
say "## C. under MPS, receivers next to an AI stand-in (PyTorch matrix products)"
say "# C1 load alone (all SMs / 28 SMs in a green context)"
load 0 12; endload; load 28 12; endload
say "# C2 load at normal priority on all SMs, receivers on all SMs"
load 0 45; recv 0; endload
say "# C3 load at low MPS priority on all SMs (as Priority-max)"
load 0 45 --env=CUDA_MPS_CLIENT_PRIORITY=1; recv 0; endload
say "# C4 load at low priority with an MPS share of 26% (as a 26% cap)"
load 0 45 --env=CUDA_MPS_CLIENT_PRIORITY=1 --env=CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=26; recv 0; endload
say "# C5 load in a green context of 28 SMs, receivers on all SMs"
load 28 45; recv 0; endload
say "# C6 load in a green context of 28 SMs, receivers in a green context on the other 80 SMs"
load 28 45; recv rest:28; endload
say "# C7 as C6 with the load at low MPS priority"
load 28 45 --env=CUDA_MPS_CLIENT_PRIORITY=1; recv rest:28; endload
echo GC_NODE_PROBE_DONE | tee -a $OUT
