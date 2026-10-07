#!/usr/bin/env bash
# Check the smoke test of v22; if it fails the later steps of v22_slice2.sh skip themselves (queue_b/SKIP_SLICE2).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
Q=run_state/backstop_slot/queue_b
shifter --image="$AERIAL_IMAGE" python3 $Q/check_slice2.py > $Q/slice2_smoke_result.txt 2>&1; rc=$?
[ $rc -ne 0 ] && touch $Q/SKIP_SLICE2
tail -3 $Q/slice2_smoke_result.txt
exit $rc
