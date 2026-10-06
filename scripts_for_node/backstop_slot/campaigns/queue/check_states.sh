#!/usr/bin/env bash
# Check the smoke test of the channel states; restore the original files if it fails.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
Q=run_state/backstop_slot/queue
shifter --image="$AERIAL_IMAGE" python3 $Q/check_states.py > $Q/smoke_result.txt 2>&1; rc=$?
if [ $rc -ne 0 ]; then
  S=scripts_for_node/backstop_slot; B=run_state/backstop_slot/drafts/la_states/backup
  for f in conv_worker.py nrx_lane.py make_config.py; do cp -p $B/$f $S/.$f.old && mv $S/.$f.old $S/$f; done
  echo "RESTORED the original files" >> $Q/smoke_result.txt
fi
tail -3 $Q/smoke_result.txt
exit $rc
