#!/usr/bin/env bash
# Check the smoke test of the SM slice; restore the original files if it fails.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
Q=run_state/backstop_slot/queue
shifter --image="$AERIAL_IMAGE" python3 $Q/check_slice.py > $Q/slice_smoke_result.txt 2>&1; rc=$?
if [ $rc -ne 0 ]; then
  S=scripts_for_node/backstop_slot; B=run_state/backstop_slot/drafts/slice_backup
  for f in slot_state.py qwen_units.py ai_worker2.py controller5.py make_config.py; do cp -p $B/$f $S/.$f.old && mv $S/.$f.old $S/$f; done
  echo "RESTORED the original files" >> $Q/slice_smoke_result.txt
  touch $Q/SKIP_SLICE  # the steps of v21_slice.sh skip themselves
fi
tail -3 $Q/slice_smoke_result.txt
exit $rc
