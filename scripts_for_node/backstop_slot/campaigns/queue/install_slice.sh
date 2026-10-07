#!/usr/bin/env bash
# Install the SM slice option (default off) into the script directory.  The originals are in drafts/slice_backup;
# check_slice.sh restores them if the smoke test fails.
set -e
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
S=scripts_for_node/backstop_slot; D=run_state/backstop_slot/drafts/slice
for f in slot_state.py qwen_units.py ai_worker2.py controller5.py make_config.py; do
  cmp -s $S/$f run_state/backstop_slot/drafts/slice_backup/$f || { echo "$f changed since the backup: not installing"; exit 1; }
done
for f in slot_state.py qwen_units.py ai_worker2.py controller5.py make_config.py; do
  cp $D/$f $S/.$f.new && mv $S/.$f.new $S/$f
done
echo INSTALLED
