#!/usr/bin/env bash
# Install the channel-state extension (la_states.py and the three patched files) into the script directory.
# The originals go to drafts/la_states/backup; check_states.sh restores them if the smoke test fails.
set -e
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
S=scripts_for_node/backstop_slot; D=run_state/backstop_slot/drafts/la_states; B=$D/backup
mkdir -p $B
for f in conv_worker.py nrx_lane.py make_config.py; do
  [ -f $B/$f ] || cp -p $S/$f $B/$f
  cp $D/$f $S/.$f.new && mv $S/.$f.new $S/$f
done
cp $D/la_states.py $S/.la_states.py.new && mv $S/.la_states.py.new $S/la_states.py
echo INSTALLED
