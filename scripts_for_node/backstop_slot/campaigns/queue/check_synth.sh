#!/usr/bin/env bash
# The slots made by added noise must decode like the slots generated at the same Es/No; otherwise the steps that
# use them are skipped (queue/SKIP_SYNTH).   usage: check_synth.sh [made:generated ...]
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
Q=run_state/backstop_slot/queue
out=$Q/synth_result${1:+_$(echo $1 | tr ':' '_')}.txt
shifter --image="$AERIAL_IMAGE" python3 $Q/check_synth.py "$@" > $out 2>&1; rc=$?
[ $rc -ne 0 ] && [ $# -eq 0 ] && touch $Q/SKIP_SYNTH
tail -2 $out
exit $rc
