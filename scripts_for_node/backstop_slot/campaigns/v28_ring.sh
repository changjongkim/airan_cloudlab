#!/usr/bin/env bash
# v28: the headline condition with a replay of 1,000 slots per cell (2.5 s) instead of 512 (1.28 s).  A replay of
# 128 slots had biased the results (README 4.13); this checks that 512 slots are enough.  The data set has 1,000
# slots for the two-user cells; the single-user cells repeat their 512 slots.  Everything else is v14h.sh full.
#   usage: bash v28_ring.sh      (env: SEEDS, POLS;  tag fr, 20 s per run)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
( export SEEDS="${SEEDS:-1 2 3}" PERIODS=8000
  cond fr 16 "32" "--weak-fraction 0.25 --ring 1000" "${POLS:-n wd s10 p30 p70}" )
echo V28_RING_DONE
