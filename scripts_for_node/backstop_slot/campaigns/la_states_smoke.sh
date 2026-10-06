#!/usr/bin/env bash
# Smoke test of the channel states (la_states.py): one state with the new code, then two states that are the same
# dataset, which must give the same radio results.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl3.sh
PERIODS=2400 bash $C lsr 0.10 "1" "n wm"
PERIODS=2400 STATES="high16 high16" PHASE=200 bash $C lss 0.10 "1" "n wm"
echo LA_STATES_SMOKE_DONE
