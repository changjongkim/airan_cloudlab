#!/usr/bin/env bash
# Stop every campaign process of this user on the node of the allocation (the node is exclusive to the job).
for round in 1 2 3; do
  for pat in "v16_long" "v14h.sh" "run_matrix.sh" "run_slot.sh" "launch_slot.py" "conv_worker.py" "nrx_lane.py" "ai_worker" "controller5.py" "controller6.py"; do
    pkill -u "$USER" -f "$pat" 2>/dev/null
  done
  sleep 2
done
pkill -u "$USER" -f nvidia-cuda-mps-control 2>/dev/null; pkill -u "$USER" -f nvidia-cuda-mps-server 2>/dev/null
sleep 3
echo "left: chains $(pgrep -u "$USER" -fc 'v16_long|run_matrix|run_slot') workers $(pgrep -u "$USER" -fc 'conv_worker|nrx_lane|ai_worker|controller5') mps $(pgrep -u "$USER" -fc nvidia-cuda-mps)"
pgrep -u "$USER" -af python3 | cut -c1-120 | head -5
