#!/usr/bin/env bash
# Cell scaling on 2g.10gb (GPU 0 split-40-60) and 4g.20gb (GPU 1 3way-asym).
# Tests HBM bandwidth saturation point per partition.
# cells = 5, 10, 20, 40 (skip cells where redundant with existing data).
#
# Run in parallel (different GPUs).

set -uo pipefail
cd "$HOME/cloudlab_aerial"

N=5
DURATION=15
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR

# ----------------------------------------------------------------------------
# 2g.10gb on GPU 0
# ----------------------------------------------------------------------------
(
  GPU=0
  echo "=== Setup GPU $GPU to split-40-60 (3g + 2g) ==="
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi 14,9 -C >/dev/null 2>&1
  sleep 2

  for CELLS in 5 10 20 40; do
    TAG="L1_alone_2g_cells${CELLS}"
    echo ""
    echo "=== B-2g: cells=$CELLS on 2g.10gb L1 alone N=$N (GPU 0) ==="
    GPU=$GPU N=$N PRESET=split-40-60 AI=none CELLS=$CELLS TAG=$TAG DMON=0 DURATION=$DURATION \
      bash ./run_n20.sh 2>&1 | tail -8
  done
) > /tmp/cells_2g.log 2>&1 &
PID_2G=$!
echo "2g pid: $PID_2G"

sleep 8

# ----------------------------------------------------------------------------
# 4g.20gb on GPU 1
# ----------------------------------------------------------------------------
(
  GPU=1
  echo "=== Setup GPU $GPU to 3way-asym (4g + 2g + 1g) ==="
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi 5,14,19 -C >/dev/null 2>&1
  sleep 2

  for CELLS in 5 10 40; do
    TAG="L1_alone_4g_cells${CELLS}"
    echo ""
    echo "=== B-4g: cells=$CELLS on 4g.20gb L1 alone N=$N (GPU 1) ==="
    GPU=$GPU N=$N PRESET=3way-asym AI=none CELLS=$CELLS TAG=$TAG DMON=0 DURATION=$DURATION \
      bash ./run_n20.sh 2>&1 | tail -8
  done
) > /tmp/cells_4g.log 2>&1 &
PID_4G=$!
echo "4g pid: $PID_4G"

echo "Waiting for both..."
wait $PID_2G
echo "2g done"
wait $PID_4G
echo "4g done"
echo "DONE"
