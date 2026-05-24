#!/usr/bin/env bash
# B: Cell count scaling on 3g.20gb L1 alone.
# Varies cells = 5, 10, 40 (we already have 20 from L1_alone_3g20gb).
# Run on GPU 0 (need split-60-40 MIG re-config).

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
DURATION=20

# Force split-60-40 (3g + 2g) on GPU 0 first
echo "=== Reconfigure GPU $GPU to split-60-40 ==="
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -cgi 9,14 -C >/dev/null 2>&1 || true
sleep 2
nvidia-smi mig -i $GPU -lgi 2>&1 | head -5

for CELLS in 5 10 40; do
  TAG="L1_alone_3g_cells${CELLS}"
  echo ""
  echo "===================================="
  echo "B: cells=$CELLS on 3g.20gb L1 alone N=$N"
  echo "===================================="
  GPU=$GPU N=$N PRESET=split-60-40 AI=none CELLS=$CELLS TAG=$TAG DMON=0 DURATION=$DURATION \
    bash ./run_n20.sh 2>&1 | tail -20
done

echo "DONE — Cell scaling"
