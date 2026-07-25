#!/usr/bin/env bash
# Pull all sweep results JSON+logs from the CloudLab node to local laptop.
# Run from local laptop.
set -uo pipefail

REMOTE=${REMOTE:-sgkim@d8545-10s10501.wisc.cloudlab.us}
LOCAL_DIR=${LOCAL_DIR:-/Users/changjongkim/New_research/cloudlab_results}
mkdir -p "$LOCAL_DIR"

echo "syncing from $REMOTE -> $LOCAL_DIR"
rsync -avz --include='*/' --include='*.json' --include='*.log' --include='*.txt' --exclude='*' \
  "$REMOTE":/users/sgkim/cloudlab_aerial/results/ "$LOCAL_DIR"/results/

# Also sweep logs at top of cloudlab_aerial
rsync -avz "$REMOTE":/users/sgkim/cloudlab_aerial/*.log "$LOCAL_DIR"/ 2>/dev/null || true

ls -la "$LOCAL_DIR/results/" | head -20
