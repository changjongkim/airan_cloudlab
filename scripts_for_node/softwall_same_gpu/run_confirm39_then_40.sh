#!/usr/bin/env bash

# Use one direct-owned Slurm step. Confirm39 and Confirm40 each create their
# own exclusive GPU lock and separate MPS epochs on this node.
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation
result="$SOFTWALL_ROOT/results/softwall_same_gpu"

status39=0
bash "$SOFTWALL_SCRIPTS/run_confirm39_clean_lifecycle.sh" || status39=$?

# A frozen-gate failure still leaves complete raw data and an audit, so the
# independent cold-start experiment can proceed. An incomplete run cannot.
if ! python3 - "$result/confirm39_clean_lifecycle_audit.json" "$SLURM_JOB_ID" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.load(path.open()).get('job') != sys.argv[2]:
    raise SystemExit(1)
PY
then
    echo 'Confirm39 ended before a complete lifecycle audit; Confirm40 not started' >&2
    exit 1
fi

status40=0
bash "$SOFTWALL_SCRIPTS/run_confirm40_cold_start_ablation.sh" || status40=$?

python3 - "$result" "$SLURM_JOB_ID" "$status39" "$status40" <<'PY'
import json, sys
from pathlib import Path
root, job, status39, status40 = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
manifest = {
    'schema': 'softwall-followon-run-status-v1',
    'job': job,
    'confirm39_runner_exit': status39,
    'confirm40_runner_exit': status40,
    'confirm39_audit': 'confirm39_clean_lifecycle_audit.json',
    'confirm40_audit': 'confirm40_cold_start_ablation.json',
}
(root / 'confirm39_40_run_status.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(manifest)
PY

if ((status39 != 0 || status40 != 0)); then
    exit 1
fi
