#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
if [[ "${SOFTWALL_GPU_LOCK_PREHELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || { echo 'pre-held GPU0 lock is absent' >&2; exit 4; }
else
    mkdir "$gpu_lock_dir" 2>/dev/null || {
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    }
fi
cleanup() {
    softwall_mps_stop || true
    rm -f "$gpu_lock_dir/owner"
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
AI_RPC_TIMEOUT_MS=45 PERIOD_MS=90 DEADLINE_MS=80 \
ENDPOINT_TIMEOUT_MS=30 WARMUP=20 SEED=20321038 SNR_DB=-8.5 \
CHANNEL_SEED_BASE=20328000 \
    bash "$SOFTWALL_SCRIPTS/run_same_request_ipc.sh" \
    confirm37b_external_background_robust 500

python3 - "$raw" "$result" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

raw, result, job = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
prefix = 'confirm37b_external_background_robust_cap80_job' + job
controller = json.loads((raw / (prefix + '_controller.json')).read_text())
endpoint = json.loads((raw / (prefix + '_worker.json')).read_text())
background = json.loads((raw / (prefix + '_background.json')).read_text())
gates = {
    'exactly_500_releases': len(controller['records']) == 500,
    'zero_deadline_misses': controller['deadline_misses'] == 0,
    'zero_endpoint_timeouts': controller['endpoint_timeouts'] == 0,
    'background_work_positive': controller['background_units'] > 0,
    'background_units_match': (
        controller['background_units'] == background['completed_units']
    ),
    'background_budget_violations_zero': (
        controller['background_budget_violations'] == 0
    ),
    'background_release_crossings_zero': (
        controller['background_release_crossings'] == 0
    ),
    'background_faults_zero': not controller['background_faults'],
    'endpoint_units_match': endpoint['completed_units'] == 521,
    'same_node': len({controller['host'], endpoint['host'], background['host']}) == 1,
    'same_job': len({controller['slurm_job_id'], endpoint['slurm_job_id'],
                     background['slurm_job_id']}) == 1,
    'background_visible_sms_limited': background['visible_sm_count'] <= 22,
}
gates['all_pass'] = all(gates.values())
report = {
    'schema': 'softwall-external-background-smoke-v1',
    'campaign': 'confirm37b_external_background_robust',
    'job': job,
    'host': controller['host'],
    'correct': controller['correct_releases'],
    'releases': controller['iterations'],
    'deadline_misses': controller['deadline_misses'],
    'response_ms': controller['response_ms'],
    'background_units': controller['background_units'],
    'background_gpu_ms': controller['background_gpu_ms'],
    'gates': gates,
}
(result / 'confirm37b_external_background_robust.json').write_text(
    json.dumps(report, indent=2) + '\n'
)
(result / 'confirm37b_external_background_robust.md').write_text(
    '# External endpoint with bounded background NeuralRx\n\n'
    'Releases: 500; correct: {correct}; deadline misses: {miss}; '
    'background units: {bg}; response p99: {p99:.3f} ms.\n\n'
    'All frozen gates pass: {passed}.\n'.format(
        correct=controller['correct_releases'],
        miss=controller['deadline_misses'],
        bg=controller['background_units'],
        p99=controller['response_ms']['p99'],
        passed=gates['all_pass'],
    )
)
print((result / 'confirm37b_external_background_robust.md').read_text())
if not gates['all_pass']:
    raise SystemExit('confirm37b background smoke failed')
PY

cleanup
trap - EXIT INT TERM
echo 'confirm37b background smoke completed'
