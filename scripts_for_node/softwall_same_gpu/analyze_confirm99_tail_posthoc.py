#!/usr/bin/env python3.11
"""Explain the frozen C99 failure without changing its gate result."""

from __future__ import annotations

import json
from pathlib import Path


root = Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
base = root / 'results/softwall_same_gpu'
frozen = json.loads((base / 'confirm99_qwen_fault_abba_job58747070.json').read_text())
assert not frozen['safety_and_coverage_pass']
prefix = 'confirm99_d_nofault_job58747070'
ran = json.loads((base / 'raw' / f'{prefix}_controller.json').read_text())
worker = json.loads((base / 'raw' / f'{prefix}_worker0.json').read_text())
violations = [row for row in ran['records'] if row['nrx_bound_violation']]
assert len(violations) == 1
row = violations[0]
release = row['release_ns']
windows = [w for w in worker['execution_windows']
           if row['nrx_dispatched_ns'] <= w['observed_ns'] <= row['nrx_observed_ns']]
assert len(windows) == 1
window = windows[0]
group = ran['records'][0:4]
ai = [item for item in ran['background_records']
      if item['release_index'] == row['index']
      and item.get('phase') == 'before_nrx_observation']
assert len(ai) == 1

def ms(timestamp_ns: int) -> float:
    return (timestamp_ns - release) / 1_000_000

report = {
    'schema': 'softwall-confirm99-first-release-tail-posthoc-v1',
    'evidence_grade': 'posthoc_timing_explanation',
    'frozen_parent_safety_pass': False,
    'violating_release': row['index'], 'cell': row['cell'],
    'declared_nrx_bound_ms': 45,
    'observed_nrx_response_ms': row['nrx_response_ms'],
    'nrx_worker_published_ms_from_release': ms(window['published_ns']),
    'nrx_controller_observed_ms_from_release': ms(row['nrx_observed_ns']),
    'worker_to_controller_observation_gap_ms':
        (row['nrx_observed_ns'] - window['published_ns']) / 1_000_000,
    'pre_observation_ai_interval_ms': [ms(ai[0]['admitted_ns']), ms(ai[0]['returned_ns'])],
    'early_mandatory_conventional_intervals_ms': [
        [ms(item['fallback_actual_start_ns']), ms(item['commit_return_ns'])]
        for item in group if not item['admitted']
    ],
    'radio_deadline_misses': ran['deadline_misses'],
    'interpretation': 'The worker published the NRx result before 17 ms, but the controller waited for an asynchronous Qwen RPC and committed three mandatory conventional cells before consuming it at 48.605 ms. The NRx45 contract is release-to-controller-observation, so this is a real frozen bound violation, not a GPU-kernel service tail to delete. The N/F/F/N AI differences remain descriptive because the safety gate failed.',
}
path = base / 'confirm99_tail_posthoc.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(path)
