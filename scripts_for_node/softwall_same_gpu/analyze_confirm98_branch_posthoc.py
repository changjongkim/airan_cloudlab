#!/usr/bin/env python3.11
"""Descriptive branch audit of completed C98 physical runs; not a frozen gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
BASE = ROOT / 'results/softwall_same_gpu'
AUDIT = json.loads((BASE / 'confirm98_qwen_nrx45_ai50_job58747070.json').read_text())
if not AUDIT['all_pass']:
    raise SystemExit('C98 frozen audit did not pass')

arms = []
for arm in AUDIT['arms']:
    ran = json.loads((BASE / 'raw' / f"{arm['prefix']}_controller.json").read_text())
    groups = [ran['records'][4*i:4*i+4] for i in range(ran['iterations'])]
    timely = Counter(
        item['release_index'] for item in ran['background_records']
        if item['returned_ns'] <= groups[item['release_index']][0]['release_ns']
        + 153_000_000
    )
    categories = defaultdict(list)
    for i, group in enumerate(groups):
        admitted = sum(bool(row['admitted']) for row in group)
        injected = sum(bool(row['forced_nrx_failure']) for row in group)
        categories[(admitted, injected)].append(timely[i])
    rows = []
    for (admitted, injected), values in sorted(categories.items()):
        distribution = Counter(values)
        rows.append({
            'admitted_nrx': admitted, 'injected_failures': injected,
            'releases': len(values), 'timely_ai_min': min(values),
            'timely_ai_max': max(values),
            'timely_ai_mean': sum(values) / len(values),
            'timely_ai_distribution': dict(sorted(distribution.items())),
        })
    arms.append({'name': arm['name'], 'categories': rows,
                 'seven_ai_observed_any_release': any(value >= 7 for value in timely.values())})

result = {
    'schema': 'softwall-confirm98-branch-posthoc-v1',
    'evidence_grade': 'posthoc_descriptive', 'frozen_parent_pass': True,
    'arms': arms,
    'interpretation': 'C98 has real Qwen workload pressure but no seven-job conditional-capacity witness: no release, including releases without admitted NRx, completed seven by D153. All-fail and no-fail categories have different six-unit rates, but this unpaired observational split does not establish a causal recovery cost or a policy advantage.',
}
path = BASE / 'confirm98_branch_posthoc.json'
path.write_text(json.dumps(result, indent=2) + '\n')
print(path)
