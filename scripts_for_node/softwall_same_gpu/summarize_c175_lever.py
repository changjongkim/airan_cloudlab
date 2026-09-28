#!/usr/bin/env python3.11
"""Build a compact, deterministic audit summary of the complete C175 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    if not source.get('all_pass') or source.get('status') != 'C175_LEVER_EVALUATION_PASS':
        raise RuntimeError('C175 source is not authoritative PASS')
    rows = source['rows']
    positive = [row['flexible_recovery_gain_over_fixed_pct'] for row in rows
                if row['flexible_recovery_gain_over_fixed_pct'] > 1e-9]
    by_gpu = {}
    for gpus in (1, 2, 4):
        group = [row for row in rows if row['gpus'] == gpus]
        by_gpu[str(gpus)] = {
            'candidate_points': len(group),
            'flexible_recovery_positive_gain_points': sum(
                row['flexible_recovery_gain_over_fixed_pct'] > 1e-9 for row in group
            ),
        }
    summary = {
        'schema': 'softwall-c175-lever-summary-v1',
        'status': 'C175_LEVER_SUMMARY_PASS',
        'all_pass': True,
        'analysis_role': ('Deterministic summary of every row in the prespecified C175 development '
                          'decomposition; no row or outcome is selected.'),
        'candidate_points': len(rows),
        'greedy_within_5pct_of_fixed_oracle_points': sum(
            row['greedy_gap_to_fixed_oracle_pct'] <= 5.0 for row in rows
        ),
        'maximum_greedy_gap_pct': max(row['greedy_gap_to_fixed_oracle_pct'] for row in rows),
        'flexible_recovery_positive_gain_points': len(positive),
        'flexible_recovery_median_positive_gain_pct': statistics.median(positive),
        'by_gpu': by_gpu,
        'sionna_recovery_debt_reduction': [
            row['recovery_debt_reduction'] for row in source['summary']['sionna_debt_aware']
        ],
        'source_sha256': sha256(args.input),
    }
    expected = {
        'candidate_points': 362,
        'greedy_within_5pct_of_fixed_oracle_points': 361,
        'flexible_recovery_positive_gain_points': 203,
        'by_gpu': {
            '1': {'candidate_points': 102, 'flexible_recovery_positive_gain_points': 0},
            '2': {'candidate_points': 175, 'flexible_recovery_positive_gain_points': 145},
            '4': {'candidate_points': 85, 'flexible_recovery_positive_gain_points': 58},
        },
        'sionna_recovery_debt_reduction': [0.0, 0.0],
    }
    for key, value in expected.items():
        if summary[key] != value:
            summary['all_pass'] = False
    summary['status'] = 'C175_LEVER_SUMMARY_PASS' if summary['all_pass'] else 'C175_LEVER_SUMMARY_FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    tmp.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
