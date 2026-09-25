#!/usr/bin/env python3
"""Non-mutating, post hoc saturation audit of the frozen Confirm55 raw trace."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty observation set")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def analyze(raw: dict) -> dict:
    n = raw["iterations"]
    if n < 2 or raw["cells"] != 2:
        raise ValueError("requires at least two two-cell releases")
    by_index = collections.defaultdict(list)
    for row in raw["records"]:
        by_index[row["index"]].append(row)
    if set(by_index) != set(range(n)) or any(
        sorted(row["cell"] for row in by_index[i]) != [0, 1] for i in range(n)
    ):
        raise ValueError("missing or duplicated cell records")
    if sum(not row["admitted"] for row in raw["records"]) != raw["admission_rejections"]:
        raise ValueError("admission summary disagrees with records")
    period_ns = round(raw["period_ms"] * 1e6)
    radio_tail_ms = []
    pair_nrx_tail_ms = []
    inter_release_slack_ms = []
    nrx_cutoff_slack_ms = []
    fallback_host_decision_to_precommit_ms = []
    fallback_non_gpu_residual_ms = []
    fallback_host_decisions_after_cutoff = 0
    fallback_precommit_path_over_configured_25ms = 0
    releases_all_nrx_before_first_cutoff = 0
    fallback_count = collections.Counter()
    for i in range(n):
        rows = by_index[i]
        release_ns = rows[0]["release_ns"]
        if rows[1]["release_ns"] != release_ns:
            raise ValueError("cells not released simultaneously")
        completed_ns = max(row["completed_ns"] for row in rows)
        radio_tail_ms.append((completed_ns - release_ns) / 1e6)
        # Frozen Confirm55 rows predate the explicit nrx_observed_ns field.
        # nrx_response_ms was recorded as (finish_ns-release_ns)/1e6, so
        # reconstruct the same host observation to nanosecond precision.
        observed_nrx = []
        for row in rows:
            if not row["admitted"]:
                continue
            response_ms = row["nrx_response_ms"]
            if response_ms is None:
                raise ValueError("admitted NRx lacks response timestamp")
            observed = row["release_ns"] + round(response_ms * 1e6)
            if row["commit_kind"] == "nrx" and observed != row["completed_ns"]:
                raise ValueError("reconstructed NRx timestamp disagrees with commit")
            observed_nrx.append((row, observed))
        nrx_cutoff_slack_ms.extend(
            (row["cutoff_ns"] - observed) / 1e6
            for row, observed in observed_nrx
        )
        if len(observed_nrx) == 2:
            pair_nrx_tail_ms.append(
                (max(observed for _, observed in observed_nrx) - release_ns) / 1e6
            )
        if len(observed_nrx) == 2 and max(observed for _, observed in observed_nrx) < min(
            row["cutoff_ns"] for row in rows
        ):
            releases_all_nrx_before_first_cutoff += 1
        fallback_count[sum(bool(row["fallback"]) for row in rows)] += 1
        for row in rows:
            if not row["fallback"]:
                continue
            # The legacy field name says "actual start", but the controller
            # records a host timestamp *before* start_fallback/run_conventional.
            # It is not a GPU kernel-start timestamp.
            host_ns = row["fallback_actual_start_ns"]
            gpu_ms = row["conventional_gpu_ms"]
            if host_ns is None or gpu_ms is None or row["completed_ns"] < host_ns:
                raise ValueError("incomplete fallback host/GPU timing")
            elapsed_ms = (row["completed_ns"] - host_ns) / 1e6
            fallback_host_decision_to_precommit_ms.append(elapsed_ms)
            fallback_non_gpu_residual_ms.append(elapsed_ms - gpu_ms)
            fallback_host_decisions_after_cutoff += host_ns > row["cutoff_ns"]
            fallback_precommit_path_over_configured_25ms += elapsed_ms > raw["conv_bound_ms"]
        if i < n - 1:
            if by_index[i + 1][0]["release_ns"] != release_ns + period_ns:
                raise ValueError("release cadence changed")
            inter_release_slack_ms.append((release_ns + period_ns - completed_ns) / 1e6)
    if sum(k * count for k, count in fallback_count.items()) != raw["fallbacks"]:
        raise ValueError("fallback summary disagrees with records")
    if len(fallback_host_decision_to_precommit_ms) != raw["fallbacks"]:
        raise ValueError("fallback timing count disagrees with records")
    ai_by_index = collections.Counter()
    ai_admission_after_radio_ms = []
    ai_admissions_before_radio_complete = 0
    for row in raw["background_records"]:
        i = row["release_index"]
        if i not in range(n - 1):
            raise ValueError("background unit outside inter-release interval")
        ai_by_index[i] += 1
        radio_completed_ns = max(cell["completed_ns"] for cell in by_index[i])
        offset_ms = (row["admitted_ns"] - radio_completed_ns) / 1e6
        ai_admission_after_radio_ms.append(offset_ms)
        ai_admissions_before_radio_complete += row["admitted_ns"] < radio_completed_ns
    if len(raw["background_records"]) != raw["background_units"]:
        raise ValueError("background summary disagrees with records")
    ai_per_interval = [ai_by_index[i] for i in range(n - 1)]
    return {
        "schema": "softwall-confirm55-novelty-gap-v1",
        "experiment_job": raw["slurm_job_id"],
        "host": raw["host"],
        "releases": n,
        "period_ms": raw["period_ms"],
        "deadline_ms": raw["deadline_ms"],
        "admission_rejections": raw["admission_rejections"],
        "fallbacks_per_release": {str(i): fallback_count[i] for i in range(3)},
        "fallback_host_decisions_after_cutoff": fallback_host_decisions_after_cutoff,
        "fallback_precommit_path_over_configured_25ms": fallback_precommit_path_over_configured_25ms,
        "fallback_host_decision_to_precommit_ms": summarize(fallback_host_decision_to_precommit_ms),
        "fallback_non_gpu_residual_ms": summarize(fallback_non_gpu_residual_ms),
        "fallback_timestamp_semantics": "fallback_actual_start_ns is a pre-run host decision timestamp; completed_ns is sampled before the commit call returns; neither gives GPU kernel start or actual commit completion",
        "radio_tail_ms": summarize(radio_tail_ms),
        "pair_nrx_tail_ms": summarize(pair_nrx_tail_ms),
        "pair_nrx_tail_semantics": "last host-observed NRx PHY finish before complete_nrx returns; not an after-commit bound",
        "nrx_cutoff_slack_ms": summarize(nrx_cutoff_slack_ms),
        "releases_all_nrx_before_first_cutoff": releases_all_nrx_before_first_cutoff,
        "post_radio_to_next_release_ms": summarize(inter_release_slack_ms),
        "ai_units_per_interval": summarize(ai_per_interval),
        "intervals_without_ai": sum(count == 0 for count in ai_per_interval),
        "total_ai_units": raw["background_units"],
        "ai_admissions_before_radio_complete": ai_admissions_before_radio_complete,
        "ai_admission_after_radio_ms": summarize(ai_admission_after_radio_ms),
        "evidence_scope": "post-hoc descriptive saturation audit; no policy comparison or GPU overlap proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_bytes = args.controller.read_bytes()
    result = analyze(json.loads(raw_bytes))
    result["controller_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
