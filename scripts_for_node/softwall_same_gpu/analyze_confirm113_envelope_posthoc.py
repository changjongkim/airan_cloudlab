#!/usr/bin/env python3
"""Explain Confirm113 with a work-unit/recovery-credit feasibility envelope."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RESULTS = ROOT / "results/softwall_same_gpu"
RAW = RESULTS / "raw"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read(RESULTS / "confirm113_strong_baselines_protocol.json")
    confirm113 = read(RESULTS / "confirm113_strong_baselines_job58815435.json")
    trace = read(ROOT / protocol["trace"]["path"])
    cells = protocol["cells"]
    recovery_ms = protocol["conv_bound_ms"]
    max_incremental_slack_ms = (cells - 1) * recovery_ms
    bounds = {int(length): value for length, value in protocol["ai"]["bound_ms"].items()}
    counts = Counter(request["context_length"] for request in trace["requests"])
    gain_only_eligible = sum(
        count for length, count in counts.items()
        if bounds[length] <= max_incremental_slack_ms
    )

    q05_soft = read(RAW / "c110b_sw_q05_u200_s1_j58815435_controller.json")
    q05_work = read(RAW / "c110c_wc_q05_u200_s1_j58815435_controller.json")
    q05_overload_failure = read(
        RAW / "c111b_wc_q05_u3_s1_j58815435_controller.json"
    )
    result = {
        "schema": "softwall-confirm113-feasibility-envelope-posthoc-v1",
        "status": "posthoc explanation; not a new confirmatory performance result",
        "certificate_geometry": {
            "cells": cells,
            "recovery_credit_ms": recovery_ms,
            "maximum_incremental_slack_from_tail_compaction_ms": max_incremental_slack_ms,
            "derivation": "(cells - 1) * recovery_credit_ms",
            "exclusive_admission_condition": (
                "For decision-time base slack W, a unit is enabled only by compaction "
                "when W < B_AI <= W + Delta_recovery. B_AI <= Delta_recovery is a "
                "necessary zero-base-slack special case, not a sufficient dynamic condition."
            ),
        },
        "primary_trace": {
            "requests": len(trace["requests"]),
            "bucket_counts": {str(length): counts[length] for length in sorted(counts)},
            "bound_ms": {str(length): bounds[length] for length in sorted(bounds)},
            "units_no_larger_than_max_incremental_slack": gain_only_eligible,
            "eligible_fraction": gain_only_eligible / len(trace["requests"]),
            "coarse_256_or_512_units": counts[256] + counts[512],
            "coarse_fraction": (counts[256] + counts[512]) / len(trace["requests"]),
            "synthetic_slo_ms": protocol["trace"]["synthetic_slo_ms"],
            "radio_period_ms": protocol["period_ms"],
        },
        "confirm113_observation": {
            "gates": confirm113["gates"],
            "paired_comparisons": confirm113["paired_comparisons"],
            "softwall_exchange_counts": {
                name: arm["recovery_retime_count"]
                for name, arm in confirm113["arms"].items()
                if arm["system"] == "softwall"
            },
        },
        "small_unit_sensitivity": {
            "mode": "Qwen2.5-0.5B, uniform within-second arrivals, 200 ms SLO, 35 ms bound",
            "softwall": q05_soft["trace_summary"],
            "work_conserving": q05_work["trace_summary"],
            "same_timely_value": (
                q05_soft["trace_summary"]["timely_value_tokens"]
                == q05_work["trace_summary"]["timely_value_tokens"]
            ),
            "interpretation": (
                "The smaller qualified unit fits the recovery increment, but both systems "
                "completed 216 of 218 offered requests; demand pressure was absent."
            ),
        },
        "pressure_sensitivity": {
            "mode": "Qwen2.5-0.5B, uniform 3x arrivals, 200 ms SLO, candidate 30 ms bound",
            "qualified": False,
            "failure": q05_overload_failure["failure"],
            "violating_record": q05_overload_failure["background_records"][-1],
            "interpretation": (
                "Demand pressure existed, but a 43.628 ms host execution violated the "
                "candidate 30 ms contract. This point is unqualified and cannot support "
                "a throughput claim."
            ),
        },
        "conclusion": (
            "The measured no-benefit region follows from the conjunction of coarse qualified "
            "AI units, loose AI deadlines, and insufficient demand in the smaller-unit mode. "
            "SoftWall's atomic exchange is exercised and safe, but a useful performance region "
            "requires a qualified unit that fits reclaimed recovery slack while demand and SLO "
            "make earlier completion valuable."
        ),
    }
    output = RESULTS / "confirm113_feasibility_envelope_posthoc.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "max_incremental_slack_ms": max_incremental_slack_ms,
        "gain_only_eligible": gain_only_eligible,
        "requests": len(trace["requests"]),
        "coarse_fraction": result["primary_trace"]["coarse_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
