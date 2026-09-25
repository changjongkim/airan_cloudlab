#!/usr/bin/env python3
"""Summarize separate-client physical-overrun campaigns."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def optional_mean(values):
    valid = [value for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def optional_max(values):
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_cap(?P<cap>[0-9]+)_job(?P<job>[0-9]+)_ran[.]json$"
    )
    rows = []
    for path in sorted(args.raw.glob(f"{args.campaign}_r*_ran.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        ran = json.loads(path.read_text(encoding="utf-8"))
        worker_path = Path(str(path).replace("_ran.json", "_worker.json"))
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        retired_at = ran.get("endpoint_retired_at_release")
        post_retirement_misses = [
            item for item in ran["records"]
            if item["deadline_miss"]
            and retired_at is not None
            and item["index"] > retired_at
        ]
        rows.append({
            "round": int(match["round"]),
            "cap": int(match["cap"]),
            "iterations": ran["iterations"],
            "correct": ran["correct_releases"],
            "misses": ran["deadline_misses"],
            "injected": ran["injected_overruns"],
            "timeouts": ran["timeouts_detected"],
            "drained": ran["responses_drained"],
            "outstanding_at_end": ran["outstanding_at_end"],
            "retire_after_timeout": ran.get("retire_after_timeout", False),
            "retire_before_first_release": ran.get(
                "retire_before_first_release", False
            ),
            "endpoint_retired": ran.get("endpoint_retired", False),
            "endpoint_retired_at_release": retired_at,
            "post_retirement_misses": len(post_retirement_misses),
            "post_retirement_miss_offsets": [
                item["index"] - retired_at for item in post_retirement_misses
            ],
            "worst_response_ms": ran["response_ms"]["max"],
            "active_releases": ran["worker_active_releases"],
            "active_misses": ran["worker_active_deadline_misses"],
            "active_p99_ms": ran["worker_active_response_ms"]["p99"],
            "active_max_ms": ran["worker_active_response_ms"]["max"],
            "worker_units": worker["completed_units"],
            "visible_sms": worker["visible_sm_count"],
            "worker_gpu_mean_ms": worker["gpu_ms"]["mean"],
            "worker_gpu_max_ms": worker["gpu_ms"]["max"],
        })
    if not rows:
        raise SystemExit("no overrun results")
    gates = {
        "all_correct": all(x["correct"] == x["iterations"] for x in rows),
        "deadline_misses_zero": sum(x["misses"] for x in rows) == 0,
        "active_deadline_misses_zero": sum(x["active_misses"] for x in rows) == 0,
        "all_requests_timed_out": all(x["timeouts"] == x["injected"] for x in rows),
        "all_responses_drained": all(x["drained"] == x["injected"] for x in rows),
        "no_outstanding_at_end": not any(x["outstanding_at_end"] for x in rows),
        "requested_endpoints_retired": all(
            not (
                x["retire_after_timeout"]
                or x["retire_before_first_release"]
            ) or x["endpoint_retired"]
            for x in rows
        ),
        "worker_units_match": all(x["worker_units"] == x["injected"] for x in rows),
        "cap100_full_sm": all(
            x["visible_sms"] == 108 for x in rows if x["cap"] == 100
        ),
        "cap20_limited_sm": all(
            x["visible_sms"] <= 22 for x in rows if x["cap"] == 20
        ),
    }
    gates["all_pass"] = all(gates.values())
    grouped = {}
    for cap in sorted({x["cap"] for x in rows}):
        items = [x for x in rows if x["cap"] == cap]
        cap_gates = {
            "all_correct": all(x["correct"] == x["iterations"] for x in items),
            "deadline_misses_zero": sum(x["misses"] for x in items) == 0,
            "active_deadline_misses_zero": sum(x["active_misses"] for x in items) == 0,
            "all_requests_timed_out": all(x["timeouts"] == x["injected"] for x in items),
            "all_responses_drained": all(x["drained"] == x["injected"] for x in items),
            "no_outstanding_at_end": not any(x["outstanding_at_end"] for x in items),
            "requested_endpoints_retired": all(
                not (
                    x["retire_after_timeout"]
                    or x["retire_before_first_release"]
                ) or x["endpoint_retired"]
                for x in items
            ),
        }
        cap_gates["all_pass"] = all(cap_gates.values())
        grouped[str(cap)] = {
            "runs": len(items),
            "releases": sum(x["iterations"] for x in items),
            "active_releases": sum(x["active_releases"] for x in items),
            "misses": sum(x["misses"] for x in items),
            "post_retirement_misses": sum(
                x["post_retirement_misses"] for x in items
            ),
            "post_retirement_miss_offsets": sorted(
                offset
                for x in items
                for offset in x["post_retirement_miss_offsets"]
            ),
            "worst_response_ms": max(x["worst_response_ms"] for x in items),
            "active_p99_ms_mean": optional_mean(
                x["active_p99_ms"] for x in items
            ),
            "worst_active_max_ms": optional_max(
                x["active_max_ms"] for x in items
            ),
            "worker_gpu_mean_ms": optional_mean(
                x["worker_gpu_mean_ms"] for x in items
            ),
            "worker_gpu_worst_max_ms": optional_max(
                x["worker_gpu_max_ms"] for x in items
            ),
            "visible_sms": sorted({x["visible_sms"] for x in items}),
            "gates": cap_gates,
        }
    result = {
        "schema": "softwall-physical-overrun-analysis-v1",
        "campaign": args.campaign,
        "gates": gates,
        "by_cap": grouped,
        "rows": rows,
    }
    lines = [
        "# SoftWall physical optional-client overrun",
        "",
        "| cap | runs | releases | physically overlapped releases | misses | misses after retirement | worst response (ms) | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cap, item in grouped.items():
        lines.append(
            f"| {cap} | {item['runs']} | {item['releases']} | "
            f"{item['active_releases']} | {item['misses']} | "
            f"{item['post_retirement_misses']} | {item['worst_response_ms']:.3f} | "
            f"{fmt(item['active_p99_ms_mean'])} | {fmt(item['worst_active_max_ms'])} | "
            f"{fmt(item['worker_gpu_mean_ms'])} | {fmt(item['worker_gpu_worst_max_ms'])} | "
            f"{','.join(map(str, item['visible_sms']))} | {item['gates']['all_pass']} |"
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    if any(
        x["retire_after_timeout"] or x["retire_before_first_release"]
        for x in rows
    ):
        lines.extend([
            "",
            "## Retirement diagnostics",
            "",
        ])
        for cap, item in grouped.items():
            offsets = ", ".join(map(str, item["post_retirement_miss_offsets"]))
            lines.append(
                f"- cap{cap}: {item['post_retirement_misses']} misses after "
                f"retirement; release offsets [{offsets or 'none'}]."
            )
        lines.extend([
            "",
            "A post-retirement miss is not counted as physical worker overlap. "
            "The timing association does not by itself prove a teardown cause; "
            "compare against a worker-free control and inspect GPU-event time.",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
