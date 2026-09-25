#!/usr/bin/env python3
"""Analyze timeout quarantine followed by maintenance-window requalification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_cap(?P<cap>[0-9]+)_job(?P<job>[0-9]+)_pre_ran[.]json$"
    )
    rows = []
    for pre_path in sorted(args.raw.glob(f"{args.campaign}_r*_pre_ran.json")):
        match = pattern.fullmatch(pre_path.name)
        if not match:
            continue
        prefix = str(pre_path)[:-len("_pre_ran.json")]
        worker_path = Path(prefix + "_worker.json")
        post_path = Path(prefix + "_post_ran.json")
        if not worker_path.is_file() or not post_path.is_file():
            continue
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        post = json.loads(post_path.read_text(encoding="utf-8"))
        rows.append({
            "round": int(match["round"]),
            "cap": int(match["cap"]),
            "job": match["job"],
            "pre_releases": pre["iterations"],
            "pre_misses": pre["deadline_misses"],
            "faults": pre["injected_overruns"],
            "timeouts": pre["timeouts_detected"],
            "drained": pre["responses_drained"],
            "outstanding_at_end": pre["outstanding_at_end"],
            "worker_units": worker["completed_units"],
            "post_releases": post["iterations"],
            "post_correct": post["correct_releases"],
            "post_misses": post["deadline_misses"],
            "post_response_max_ms": post["response_ms"]["max"],
            "post_gpu_max_ms": post["gpu_ms"]["max"],
        })
    if not rows:
        raise SystemExit("no complete maintenance results")
    gates = {
        "one_fault_per_run": all(x["faults"] == 1 for x in rows),
        "all_faults_timeout": all(x["timeouts"] == x["faults"] for x in rows),
        "all_faults_drained": all(x["drained"] == x["faults"] for x in rows),
        "no_outstanding_before_maintenance": not any(
            x["outstanding_at_end"] for x in rows
        ),
        "one_worker_unit_per_run": all(x["worker_units"] == 1 for x in rows),
        "post_all_correct": all(
            x["post_correct"] == x["post_releases"] for x in rows
        ),
        "post_deadline_misses_zero": sum(x["post_misses"] for x in rows) == 0,
    }
    gates["all_pass"] = all(gates.values())
    grouped = {}
    for cap in sorted({x["cap"] for x in rows}):
        items = [x for x in rows if x["cap"] == cap]
        grouped[str(cap)] = {
            "runs": len(items),
            "pre_releases": sum(x["pre_releases"] for x in items),
            "pre_misses": sum(x["pre_misses"] for x in items),
            "post_releases": sum(x["post_releases"] for x in items),
            "post_misses": sum(x["post_misses"] for x in items),
            "post_worst_response_ms": max(x["post_response_max_ms"] for x in items),
            "post_worst_gpu_ms": max(x["post_gpu_max_ms"] for x in items),
        }
    result = {
        "schema": "softwall-maintenance-requalification-analysis-v1",
        "campaign": args.campaign,
        "gates": gates,
        "by_cap": grouped,
        "rows": rows,
    }
    lines = [
        "# SoftWall maintenance-window requalification",
        "",
        "| cap | runs | fault-phase releases | fault-phase misses | post-maintenance releases | post-maintenance misses | post worst response (ms) | post worst GPU (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cap, item in grouped.items():
        lines.append(
            f"| {cap} | {item['runs']} | {item['pre_releases']} | "
            f"{item['pre_misses']} | {item['post_releases']} | "
            f"{item['post_misses']} | {item['post_worst_response_ms']:.3f} | "
            f"{item['post_worst_gpu_ms']:.3f} |"
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
