#!/usr/bin/env python3
"""Assess an ABBA conventional-recovery warmup intervention."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = []
    for round_index, mode in enumerate(("off", "on", "on", "off"), start=1):
        prefix = f"confirm40_cold_start_r{round_index}_{mode}_job{args.job}"
        path = args.raw / (prefix + "_controller.json")
        worker_path = args.raw / (prefix + "_worker.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        fallbacks = [
            item for item in data["records"]
            if item.get("conventional_gpu_ms") is not None
        ]
        if len(fallbacks) < 2:
            raise SystemExit(f"too few fallbacks in {path}")
        first = fallbacks[0]
        rows.append({
            "round": round_index,
            "mode": mode,
            "job": data["slurm_job_id"],
            "host": data["host"],
            "worker_job": worker["slurm_job_id"],
            "worker_host": worker["host"],
            "worker_visible_sms": worker["visible_sm_count"],
            "payload_seed": data["payload_seed"],
            "channel_seed_base": data["channel_seed_base"],
            "snr_db": data["snr_db"],
            "period_ms": data["period_ms"],
            "deadline_ms": data["deadline_ms"],
            "conventional_warmup": data["conventional_warmup"],
            "iterations": data["iterations"],
            "correct": data["correct_releases"],
            "deadline_misses": data["deadline_misses"],
            "endpoint_timeouts": data["endpoint_timeouts"],
            "warmup_units": (
                (data["warmup"] if mode == "on" else 0)
                + data.get("noisy_conventional_warmup_units", 0)
            ),
            "first_fallback_index": first["index"],
            "first_fallback_gpu_ms": first["conventional_gpu_ms"],
            "first_fallback_response_ms": first["response_ms"],
            "later_fallback_gpu_median_ms": statistics.median(
                item["conventional_gpu_ms"] for item in fallbacks[1:]
            ),
            "endpoint_worker_units": worker["completed_units"],
            "records": data["records"],
        })
    off = [row["first_fallback_gpu_ms"] for row in rows if row["mode"] == "off"]
    on = [row["first_fallback_gpu_ms"] for row in rows if row["mode"] == "on"]
    gates = {
        "protocol_identity": (
            protocol["campaign"] == "confirm40_cold_start_ablation"
            and protocol["order"] == ["off", "on", "on", "off"]
        ),
        "all_same_host_job_seed": len({
            (row["host"], row["job"], row["payload_seed"],
             row["channel_seed_base"])
            for row in rows
        }) == 1,
        "all_match_frozen_radio": all(
            row["job"] == args.job
            and row["worker_job"] == args.job
            and row["host"] == row["worker_host"]
            and row["worker_visible_sms"] == 86
            and row["payload_seed"] == protocol["radio"]["payload_seed"]
            and row["channel_seed_base"] == protocol["radio"]["channel_seed_base"]
            and row["snr_db"] == protocol["radio"]["snr_db"]
            and row["period_ms"] == protocol["radio"]["period_ms"]
            and row["deadline_ms"] == protocol["radio"]["deadline_ms"]
            and row["conventional_warmup"] == row["mode"]
            for row in rows
        ),
        "all_500_complete": all(row["iterations"] == len(row["records"]) == 500 for row in rows),
        "channel_trace_identical": all(
            len({row["records"][index]["channel_seed"] for row in rows}) == 1
            for index in range(500)
        ),
        "first_fallback_same_release": len({
            row["first_fallback_index"] for row in rows
        }) == 1,
        "all_deadline_misses_zero": all(row["deadline_misses"] == 0 for row in rows),
        "all_endpoint_timeouts_zero": all(row["endpoint_timeouts"] == 0 for row in rows),
        "warmup_modes_applied": all(
            (row["warmup_units"] == 21) if row["mode"] == "on"
            else (row["warmup_units"] == 0)
            for row in rows
        ),
        "endpoint_units_match": all(row["endpoint_worker_units"] == 521 for row in rows),
        "both_off_runs_slower_than_both_on_runs_by_frozen_margin": (
            min(off) > max(on) + protocol["primary_gate"]["each_off_first_fallback_gpu_ms_exceeds_each_on_by_at_least"]
        ),
    }
    gates["all_pass"] = all(gates.values())
    for row in rows:
        row.pop("records")
    report = {
        "schema": "softwall-recovery-cold-start-abba-v1",
        "job": args.job,
        "rows": rows,
        "first_fallback_off_median_ms": statistics.median(off),
        "first_fallback_on_median_ms": statistics.median(on),
        "gates": gates,
    }
    lines = [
        "# Conventional recovery cold-start ABBA ablation",
        "",
        "| round | warm-up | first fallback release | first conventional GPU ms | "
        "later median ms | deadline misses |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['mode']} | {row['first_fallback_index']} | "
            f"{row['first_fallback_gpu_ms']:.3f} | "
            f"{row['later_fallback_gpu_median_ms']:.3f} | "
            f"{row['deadline_misses']} |"
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
