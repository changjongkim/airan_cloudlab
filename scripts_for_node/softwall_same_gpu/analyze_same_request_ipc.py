#!/usr/bin/env python3
"""Summarize persistent same-request CUDA-IPC NeuralRx campaigns."""

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
        + r"_cap(?P<cap>[0-9]+)_job(?P<job>[0-9]+)_controller[.]json$"
    )
    rows = []
    for path in sorted(args.raw.glob(f"{args.campaign}_*_controller.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        controller = json.loads(path.read_text(encoding="utf-8"))
        worker_path = Path(str(path).replace("_controller.json", "_worker.json"))
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        recoveries = sum(
            item.get("neural_correct") is False
            and item.get("conventional_correct") is True
            for item in controller["records"]
        )
        unrecovered = sum(not item["correct"] for item in controller["records"])
        expected_worker_units = (
            controller["warmup"]
            + controller.get("noisy_warmup_units", 0)
            + controller.get("endpoint_requests", controller["iterations"])
        )
        rows.append({
            "cap": int(match["cap"]),
            "job": int(match["job"]),
            "iterations": controller["iterations"],
            "snr_db": controller.get("snr_db"),
            "policy": controller.get("policy", "s2"),
            "correct": controller["correct_releases"],
            "misses": controller["deadline_misses"],
            "nrx_commits": controller["nrx_commits"],
            "fallbacks": controller["fallbacks"],
            "conventional_runs": controller.get(
                "conventional_runs", controller["fallbacks"]
            ),
            "recoveries": recoveries,
            "unrecovered": unrecovered,
            "timeouts": controller["endpoint_timeouts"],
            "response_p99_ms": controller["response_ms"]["p99"],
            "response_max_ms": controller["response_ms"]["max"],
            "front_mean_ms": controller["front_gpu_ms"]["mean"],
            "post_mean_ms": controller["post_gpu_ms"]["mean"],
            "worker_units": worker["completed_units"],
            "expected_worker_units": expected_worker_units,
            "worker_gpu_mean_ms": worker["gpu_ms"]["mean"],
            "visible_sms": worker["visible_sm_count"],
            "lifecycle": controller["lifecycle"],
        })
    if not rows:
        raise SystemExit("no same-request IPC results")

    gates = {
        "clean_runs_all_correct": all(
            x["snr_db"] is not None or x["correct"] == x["iterations"]
            for x in rows
        ),
        "natural_runs_have_recoveries": all(
            x["snr_db"] is None
            or x["policy"] == "conventional"
            or x["recoveries"] > 0
            for x in rows
        ),
        "correct_at_least_nrx_commits": all(
            x["correct"] >= x["nrx_commits"] for x in rows
        ),
        "deadline_misses_zero": sum(x["misses"] for x in rows) == 0,
        "endpoint_timeouts_zero": sum(x["timeouts"] for x in rows) == 0,
        "worker_units_match": all(
            x["worker_units"] == x["expected_worker_units"] for x in rows
        ),
        "persistent_lifecycle": all(
            x["lifecycle"]
            == "persistent during RAN epoch; teardown after final release"
            for x in rows
        ),
        "cap80_visible_sms": all(
            x["visible_sms"] == 86 for x in rows if x["cap"] == 80
        ),
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema": "softwall-same-request-ipc-analysis-v1",
        "campaign": args.campaign,
        "gates": gates,
        "totals": {
            "releases": sum(x["iterations"] for x in rows),
            "correct": sum(x["correct"] for x in rows),
            "misses": sum(x["misses"] for x in rows),
            "nrx_commits": sum(x["nrx_commits"] for x in rows),
            "fallbacks": sum(x["fallbacks"] for x in rows),
            "conventional_runs": sum(x["conventional_runs"] for x in rows),
            "recoveries": sum(x["recoveries"] for x in rows),
            "unrecovered": sum(x["unrecovered"] for x in rows),
            "timeouts": sum(x["timeouts"] for x in rows),
        },
        "rows": rows,
    }
    lines = [
        "# Persistent same-request CUDA-IPC NeuralRx",
        "",
        "| policy | cap | SNR (dB) | releases | correct | NRx commit | conventional runs | fallback | recovered | miss | timeout | response p99/max (ms) | worker GPU mean (ms) | visible SMs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        snr = "clean" if row["snr_db"] is None else f"{row['snr_db']:.1f}"
        lines.append(
            f"| {row['policy']} | {row['cap']} | {snr} | {row['iterations']} | "
            f"{row['correct']} | {row['nrx_commits']} | "
            f"{row['conventional_runs']} | {row['fallbacks']} | "
            f"{row['recoveries']} | {row['misses']} | {row['timeouts']} | "
            f"{row['response_p99_ms']:.3f}/{row['response_max_ms']:.3f} | "
            f"{row['worker_gpu_mean_ms']:.3f} | {row['visible_sms']} |"
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
