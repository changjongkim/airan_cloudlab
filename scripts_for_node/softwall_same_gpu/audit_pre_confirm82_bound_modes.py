#!/usr/bin/env python3.11
"""Recompute the mode mismatch behind proposed tighter PHY service bounds."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"
RAW = BASE / "raw"


def read(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def path_stats(name: str) -> dict:
    data = read(name)
    nrx = [row["nrx_response_ms"] for row in data["records"]
           if row.get("nrx_response_ms") is not None]
    conv = [row["conventional_host_path_ms"] for row in data["records"]
            if row.get("conventional_host_path_ms") is not None]
    return {
        "source": f"raw/{name}", "cells": data["cells"],
        "releases": data["iterations"],
        "nrx_n": len(nrx), "nrx_max_ms": max(nrx),
        "nrx_over_25_ms": sum(value > 25 for value in nrx),
        "nrx_over_30_ms": sum(value > 30 for value in nrx),
        "conv_host_n": len(conv), "conv_host_max_ms": max(conv),
        "conv_host_over_8_ms": sum(value > 8 for value in conv),
        "deadline_misses": data["deadline_misses"],
    }


def main() -> None:
    long_tail = []
    for campaign, job in ((43, "58682997"), (44, "58685929")):
        for name in sorted(RAW.glob(f"confirm{campaign}_gap_r*_cap80_job{job}_controller.json")):
            data = json.loads(name.read_text(encoding="utf-8"))
            violations = [row for row in data["records"] if row["nrx_bound_violation"]]
            long_tail.append({
                "source": f"raw/{name.name}",
                "releases": data["iterations"],
                "declared_nrx_bound_ms": data["nrx_bound_ms"],
                "violations": [{
                    "release_index": row["index"],
                    "front_gpu_ms": row["front_gpu_ms"],
                    "post_gpu_ms": row["post_gpu_ms"],
                    "response_ms": row["response_ms"],
                    "start_lateness_ms": row["start_lateness_ms"],
                    "deadline_miss": row["deadline_miss"],
                } for row in violations],
            })
    samples = [path_stats(name) for name in (
        "confirm56_host_path_job58729926_controller.json",
        "confirm76_ai15_job58738952_controller.json",
        "confirm80_fourcell_job58738952_controller.json",
        "confirm81_fenced_fault_job58738952_controller.json",
    )]
    cold = json.loads((BASE / "confirm40_cold_start_ablation.json").read_text())
    report = {
        "schema": "softwall-pre-confirm82-bound-mode-audit-v1",
        "historical_nrx_30ms_failures": long_tail,
        "historical_violation_count": sum(len(row["violations"]) for row in long_tail),
        "shorter_prewarmed_samples": samples,
        "cold_conventional_source": "confirm40_cold_start_ablation.json",
        "cold_conventional_audit": cold,
        "interpretation": (
            "The 30 ms NeuralRx failures in the long older cap80 mode include "
            "GPU front/post intervals of tens of milliseconds around release 2740; "
            "they are not explained by host wait alone. Short four-cell warm "
            "samples cannot qualify 8/25/30 ms bounds. Confirm40's unprewarmed "
            "first fallback is a distinct cold mode with a 35.303 ms GPU sample."
        ),
    }
    output = BASE / "pre_confirm82_bound_mode_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical_nrx_30ms_failures": report["historical_violation_count"],
                      "short_sample_max_nrx_ms": [row["nrx_max_ms"] for row in samples],
                      "short_sample_max_conv_ms": [row["conv_host_max_ms"] for row in samples]}, indent=2))


if __name__ == "__main__":
    main()
