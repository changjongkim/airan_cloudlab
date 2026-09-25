#!/usr/bin/env python3.11
"""Audit early conventional versus other-cell NeuralRx GPU kernel overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_nsys_client_pair import activity
from analyze_nsys_overlap_union import clipped_union, intersect


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm70_conv_nrx_nodes_protocol.json")
    prefix = f"confirm70_nodes_job{protocol['job']}"
    raw = base / "raw"
    ran = read(raw / f"{prefix}_controller.json")
    worker = read(raw / f"{prefix}_worker0.json")
    controller_kernels, controller_base = activity(
        raw / f"{prefix}_controller_profile.sqlite", "CUPTI_ACTIVITY_KIND_KERNEL"
    )
    nrx_kernels, nrx_base = activity(
        raw / f"{prefix}_worker0_profile.sqlite", "CUPTI_ACTIVITY_KIND_KERNEL"
    )
    windows = []
    for row in ran["records"]:
        if not (row["cell"] == 1 and row["gate_skipped"]
                and row["host_overlap_with_other_nrx"]):
            continue
        other = ran["records"][2 * row["index"]]
        if not other["admitted"] or other["endpoint_id"] != "nrx0":
            continue
        lower = row["fallback_actual_start_ns"]
        upper = row["commit_return_ns"]
        controller_in = clipped_union(controller_kernels, lower, upper)
        nrx_in = clipped_union(nrx_kernels, lower, upper)
        segments, overlap_ns = intersect(controller_in, nrx_in)
        windows.append({
            "release_index": row["index"],
            "controller_kernel_segments": len(controller_in),
            "nrx0_kernel_segments": len(nrx_in),
            "overlap_segments": segments,
            "unique_kernel_overlap_ms": overlap_ns / 1_000_000,
            "conventional_host_path_ms": row["conventional_host_path_ms"],
        })
    gates = {
        "source_hashes": (
            hashlib.sha256((root / protocol["runner"]).read_bytes()).hexdigest()
            == protocol["runner_sha256"]
            and hashlib.sha256((root / "scripts_for_node/softwall_same_gpu/two_endpoint_corun_gate_controller.py").read_bytes()).hexdigest()
            == protocol["controller_sha256"]
        ),
        "provenance": (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and str(ran["slurm_job_id"]) == protocol["job"]
            and str(worker["slurm_job_id"]) == protocol["job"]
            and ran["host"] == worker["host"]
        ),
        "safety_sample": (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["fallback_calendar_final"]["outstanding"] == 0
        ),
        "runtime_kernel_tables": (
            any(start >= ran["records"][0]["release_ns"] for start, _ in controller_kernels)
            and any(start >= ran["records"][0]["release_ns"] for start, _ in nrx_kernels)
        ),
        "conventional_nrx_kernel_overlap": any(
            row["unique_kernel_overlap_ms"] > 0 for row in windows
        ),
    }
    report = {
        "schema": "softwall-confirm70-conventional-nrx-kernel-overlap-v1",
        "job": protocol["job"],
        "gates": gates,
        "all_pass": all(gates.values()),
        "host_overlap_count_total": ran["host_overlapped_early_mandatory"],
        "profiled_cell1_windows": len(windows),
        "physical_overlap_windows": sum(row["unique_kernel_overlap_ms"] > 0 for row in windows),
        "physical_overlap_ms_sum": sum(row["unique_kernel_overlap_ms"] for row in windows),
        "controller_session_start_ns": controller_base,
        "worker0_session_start_ns": nrx_base,
        "windows": windows,
        "interpretation": "One profiled NeuralRx worker and the radio controller physically overlap GPU kernel execution during early conventional host-call windows. Forty synthetic releases, profiler perturbation, and sampled conventional path times do not qualify an NRx-conventional worst-case service bound or production MAC deadline.",
    }
    output = base / f"confirm70_conv_nrx_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "host_overlap_count_total": report["host_overlap_count_total"],
                      "profiled_cell1_windows": report["profiled_cell1_windows"],
                      "physical_overlap_windows": report["physical_overlap_windows"],
                      "physical_overlap_ms_sum": report["physical_overlap_ms_sum"]}))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
