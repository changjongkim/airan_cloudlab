#!/usr/bin/env python3.11
"""Posthoc audit of Confirm97 after arm B aborted before controller JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    raw = base / "raw"
    protocol_path = base / "confirm97_qwen_fourcell_protocol.json"
    protocol = read(protocol_path)
    source_ok = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                    for name, digest in protocol["source_sha256_before_run"].items())
    a_path = raw / "confirm97_a_qwen_job58747070_controller.json"
    a = read(a_path)
    ai = a["background_records"]
    groups = [a["records"][4*i:4*i+4] for i in range(a["iterations"])]
    by_release = defaultdict(list)
    for item in ai:
        by_release[item["release_index"]].append(item)
    all_fail = [i for i, group in enumerate(groups)
                if sum(row["admitted"] for row in group) == 2
                and all(row["forced_nrx_failure"] for row in group if row["admitted"])
                and all(row["commit_kind"] == "conventional" for row in group)]
    by_d153 = [sum(item["returned_ns"] <= groups[i][0]["release_ns"]
                       + 153_000_000 for item in by_release[i]) for i in all_fail]
    nrx = [row["nrx_response_ms"] for row in a["records"]
           if row["nrx_response_ms"] is not None]
    b_ai_path = raw / "confirm97_b_qwen_job58747070_background.json"
    b_ai = read(b_ai_path)
    b_log_path = base / "confirm97_b_qwen_job58747070.log"
    b_log = b_log_path.read_text(encoding="utf-8")
    report = {
        "schema": "softwall-confirm97-qwen-partial-failure-v1",
        "posthoc_partial_audit": True,
        "job": protocol["job"], "node": protocol["node"],
        "source_hashes": source_ok,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "arm_a": {
            "raw_sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
            "completed_releases": a["iterations"],
            "radio_deadline_misses": a["deadline_misses"],
            "nrx30_violations": a["nrx_bound_violations"],
            "nrx_response_max_ms": max(nrx),
            "nrx_response_median_ms": statistics.median(nrx),
            "conv12_violations": a["conv_path_bound_violations"],
            "ai40_violations": a["background_budget_violations"],
            "ai_host_max_ms": max(item["execution_ms"] for item in ai),
            "qwen_units": len(ai),
            "injected_all_fail_releases": len(all_fail),
            "all_fail_below_seven_ai_by_d153": sum(count < 7 for count in by_d153),
            "all_fail_min_ai_by_d153": min(by_d153),
            "all_fail_median_ai_by_d153": statistics.median(by_d153),
            "all_fail_max_ai_by_d153": max(by_d153),
            "joint_ai_leases_retired": a["joint_lease_retired_count"],
        },
        "arm_b": {
            "worker_raw_sha256": hashlib.sha256(b_ai_path.read_bytes()).hexdigest(),
            "log_sha256": hashlib.sha256(b_log_path.read_bytes()).hexdigest(),
            "controller_raw_exists":
                (raw / "confirm97_b_qwen_job58747070_controller.json").exists(),
            "qwen_gpu_units_before_abort": b_ai["completed_units"],
            "qwen_gpu_max_ms_before_abort": b_ai["gpu_ms"]["max"],
            "qwen_gpu_exceeded_rpc_timeout35": b_ai["gpu_ms"]["max"] > 35,
            "unconfirmed_joint_lease_error_in_log":
                "joint AI physical completion unconfirmed; lease retained" in b_log,
            "broken_pipe_in_log": "BrokenPipeError" in b_log,
        },
        "overall_pass": False,
        "interpretation": "Frozen Confirm97 failed: arm A finished with 705 NRx30 bound violations; arm B aborted before controller JSON after an unconfirmed joint AI completion. Its Qwen worker recorded a 36.246-ms GPU unit, above the 35-ms RPC timeout, then BrokenPipe. The timeout is a supported inference, not directly timestamped in a surviving controller fault record. Arm A's fewer-than-seven Qwen returns by D153 show pressure under this controller, not optimal infeasibility.",
    }
    output = base / "confirm97_qwen_fourcell_failure_job58747070.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
