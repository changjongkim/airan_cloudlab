#!/usr/bin/env python3.11
"""Frozen audit of instrumented NRx pre-dispatch tail in four-cell MPS mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm85_tail_probe_protocol.json")
    raw = base / "raw"
    prefix = protocol["prefix"]
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    admitted = [r for r in records if r["admitted"]]
    late = [r for r in admitted if r["nrx_response_ms"] > 30]
    conv = [r["conventional_host_path_ms"] for r in records
            if r["conventional_host_path_ms"] is not None]
    details = []
    for row in late:
        endpoint = int(row["endpoint_id"][-1])
        dispatch_ns = row["nrx_dispatched_ns"]
        window = min(workers[endpoint]["execution_windows"],
                     key=lambda w: abs(w["observed_ns"] - dispatch_ns))
        group = records[4 * row["index"]:4 * row["index"] + 4]
        feature_end_ns = max(r["feature_return_ns"] for r in group)
        details.append({
            "release_index": row["index"], "cell": row["cell"],
            "response_ms": row["nrx_response_ms"],
            "features_done_to_prepare_start_ms":
                (row["nrx_prepare_begin_ns"] - feature_end_ns) / 1e6,
            "admission_host_ms": row["nrx_admission_host_ms"],
            "prepare_host_ms": row["nrx_prepare_host_ms"],
            "prepare_gpu_ms": row["nrx_prepare_gpu_ms"],
            "prepare_start_after_release_ms":
                (row["nrx_prepare_begin_ns"] - row["release_ns"]) / 1e6,
            "dispatch_after_release_ms":
                (dispatch_ns - row["release_ns"]) / 1e6,
            "dispatch_to_observation_ms":
                (row["nrx_observed_ns"] - dispatch_ns) / 1e6,
            "worker_gpu_ms": window["gpu_ms"],
            "worker_observed_after_dispatch_ms":
                (window["observed_ns"] - dispatch_ns) / 1e6,
        })
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "contract": (
            ran["schema"] == "softwall-four-cell-tail-probe-v1"
            and ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 4
            and ran["period_ms"] == 180
            and ran["deadline_ms"] == 155
            and ran["nrx_bound_ms"] == 50
            and ran["conv_bound_ms"] == 25
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and len(records) == 4 * protocol["iterations"]
            and all(r["channel_seed"] == protocol["channel_seed_base"]
                    + r["index"] * 4 + r["cell"] for r in records)
            and requal["iterations"] == 200
            and requal["deadline_misses"] == 0
        ),
        "provenance": (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and ran["host"] == protocol["node"]
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
        ),
        "measurement_complete": (
            len(admitted) >= protocol["min_admitted"]
            and all(r["nrx_admission_host_ms"] is not None
                    and r["nrx_prepare_host_ms"] is not None
                    and r["nrx_prepare_gpu_ms"] is not None for r in admitted)
        ),
        "current_contract_sample": (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and ran["background_units"] == background["completed_units"]
        ),
        "late_tail_reproduced": len(late) >= 1,
    }
    report = {
        "schema": "softwall-confirm85-fourcell-tail-probe-v1",
        "job": protocol["job"], "gates": gates,
        "all_pass": all(gates.values()),
        "admitted_nrx": len(admitted),
        "late_nrx_over_30_ms": details,
        "max_nrx_response_ms": max(r["nrx_response_ms"] for r in admitted),
        "max_nrx_prepare_host_ms": max(r["nrx_prepare_host_ms"] for r in admitted),
        "max_nrx_prepare_gpu_ms": max(r["nrx_prepare_gpu_ms"] for r in admitted),
        "max_conv_host_ms": max(conv),
        "deadline_misses": ran["deadline_misses"],
        "interpretation": "Direct admission, prepare_neural_ipc host/GPU, and endpoint worker timing for a reproducible around-release-620 tail. This is a single synthetic mode diagnostic, not WCET or policy superiority.",
    }
    output = base / f"confirm85_tail_probe_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [k for k, v in gates.items() if not v],
                      "late_nrx": len(late),
                      "max_prepare_host_ms": report["max_nrx_prepare_host_ms"],
                      "max_prepare_gpu_ms": report["max_nrx_prepare_gpu_ms"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
