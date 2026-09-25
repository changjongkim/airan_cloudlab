#!/usr/bin/env python3.11
"""Frozen audit of release-stage and Python-GC timing around the NRx tail."""

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
    protocol = read(base / "confirm86_stage_gc_probe_protocol.json")
    raw = base / "raw"
    prefix = protocol["prefix"]
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    stages = ran["release_stage_records"]
    gc_events = ran["gc_events"]
    late_rows = [r for r in records if r["nrx_response_ms"] is not None
                 and r["nrx_response_ms"] > 30]
    late = []
    for row in late_rows:
        stage = stages[row["index"]]
        release_ns = row["release_ns"]
        group = records[4 * row["index"]:4 * row["index"] + 4]
        feature_end_ns = max(r["feature_return_ns"] for r in group)
        first_dispatch_ns = min(r["nrx_dispatch_begin_ns"] for r in group
                                if r["nrx_dispatch_begin_ns"] is not None)
        nearby_gc = [e for e in gc_events if e["end_ns"] >= release_ns - 50_000_000
                     and e["begin_ns"] <= release_ns + 60_000_000]
        late.append({
            "release_index": row["index"], "cell": row["cell"],
            "response_ms": row["nrx_response_ms"],
            "channel_prep_begin_after_release_ms":
                (stage["channel_prep_begin_ns"] - release_ns) / 1e6,
            "channel_prep_host_ms":
                (stage["channel_prep_end_ns"] - stage["channel_prep_begin_ns"]) / 1e6,
            "channel_prep_end_after_release_ms":
                (stage["channel_prep_end_ns"] - release_ns) / 1e6,
            "wait_begin_after_release_ms": (stage["wait_begin_ns"] - release_ns) / 1e6,
            "wait_return_after_release_ms": (stage["wait_return_ns"] - release_ns) / 1e6,
            "first_feature_begin_after_release_ms":
                (group[0]["feature_begin_ns"] - release_ns) / 1e6,
            "features_done_after_release_ms": (feature_end_ns - release_ns) / 1e6,
            "admission_loop_host_ms":
                (stage["admission_loop_end_ns"] - stage["admission_loop_begin_ns"]) / 1e6,
            "first_dispatch_after_release_ms": (first_dispatch_ns - release_ns) / 1e6,
            "admission_host_ms": row["nrx_admission_host_ms"],
            "prepare_host_ms": row["nrx_prepare_host_ms"],
            "prepare_gpu_ms": row["nrx_prepare_gpu_ms"],
            "nearby_gc": [{"generation": e["generation"],
                           "begin_after_release_ms": (e["begin_ns"] - release_ns) / 1e6,
                           "duration_ms": (e["end_ns"] - e["begin_ns"]) / 1e6}
                          for e in nearby_gc],
        })
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "contract": (
            ran["schema"] == "softwall-four-cell-stage-gc-probe-v1"
            and ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 4
            and ran["nrx_bound_ms"] == 50
            and ran["conv_bound_ms"] == 25
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and len(records) == 4 * protocol["iterations"]
            and len(stages) == protocol["iterations"]
            and all(s["index"] == i for i, s in enumerate(stages))
            and requal["iterations"] == 200
            and requal["deadline_misses"] == 0
        ),
        "provenance": (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and ran["host"] == protocol["node"]
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
        ),
        "sample_safety": (
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
        "tail_reproduced": len(late) >= 1,
    }
    report = {
        "schema": "softwall-confirm86-stage-gc-probe-v1",
        "job": protocol["job"], "gates": gates,
        "all_pass": all(gates.values()),
        "late_nrx_over_30_ms": late,
        "gc_events_count": len(gc_events),
        "max_gc_duration_ms": max(((e["end_ns"] - e["begin_ns"]) / 1e6
                                   for e in gc_events), default=0),
        "max_wait_return_lateness_ms": max(
            (s["wait_return_ns"] - s["release_ns"]) / 1e6 for s in stages),
        "max_channel_prep_host_ms": max(
            (s["channel_prep_end_ns"] - s["channel_prep_begin_ns"]) / 1e6
            for s in stages),
        "interpretation": "Prospectively instrumented release prep, wait, feature, admission, dispatch, and Python GC event intervals. Temporal overlap alone does not prove GC causality; no new service bound or policy advantage follows.",
    }
    output = base / f"confirm86_stage_gc_probe_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [k for k, v in gates.items() if not v],
                      "late_nrx": len(late),
                      "gc_events": len(gc_events)}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
