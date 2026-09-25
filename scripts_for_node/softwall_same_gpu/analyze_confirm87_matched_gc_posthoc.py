#!/usr/bin/env python3.11
"""Posthoc matched NRx-response audit at the longest natural gen-2 GC event."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"
RAW = BASE / "raw"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(on_name: str, off_name: str) -> dict:
    on_path = RAW / f"confirm87_{on_name}_job58743005_controller.json"
    off_path = RAW / f"confirm87_{off_name}_job58743005_controller.json"
    on = read(on_path)
    off = read(off_path)
    on_rows = on["records"]
    off_rows = off["records"]
    event = max((e for e in on["gc_events"] if e["generation"] == 2),
                key=lambda e: e["end_ns"] - e["begin_ns"])
    release = min(on["release_stage_records"],
                  key=lambda s: abs(s["release_ns"] - event["begin_ns"]))
    index = release["index"]
    workers = [read(RAW / f"confirm87_{on_name}_job58743005_worker{i}.json")
               for i in range(2)]
    matched = []
    paired_deltas = sorted(
        a["nrx_response_ms"] - b["nrx_response_ms"]
        for a, b in zip(on_rows, off_rows)
        if a["nrx_response_ms"] is not None
        and b["nrx_response_ms"] is not None
    )
    for ran, baseline in zip(on_rows[4 * index:4 * index + 4],
                             off_rows[4 * index:4 * index + 4]):
        if ran["nrx_response_ms"] is None:
            continue
        endpoint = int(ran["endpoint_id"][-1])
        window = min(workers[endpoint]["execution_windows"],
                     key=lambda w: abs(w["observed_ns"] - ran["nrx_dispatched_ns"]))
        matched.append({
            "cell": ran["cell"],
            "on_nrx_response_ms": ran["nrx_response_ms"],
            "off_nrx_response_ms": baseline["nrx_response_ms"],
            "on_minus_off_response_ms":
                ran["nrx_response_ms"] - baseline["nrx_response_ms"],
            "worker_gpu_ms": window["gpu_ms"],
            "worker_published_after_release_ms":
                (window["published_ns"] - release["release_ns"]) / 1e6,
            "on_controller_observed_after_release_ms":
                (ran["nrx_observed_ns"] - release["release_ns"]) / 1e6,
            "worker_published_before_gc_start":
                window["published_ns"] <= event["begin_ns"],
            "controller_observed_after_gc_end":
                ran["nrx_observed_ns"] >= event["end_ns"],
        })
    return {
        "on": on_name, "off": off_name,
        "on_controller_sha256": hashlib.sha256(on_path.read_bytes()).hexdigest(),
        "off_controller_sha256": hashlib.sha256(off_path.read_bytes()).hexdigest(),
        "matched_channel_seeds": all(a["channel_seed"] == b["channel_seed"]
                                     for a, b in zip(on_rows, off_rows)),
        "exact_feature_matches": sum(a["channel_estimate_power"]
                                     == b["channel_estimate_power"]
                                     for a, b in zip(on_rows, off_rows)),
        "same_admission_decisions": sum(a["admitted"] == b["admitted"]
                                        for a, b in zip(on_rows, off_rows)),
        "total_records": len(on_rows),
        "matched_nrx_count": len(paired_deltas),
        "matched_nrx_median_delta_ms": statistics.median(paired_deltas),
        "matched_nrx_p99_delta_ms": paired_deltas[int(0.99 * len(paired_deltas))],
        "gc_release_index": index,
        "gc_start_after_release_ms":
            (event["begin_ns"] - release["release_ns"]) / 1e6,
        "gc_end_after_release_ms":
            (event["end_ns"] - release["release_ns"]) / 1e6,
        "gc_duration_ms": (event["end_ns"] - event["begin_ns"]) / 1e6,
        "off_gen2_gc_events": sum(e["generation"] == 2 for e in off["gc_events"]),
        "matched_nrx": matched,
    }


def main() -> None:
    frozen = read(BASE / "confirm87_gc_intervention_job58743005.json")
    rows = [pair("a_on", "b_off"), pair("d_on", "c_off")]
    report = {
        "schema": "softwall-confirm87-matched-gc-posthoc-v1",
        "frozen_tail_attribution_gate_pass": frozen["tail_attribution_gate_pass"],
        "pairs": rows,
        "interpretation": "Posthoc secondary paired response analysis. Both seeds match all channel features and admission decisions; natural gen-2 GC falls between physical worker publication and host observation for the listed late NRx requests. This supports a causal host-observation mechanism but does not revise the failed frozen C87 primary gate. Independent prospective replication is required.",
    }
    output = BASE / "confirm87_matched_gc_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_primary_pass": report["frozen_tail_attribution_gate_pass"],
        "pairs": [
            {"release": p["gc_release_index"],
             "gc_ms": p["gc_duration_ms"],
             "deltas_ms": [x["on_minus_off_response_ms"]
                           for x in p["matched_nrx"]]}
            for p in rows
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
