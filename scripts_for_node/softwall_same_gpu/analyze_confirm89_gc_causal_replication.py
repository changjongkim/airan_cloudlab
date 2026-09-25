#!/usr/bin/env python3.11
"""Prospective ON/OFF replication using matched NRx response at gen-2 GC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_data(base: Path, arm: dict, protocol: dict) -> dict:
    raw = base / "raw"
    prefix = arm["prefix"]
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    gates = {
        "contract": (
            ran["schema"] == "softwall-four-cell-gc-intervention-v1"
            and ran["gc_mode"] == arm["gc_mode"]
            and ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 4
            and ran["period_ms"] == 180
            and ran["deadline_ms"] == 155
            and ran["nrx_bound_ms"] == 50
            and ran["conv_bound_ms"] == 25
            and ran["payload_seed"] == arm["payload_seed"]
            and ran["channel_seed_base"] == arm["channel_seed_base"]
            and len(records) == 4 * protocol["iterations"]
            and len(ran["release_stage_records"]) == protocol["iterations"]
            and all(r["channel_seed"] == arm["channel_seed_base"]
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
        "sample_safety": (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and ran["pre_radio_ai_guard_violations"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and ran["background_units"] == background["completed_units"]
        ),
    }
    return {"name": arm["name"], "ran": ran, "workers": workers,
            "gates": gates, "all_pass": all(gates.values())}


def compare(on: dict, off: dict, protocol: dict) -> dict:
    x = on["ran"]
    y = off["ran"]
    on_rows = x["records"]
    off_rows = y["records"]
    gen2 = [e for e in x["gc_events"] if e["generation"] == 2]
    off_gen2 = [e for e in y["gc_events"] if e["generation"] == 2]
    event = max(gen2, key=lambda e: e["end_ns"] - e["begin_ns"], default=None)
    event_duration_ms = ((event["end_ns"] - event["begin_ns"]) / 1e6
                         if event else 0)
    candidates = []
    for on_row, off_row in zip(on_rows, off_rows):
        if on_row["nrx_response_ms"] is None or off_row["nrx_response_ms"] is None:
            continue
        stage = x["release_stage_records"][on_row["index"]]
        if event is None or not (event["end_ns"] >= stage["channel_prep_begin_ns"]
                and event["begin_ns"] <= on_row["nrx_observed_ns"]):
            continue
        endpoint = int(on_row["endpoint_id"][-1])
        window = min(on["workers"][endpoint]["execution_windows"],
                     key=lambda w: abs(w["observed_ns"]
                                       - on_row["nrx_dispatched_ns"]))
        candidates.append({
            "release_index": on_row["index"], "cell": on_row["cell"],
            "on_response_ms": on_row["nrx_response_ms"],
            "off_response_ms": off_row["nrx_response_ms"],
            "on_minus_off_response_ms":
                on_row["nrx_response_ms"] - off_row["nrx_response_ms"],
            "on_worker_gpu_ms": window["gpu_ms"],
            "on_worker_published_after_release_ms":
                (window["published_ns"] - on_row["release_ns"]) / 1e6,
            "on_controller_observed_after_release_ms":
                (on_row["nrx_observed_ns"] - on_row["release_ns"]) / 1e6,
            "gc_start_after_release_ms":
                (event["begin_ns"] - on_row["release_ns"]) / 1e6,
            "gc_end_after_release_ms":
                (event["end_ns"] - on_row["release_ns"]) / 1e6,
            "worker_published_before_gc_start":
                window["published_ns"] <= event["begin_ns"],
            "controller_observed_after_gc_end":
                on_row["nrx_observed_ns"] >= event["end_ns"],
        })
    selected = max(candidates, key=lambda z: z["on_minus_off_response_ms"],
                   default=None)
    feature_matches = sum(a["channel_estimate_power"]
                          == b["channel_estimate_power"]
                          for a, b in zip(on_rows, off_rows))
    admission_matches = sum(a["admitted"] == b["admitted"]
                            for a, b in zip(on_rows, off_rows))
    gate = (
        event_duration_ms >= protocol["min_on_gen2_ms"]
        and not off_gen2
        and feature_matches == len(on_rows)
        and admission_matches == len(on_rows)
        and selected is not None
        and selected["on_minus_off_response_ms"]
            >= protocol["min_matched_nrx_delay_ms"]
    )
    return {
        "on": on["name"], "off": off["name"],
        "on_gen2_max_ms": event_duration_ms,
        "off_gen2_events": len(off_gen2),
        "exact_feature_matches": feature_matches,
        "same_admission_decisions": admission_matches,
        "total_records": len(on_rows),
        "selected_matched_nrx": selected,
        "candidate_nrx_count": len(candidates),
        "frozen_mechanism_gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm89_gc_causal_replication_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = [arm_data(base, a, protocol) for a in protocol["arms"]]
    pairs = [compare(arms[0], arms[1], protocol),
             compare(arms[3], arms[2], protocol)]
    report = {
        "schema": "softwall-confirm89-gc-causal-replication-v1",
        "job": protocol["job"], "source_hashes": source_ok,
        "arm_gates": {a["name"]: a["gates"] for a in arms},
        "all_contract_gates_pass": source_ok and all(a["all_pass"] for a in arms),
        "pairs": pairs,
        "mechanism_gate_pass": all(p["frozen_mechanism_gate"] for p in pairs),
        "interpretation": "Independent prospective two-seed ON/OFF/OFF/ON replication using matched NRx response at the longest natural gen-2 GC event, with identical PHY features/admission inside each pair. Passing supports a causal GC host-latency mechanism in this synthetic mode, not hard WCET, universal tail elimination or joint-policy advantage.",
    }
    output = base / f"confirm89_gc_causal_replication_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_contract_gates_pass": report["all_contract_gates_pass"],
        "mechanism_gate_pass": report["mechanism_gate_pass"],
        "pairs": [{"on": p["on"], "off": p["off"],
                   "gc_ms": p["on_gen2_max_ms"],
                   "matched_delay_ms": p["selected_matched_nrx"]["on_minus_off_response_ms"]
                   if p["selected_matched_nrx"] else None,
                   "pass": p["frozen_mechanism_gate"]} for p in pairs],
    }, indent=2))
    if not report["all_contract_gates_pass"] or not report["mechanism_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
