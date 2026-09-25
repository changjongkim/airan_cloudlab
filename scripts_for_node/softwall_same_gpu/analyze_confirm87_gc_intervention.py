#!/usr/bin/env python3.11
"""Frozen ABBA audit of Python cyclic-GC ON/OFF in four-cell MPS mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_arm(base: Path, arm: dict, protocol: dict) -> dict:
    raw = base / "raw"
    prefix = arm["prefix"]
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    stages = ran["release_stage_records"]
    gc_events = ran["gc_events"]
    gen2 = [e for e in gc_events if e["generation"] == 2]
    prep_ms = [(s["channel_prep_end_ns"] - s["channel_prep_begin_ns"]) / 1e6
               for s in stages]
    wait_ms = [(s["wait_return_ns"] - s["release_ns"]) / 1e6 for s in stages]
    predipatch_ms = []
    nrx = []
    for index in range(protocol["iterations"]):
        group = records[4 * index:4 * index + 4]
        first = [r["nrx_dispatch_begin_ns"] for r in group
                 if r["nrx_dispatch_begin_ns"] is not None]
        if first:
            predipatch_ms.append((min(first) - max(r["feature_return_ns"]
                                                  for r in group)) / 1e6)
        nrx.extend(r["nrx_response_ms"] for r in group
                   if r["nrx_response_ms"] is not None)
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
            and len(stages) == protocol["iterations"]
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
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and ran["background_units"] == background["completed_units"]
        ),
    }
    max_prep = max(prep_ms)
    max_wait = max(wait_ms)
    max_predispatch = max(predipatch_ms)
    return {
        "name": arm["name"], "gc_mode": arm["gc_mode"],
        "gates": gates, "all_contract_gates_pass": all(gates.values()),
        "gc_events": len(gc_events), "gen2_gc_events": len(gen2),
        "gen2_gc_max_ms": max(((e["end_ns"] - e["begin_ns"]) / 1e6
                               for e in gen2), default=0),
        "max_channel_prep_ms": max_prep,
        "max_channel_prep_release": prep_ms.index(max_prep),
        "max_wait_lateness_ms": max_wait,
        "max_wait_lateness_release": wait_ms.index(max_wait),
        "max_predispatch_gap_ms": max_predispatch,
        "max_host_stage_ms": max(max_prep, max_wait, max_predispatch),
        "max_nrx_response_ms": max(nrx),
        "nrx_over_30_ms": sum(x > 30 for x in nrx),
        "background_units": ran["background_units"],
        "deadline_misses": ran["deadline_misses"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm87_gc_intervention_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = [summarize_arm(base, arm, protocol) for arm in protocol["arms"]]
    pairs = [(arms[0], arms[1]), (arms[3], arms[2])]
    mechanism = [
        {
            "seed_pair": index + 1,
            "on_gen2_max_ms": on["gen2_gc_max_ms"],
            "off_gen2_events": off["gen2_gc_events"],
            "on_minus_off_max_host_stage_ms":
                on["max_host_stage_ms"] - off["max_host_stage_ms"],
            "on_minus_off_max_nrx_ms":
                on["max_nrx_response_ms"] - off["max_nrx_response_ms"],
            "frozen_tail_attribution_gate": (
                on["gen2_gc_max_ms"] >= protocol["min_on_gen2_ms"]
                and off["gen2_gc_events"] == 0
                and on["max_host_stage_ms"] - off["max_host_stage_ms"]
                    >= protocol["min_host_stage_gap_ms"]
            ),
        }
        for index, (on, off) in enumerate(pairs)
    ]
    report = {
        "schema": "softwall-confirm87-gc-intervention-abba-v1",
        "job": protocol["job"], "source_hashes": source_ok,
        "arms": arms, "pairs": mechanism,
        "all_contract_gates_pass": source_ok and all(
            a["all_contract_gates_pass"] for a in arms),
        "tail_attribution_gate_pass": all(
            p["frozen_tail_attribution_gate"] for p in mechanism),
        "interpretation": "Two same-trace seed pairs in ON/OFF/OFF/ON order, each separate MPS epoch. Passing the tail attribution gate supports cyclic GC as a cause of the measured host stall in this synthetic mode, not a hard service-time guarantee. Failure leaves the causal hypothesis unconfirmed; source and safety gates are separate.",
    }
    output = base / f"confirm87_gc_intervention_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_contract_gates_pass": report["all_contract_gates_pass"],
                      "tail_attribution_gate_pass": report["tail_attribution_gate_pass"],
                      "pairs": mechanism}, indent=2))
    if not report["all_contract_gates_pass"] or not report["tail_attribution_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
