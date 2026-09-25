#!/usr/bin/env python3.11
"""Audit actual B_NRx30/B_conv12 four-cell calendar under GC-OFF mode."""

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
    raw = base / "raw"
    protocol = read(base / "confirm91_conv12_nrx30_gc_off_protocol.json")
    prior = read(base / "confirm88_gc_off_long_bounds_job58743005.json")
    requested_nrx_bound = protocol["nrx_bound_ms"]
    go = (
        prior["all_current_contract_gates_pass"]
        and requested_nrx_bound in (25, 30)
        and prior["candidate_gates"][f"nrx{requested_nrx_bound}_observed"]
        and prior["candidate_gates"]["conv12_observed"]
        and protocol["conv_bound_ms"] == 12
    )
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = []
    for arm in protocol["arms"]:
        prefix = arm["prefix"]
        ran = read(raw / f"{prefix}_controller.json")
        workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
        background = read(raw / f"{prefix}_background.json")
        requal = read(raw / f"{prefix}_requalification.json")
        records = ran["records"]
        groups = [records[4 * i:4 * i + 4]
                  for i in range(protocol["iterations"])]
        nrx = [r["nrx_response_ms"] for r in records
               if r["nrx_response_ms"] is not None]
        conv = [r["conventional_host_path_ms"] for r in records
                if r["conventional_host_path_ms"] is not None]
        joint = [r for r in ran["background_records"]
                 if r.get("phase") == "after_nrx_before_recovery"]
        two_admitted = sum(sum(r["admitted"] for r in group) == 2
                           for group in groups)
        all_fail = sum(
            sum(r["admitted"] for r in group) == 2
            and all(r["forced_nrx_failure"] for r in group if r["admitted"])
            and all(r["commit_kind"] == "conventional" for r in group)
            for group in groups
        )
        gates = {
            "contract": (
                ran["schema"] == "softwall-four-cell-gc-intervention-v1"
                and ran["gc_mode"] == "off"
                and ran["iterations"] == protocol["iterations"]
                and ran["cells"] == 4
                and ran["period_ms"] == 180
                and ran["deadline_ms"] == 155
                and ran["nrx_bound_ms"] == protocol["nrx_bound_ms"]
                and ran["conv_bound_ms"] == protocol["conv_bound_ms"]
                and ran["ai_budget_ms"] == 15
                and ran["payload_seed"] == arm["payload_seed"]
                and ran["channel_seed_base"] == arm["channel_seed_base"]
                and len(records) == 4 * protocol["iterations"]
                and len(ran["recovery_decisions"]) == protocol["iterations"]
                and ran["runtime_metrics"].get("mandatory_reserved") == len(records)
                and all(r["channel_seed"] == arm["channel_seed_base"]
                        + r["index"] * 4 + r["cell"] for r in records)
                and requal["iterations"] == 200
                and requal["nrx_bound_ms"] == 50
                and requal["conv_bound_ms"] == 25
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
                and ran["conv_bound_violations"] == 0
                and ran["conv_path_bound_violations"] == 0
                and ran["background_budget_violations"] == 0
                and ran["background_release_crossings"] == 0
                and ran["pre_radio_ai_guard_violations"] == 0
                and not ran["background_faults"]
                and not ran["endpoint_faults"]
                and ran["fallback_calendar_final"]["outstanding"] == 0
                and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
                and all(x["outstanding"] == 0
                        for x in ran["endpoint_final"].values())
                and ran["background_units"] == background["completed_units"]
                and not ran["gc_events"]
                and max(nrx) <= protocol["nrx_bound_ms"]
                and max(conv) <= protocol["conv_bound_ms"]
                and all(r["commit_kind"] in ("nrx", "conventional")
                        for r in records)
            ),
            "physical_coverage": (
                two_admitted >= protocol["min_two_admitted"]
                and all_fail >= protocol["min_injected_all_fail"]
                and len(joint) >= protocol["min_joint_ai"]
                and len(joint) == ran["joint_lease_retired_count"]
                and len(joint)
                    == ran["runtime_metrics"].get("joint_admission_admitted", 0)
                and all(r["returned_ns"] + protocol["ai_guard_ns"]
                        <= r["earliest_fallback_start_ns"] for r in joint)
            ),
        }
        arms.append({
            "name": arm["name"], "prefix": prefix,
            "gates": gates, "all_pass": all(gates.values()),
            "nrx_paths": len(nrx), "nrx_max_ms": max(nrx),
            "conv_paths": len(conv), "conv_max_ms": max(conv),
            "two_admitted_releases": two_admitted,
            "injected_all_fail_releases": all_fail,
            "joint_ai_units": len(joint),
            "background_units": ran["background_units"],
            "deadline_misses": ran["deadline_misses"],
        })
    report = {
        "schema": "softwall-confirm91-actual-tight-calendar-gc-off-v1",
        "job": protocol["job"], "source_hashes": source_ok,
        "prior_go_condition": go,
        "arms": arms,
        "all_pass": go and source_ok and all(a["all_pass"] for a in arms),
        "interpretation": f"Two independent physical runs with an actual B_NRx{requested_nrx_bound}/B_conv12 calendar and cyclic GC disabled during timed traffic, conditional on Confirm88 observational candidates. PASS is finite synthetic-mode qualification only, not WCET, production DU, eight-cell support, or PHY-value joint-policy advantage.",
    }
    output = base / f"confirm91_conv12_nrx30_gc_off_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "prior_go_condition": go,
                      "arm_gates": [a["gates"] for a in arms],
                      "nrx_max_ms": [a["nrx_max_ms"] for a in arms],
                      "conv_max_ms": [a["conv_max_ms"] for a in arms]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
