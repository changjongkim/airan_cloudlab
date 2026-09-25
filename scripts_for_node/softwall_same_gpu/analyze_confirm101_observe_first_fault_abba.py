#!/usr/bin/env python3.11
"""Frozen ABBA audit: does correlated recovery reduce timely Qwen work?"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm101_observe_first_fault_abba_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    prior_ok = read(base / protocol["physical_predecessor"])["all_pass"]
    arms = []
    for spec in protocol["arms"]:
        raw = base / "raw"
        prefix = spec["prefix"]
        ran = read(raw / f"{prefix}_controller.json")
        workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
        background = read(raw / f"{prefix}_background.json")
        requal = read(raw / f"{prefix}_requalification.json")
        groups = [ran["records"][4*i:4*i+4] for i in range(protocol["iterations"])]
        observe_before_mandatory = all(
            max(row["nrx_observed_ns"] for row in group if row["admitted"])
            <= min(row["fallback_actual_start_ns"] for row in group
                   if not row["admitted"])
            for group in groups
            if any(row["admitted"] for row in group)
            and any(not row["admitted"] for row in group)
        )
        timely = Counter(
            item["release_index"] for item in ran["background_records"]
            if item["returned_ns"] <= groups[item["release_index"]][0]["release_ns"]
            + protocol["ai_deadline_ms"] * 1_000_000
        )
        nrx = [row["nrx_response_ms"] for row in ran["records"]
               if row["nrx_response_ms"] is not None]
        conv = [row["conventional_host_path_ms"] for row in ran["records"]
                if row["conventional_host_path_ms"] is not None]
        ai = [row["execution_ms"] for row in ran["background_records"]]
        safe = (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and ran["pre_radio_ai_guard_violations"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and not ran["gc_events"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and all(x["outstanding"] == 0 for x in ran["endpoint_final"].values())
            and max(nrx) <= 45 and max(conv) <= 12 and max(ai) <= 50
            and ran["background_units"] == background["completed_units"]
            and observe_before_mandatory
        )
        contract = (
            ran["schema"] == "softwall-four-cell-gc-intervention-v1"
            and ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 4 and ran["period_ms"] == 180
            and ran["deadline_ms"] == 155 and ran["ai_deadline_ms"] == 153
            and ran["nrx_bound_ms"] == 45 and ran["conv_bound_ms"] == 12
            and ran["ai_budget_ms"] == 50 and ran["ai_rpc_timeout_ms"] == 45
            and ran["gc_mode"] == "off"
            and ran["inject_correlated_failure_every"] == spec["failure_every"]
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and len(ran["records"]) == 4 * protocol["iterations"]
            and requal["iterations"] == 200 and requal["deadline_misses"] == 0
            and background["model"] == "Qwen/Qwen2.5-1.5B"
            and background["phase"] == "prefill"
            and background["context_length"] == 16
            and background["batch_size"] == 1
            and str(background["mps_active_thread_percentage"]) == "20"
        )
        provenance = (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and ran["host"] == protocol["node"]
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
        )
        arms.append({
            "name": spec["name"], "failure_every": spec["failure_every"],
            "contract": contract, "provenance": provenance, "sample_safety": safe,
            "observe_before_mandatory": observe_before_mandatory,
            "workload_coverage": len(ai) >= protocol["min_ai_units_per_arm"],
            "nrx_max_ms": max(nrx), "conv_max_ms": max(conv),
            "ai_host_max_ms": max(ai), "timely_ai_total": sum(timely.values()),
            "groups": groups, "timely": timely,
        })
    pairs = []
    for nofault_index, fault_index in ((0, 1), (3, 2)):
        nofault = arms[nofault_index]
        fault = arms[fault_index]
        eligible = []
        matched_phy = True
        for i in range(0, protocol["iterations"], 5):
            left, right = nofault["groups"][i], fault["groups"][i]
            same_input = all(
                (a["channel_seed"], a["channel_estimate_power"], a["nrx_correct"])
                == (b["channel_seed"], b["channel_estimate_power"], b["nrx_correct"])
                for a, b in zip(left, right)
            )
            matched_phy &= same_input
            if (same_input
                    and [a["admitted"] for a in left] == [b["admitted"] for b in right]
                    and sum(a["admitted"] for a in left) == 2):
                eligible.append(i)
        differences = [fault["timely"][i] - nofault["timely"][i]
                       for i in eligible]
        pairs.append({
            "nofault": nofault["name"], "fault": fault["name"],
            "matched_phy": matched_phy, "eligible_two_nrx_indices": len(eligible),
            "nofault_timely_ai": sum(nofault["timely"][i] for i in eligible),
            "fault_timely_ai": sum(fault["timely"][i] for i in eligible),
            "fault_minus_nofault_ai": sum(differences),
            "fault_minus_nofault_mean_per_release": statistics.mean(differences) if differences else None,
            "difference_histogram": dict(sorted(Counter(differences).items())),
            "coverage": len(eligible) >= protocol["min_eligible_per_pair"],
            "directional_mechanism": bool(differences) and sum(differences) < 0,
        })
    report = {
        "schema": "softwall-confirm101-observe-first-fault-abba-v1",
        "job": protocol["job"], "node": protocol["node"],
        "source_hashes": source_ok, "physical_predecessor_pass": prior_ok,
        "arms": [{k: v for k, v in arm.items() if k not in ("groups", "timely")}
                 for arm in arms],
        "pairs": pairs,
        "safety_and_coverage_pass": (
            source_ok and prior_ok
            and all(a["contract"] and a["provenance"] and a["sample_safety"]
                    and a["workload_coverage"] for a in arms)
            and all(p["matched_phy"] and p["coverage"] for p in pairs)
        ),
        "directional_mechanism_both_pairs": all(p["directional_mechanism"] for p in pairs),
        "interpretation": "Fault vs no-fault ABBA on identical synthetic PHY inputs tests whether correlated conventional recovery reduces timely Qwen returns under the separately qualified observe-first controller. It is not a policy comparison, six-job arrival trace, AI-RAN application, WCET, or proof of conditional optimality.",
    }
    (base / f"confirm101_observe_first_fault_abba_job{protocol['job']}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "safety_and_coverage_pass": report["safety_and_coverage_pass"],
        "directional_mechanism_both_pairs": report["directional_mechanism_both_pairs"],
        "pairs": pairs,
    }, indent=2))
    if not report["safety_and_coverage_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
