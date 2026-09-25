#!/usr/bin/env python3.11
"""Frozen paired audit isolating recovery retime from AI-before-recovery."""

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
    protocol = read(base / "confirm72_retime_ablation_protocol.json")
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        )
    }
    arms = {}
    for entry in protocol["arms"]:
        prefix = entry["prefix"]
        mode = entry["mode"]
        ran = read(base / "raw" / f"{prefix}_controller.json")
        workers = [read(base / "raw" / f"{prefix}_worker{cell}.json") for cell in range(2)]
        background = read(base / "raw" / f"{prefix}_background.json")
        requal = read(base / "raw" / f"{prefix}_requalification.json")
        arms[prefix] = ran
        radio = ran["records"]
        pre_ai = [row for row in ran["background_records"]
                  if row.get("phase") == "before_nrx_observation"]
        before_recovery = [row for row in ran["background_records"]
                           if row.get("phase") == "after_nrx_before_recovery"]
        gates[f"{prefix}_contract"] = (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == entry["payload_seed"]
            and ran["channel_seed_base"] == entry["channel_seed_base"]
            and ran["gate_mode"] == "low_threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["ai_during_nrx"] == "async_one"
            and ran["ai_aware_recovery"] == mode
            and ran["ai_deadline_ms"] == protocol["ai_deadline_ms"]
            and ran["early_mandatory"] == "on"
            and len(radio) == 2 * protocol["iterations"]
            and len(ran["recovery_decisions"]) == protocol["iterations"]
            and requal["iterations"] == 200 and requal["deadline_misses"] == 0
        )
        gates[f"{prefix}_provenance"] = (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
            and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        )
        gates[f"{prefix}_safety"] = (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["pre_radio_ai_guard_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"] and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and all(x["outstanding"] == 0 for x in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(row["commit_kind"] in ("nrx", "conventional") for row in radio)
        )
        gates[f"{prefix}_lease_evidence"] = (
            len(pre_ai) == ran["pre_radio_ai_units"]
            and len(before_recovery) == ran["ai_before_recovery_units"]
            and ran["recovery_retime_count"] == sum(
                item["retimed"] for item in ran["recovery_decisions"])
            and all(not row["fallback_guard_violation"]
                    and row["returned_ns"] + protocol["ai_guard_ns"]
                    <= row["earliest_fallback_start_ns"]
                    for row in pre_ai + before_recovery)
            and ran["ai_timely_units"] == sum(
                row["returned_ns"] <= radio[2 * row["release_index"]]["release_ns"]
                + round(protocol["ai_deadline_ms"] * 1e6)
                for row in ran["background_records"]
            )
        )
        if mode == "on":
            gates[f"{prefix}_retime_path"] = (
                ran["recovery_retime_count"] >= protocol["min_retime_on"]
                and ran["ai_before_recovery_units"] >= protocol["min_before_recovery_ai"]
                and ran["runtime_metrics"].get("lease_admitted", 0)
                == ran["ai_before_recovery_units"]
            )
        else:
            gates[f"{prefix}_lease_only_path"] = (
                ran["recovery_retime_count"] == 0
                and ran["ai_before_recovery_units"] >= protocol["min_before_recovery_ai"]
                and ran["runtime_metrics"].get("lease_admitted", 0)
                == ran["ai_before_recovery_units"]
            )
    pairs = []
    for pair in protocol["pairs"]:
        off, on = arms[pair["lease_only_prefix"]], arms[pair["on_prefix"]]
        gates[f"{pair['name']}_paired_trace"] = all(
            x["index"] == y["index"] and x["cell"] == y["cell"]
            and x["channel_seed"] == y["channel_seed"]
            and x["gate_skipped"] == y["gate_skipped"]
            for x, y in zip(off["records"], on["records"])
        )
        gates[f"{pair['name']}_radio_noninferiority"] = (
            on["correct_cells"] - off["correct_cells"]
            >= protocol["min_correct_cell_difference"]
        )
        pairs.append({
            "name": pair["name"],
            "lease_only_correct": off["correct_cells"], "on_correct": on["correct_cells"],
            "lease_only_ai_units": off["background_units"], "on_ai_units": on["background_units"],
            "lease_only_timely_ai": off["ai_timely_units"], "on_timely_ai": on["ai_timely_units"],
            "ai_gain_percent": 100 * (on["background_units"] - off["background_units"])
            / off["background_units"],
            "on_retime_count": on["recovery_retime_count"],
            "on_before_recovery_ai": on["ai_before_recovery_units"],
            "lease_only_before_recovery_ai": off["ai_before_recovery_units"],
            "lease_only_commit_p99_ms": off["commit_response_ms"]["p99"],
            "on_commit_p99_ms": on["commit_response_ms"]["p99"],
        })
    report = {
        "schema": "softwall-confirm72-recovery-retime-ablation-v1",
        "job": protocol["job"], "gates": gates,
        "all_pass": all(gates.values()), "pairs": pairs,
        "interpretation": "Both arms admit one physical AI unit before fallback after NRx observation; ON additionally retimes a sole survivor's recovery credit. Synthetic P150/D130, AI deadline 50 ms; a similar outcome would show retime itself has no material value in this loose workload. Not atomic recovery+AI transaction, joint policy, WCET, fault continuation, or production MAC expiry. AI gain is recorded without a positive gate.",
    }
    path = base / f"confirm72_retime_ablation_job{protocol['job']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [k for k, value in gates.items() if not value],
                      "pairs": pairs}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
