#!/usr/bin/env python3.11
"""Frozen CPU gate: qualified 30/12/15 durations versus a strong greedy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
from pathlib import Path

from mixed_snr_phy_value_model import TrainOnlyMixedSnrPhyValueModel
from offline_contingent_phy_ai_v2 import (
    AiUnit, Cell, Problem, exact_joint_policy, greedy_radio_then_ai,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm92_qualified_fourcell_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    if not source_ok:
        raise RuntimeError("frozen source changed")
    physical = read(base / protocol["physical_qualification"])
    if not physical["all_pass"]:
        raise RuntimeError("physical duration candidate did not qualify")
    model = TrainOnlyMixedSnrPhyValueModel(
        base / protocol["q_model"], root,
    )
    raw = read(base / "raw" / protocol["heldout_raw"])
    if raw["seed"] != protocol["heldout_seed"]:
        raise RuntimeError("held-out PHY seed mismatch")
    pool = [row for result in raw["results"] for row in result["records"]]
    rng = random.Random(protocol["sampling_seed"])
    records = []
    for index in range(protocol["cases_per_stratum"]):
        chosen = rng.sample(pool, protocol["cells"])
        values = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in chosen]
        deadlines = [rng.choice(protocol["ai_deadline_choices_ms"])
                     for _ in range(protocol["ai_units"])]
        heterogeneous_values = [rng.randint(*protocol["ai_value_range"])
                                for _ in range(protocol["ai_units"])]
        radio_floor = rng.choice(protocol["radio_floor_choices"])
        for stratum in protocol["strata"]:
            row = {"case": index, "stratum": stratum["name"],
                   "status": "out_of_support"}
            if any(value is None for value in values):
                records.append(row)
                continue
            cells = tuple(Cell(
                name=chr(ord("a") + cell),
                deadline_ms=stratum["radio_deadline_ms"]
                    - protocol["commit_guard_ms"],
                conventional_ms=protocol["conventional_bound_ms"],
                nrx_ready_ms=protocol["nrx_bound_ms"],
                neural_success_probability=values[cell].p_neural_success,
                incremental_rescue_probability=values[cell].q_rescue,
            ) for cell in range(protocol["cells"]))
            ai = tuple(AiUnit(
                name=f"ai{j}", deadline_ms=deadlines[j],
                isolated_ms=protocol["ai_bound_ms"],
                nrx_overlap_ms=protocol["ai_bound_ms"],
                value=(heterogeneous_values[j] if stratum["heterogeneous_ai_values"]
                       else 1),
            ) for j in range(protocol["ai_units"]))
            problem = Problem(cells, ai, radio_floor,
                              protocol["nrx_endpoint_capacity"])
            exact = exact_joint_policy(problem)
            greedy = greedy_radio_then_ai(problem)
            if (exact is None) != (greedy is None):
                raise AssertionError("exact/greedy feasibility mismatch")
            if exact is None:
                row["status"] = "infeasible"
            else:
                gap = exact["expected_ai_value"] - greedy["expected_ai_value"]
                if gap < -1e-9:
                    raise AssertionError("exact below greedy")
                row.update({
                    "status": "feasible",
                    "leaf_indices": [value.leaf_index for value in values],
                    "radio_floor": radio_floor,
                    "ai_deadlines_ms": deadlines,
                    "ai_values": [unit.value for unit in ai],
                    "exact_selected_nrx": exact["selected_nrx"],
                    "greedy_selected_nrx": greedy["selected_nrx"],
                    "exact_ai": exact["expected_ai_value"],
                    "greedy_ai": greedy["expected_ai_value"],
                    "gap": gap,
                    "exact_early_ai": exact["early_ai"],
                    "greedy_early_ai": greedy["early_ai"],
                    "exact_all_fail_certificate":
                        exact["all_fail_recovery_certificate"],
                    "greedy_all_fail_certificate":
                        greedy["all_fail_recovery_certificate"],
                })
            records.append(row)
    summaries = {}
    for stratum in protocol["strata"]:
        rows = [row for row in records if row["stratum"] == stratum["name"]]
        feasible = [row for row in rows if row["status"] == "feasible"]
        summaries[stratum["name"]] = {
            "total": len(rows), "feasible": len(feasible),
            "out_of_support": sum(row["status"] == "out_of_support"
                                  for row in rows),
            "positive_ai_gap": sum(row["gap"] > 1e-9 for row in feasible),
            "different_nrx_subset": sum(
                row["exact_selected_nrx"] != row["greedy_selected_nrx"]
                for row in feasible),
            "mean_exact_ai": statistics.mean(row["exact_ai"] for row in feasible)
                if feasible else None,
            "mean_greedy_ai": statistics.mean(row["greedy_ai"] for row in feasible)
                if feasible else None,
            "max_ai_gap": max((row["gap"] for row in feasible), default=None),
        }
    report = {
        "schema": "softwall-confirm92-qualified-duration-cpu-screen-v1",
        "source_hashes": source_ok,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "physical_qualification": protocol["physical_qualification"],
        "summaries": summaries, "records": records,
        "interpretation": "Prospectively frozen exploratory CPU screen. D155 uses a sample-qualified prewarmed GC-OFF four-cell 30/12/15-ms execution mode; D115 is a candidate synthetic deadline without a physical qualification. All four AI units are visible at t=0, at most one early AI is modeled, and NRx observations share one event. Any exact-versus-greedy difference is only a decision-model hypothesis, not online novelty, GPU throughput, or a hard deadline proof. Equal-value strata help expose generic AI-packing effects but are not a full mechanism ablation.",
    }
    output = base / "confirm92_qualified_fourcell_screen.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
