#!/usr/bin/env python3.11
"""Frozen paired audit for the two-cell online channel-gate ingredient."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_cluster_interval(always: list[dict], gate: list[dict], seed: int) -> list[float]:
    per_release = [
        sum(int(gate[2 * index + cell]["correct"]) - int(always[2 * index + cell]["correct"])
            for cell in range(2))
        for index in range(len(always) // 2)
    ]
    randomizer = random.Random(seed)
    n = len(per_release)
    draws = []
    for _ in range(4000):
        total = sum(per_release[randomizer.randrange(n)] for _ in range(n))
        draws.append(100.0 * total / (2 * n))
    draws.sort()
    return [draws[100], draws[3899]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = read(result / "confirm58_online_gate_abba_protocol.json")
    arms: dict[tuple[int, str], tuple[dict, dict, list[dict], dict]] = {}
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    threshold = protocol["threshold_from_confirm57_train"]
    count = protocol["iterations_per_arm"]
    for pair in (1, 2):
        for mode in ("always", "threshold"):
            label = f"pair{pair}_{mode}"
            prefix = f"confirm58_{label}_job{args.job}"
            ran = read(raw / f"{prefix}_controller.json")
            workers = [read(raw / f"{prefix}_worker{cell}.json") for cell in range(2)]
            background = read(raw / f"{prefix}_background.json")
            requal = read(raw / f"{prefix}_requalification.json")
            records = ran["records"]
            arms[(pair, mode)] = (ran, background, records, requal)
            gates[label + "_contract"] = (
                ran["iterations"] == count
                and ran["cells"] == 2
                and ran["period_ms"] == protocol["period_ms"]
                and ran["deadline_ms"] == protocol["deadline_ms"]
                and ran["snr_db"] == protocol["snr_db"]
                and ran["noise_reference"] == "pre_fading"
                and ran["gate_mode"] == mode
                and ran["gate_threshold"] == threshold
                and ran["payload_seed"] == protocol["pairs"][str(pair)]["payload_seed"]
                and ran["channel_seed_base"] == protocol["pairs"][str(pair)]["channel_seed_base"]
                and len(records) == 2 * count
                and requal["iterations"] == 200
                and requal["deadline_misses"] == 0
            )
            gates[label + "_same_node_job"] = (
                len({item["host"] for item in (ran, *workers, background, requal)}) == 1
                and all(str(item["slurm_job_id"]) == args.job
                        for item in (ran, *workers, background, requal))
                and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
                and int(background["mps_active_thread_percentage"]) == 20
            )
            gates[label + "_features_online"] = all(
                row["index"] == position // 2
                and row["cell"] == position % 2
                and row["channel_seed"] == ran["channel_seed_base"] + position
                and row["feature_begin_ns"] >= row["release_ns"]
                and row["feature_return_ns"] >= row["feature_begin_ns"]
                and abs(row["feature_host_ms"] - (
                    row["feature_return_ns"] - row["feature_begin_ns"]
                ) / 1e6) < 1e-9
                and row["gate_skipped"] == (
                    mode == "threshold" and row["channel_estimate_power"] >= threshold
                )
                and row["commit_return_ns"] >= row["feature_return_ns"]
                for position, row in enumerate(records)
            )
            gates[label + "_safe_terminal"] = (
                ran["deadline_misses"] == 0
                and ran["nrx_bound_violations"] == 0
                and ran["conv_path_bound_violations"] == 0
                and ran["background_budget_violations"] == 0
                and ran["background_release_crossings"] == 0
                and not ran["background_faults"]
                and not ran["endpoint_faults"]
                and ran["fallback_calendar_final"]["outstanding"] == 0
                and all(item["outstanding"] == 0 for item in ran["endpoint_final"].values())
                and ran["background_units"] == background["completed_units"]
                and all(row["commit_kind"] in ("nrx", "conventional") for row in records)
                and all(worker["completed_units"] == ran["warmup"] + 1 + sum(
                    row["cell"] == cell and row["admitted"] for row in records
                ) for cell, worker in enumerate(workers))
            )
    comparisons = []
    for pair in (1, 2):
        always, _, always_rows, _ = arms[(pair, "always")]
        gated, _, gate_rows, _ = arms[(pair, "threshold")]
        same_trace = all(
            a["index"] == b["index"]
            and a["cell"] == b["cell"]
            and a["channel_seed"] == b["channel_seed"]
            for a, b in zip(always_rows, gate_rows)
        )
        gates[f"pair{pair}_same_trace"] = same_trace
        radio_pp = 100.0 * (gated["correct_cells"] - always["correct_cells"]) / (2 * count)
        ai_gain_pct = 100.0 * (gated["background_units"] - always["background_units"]) / always["background_units"]
        ci = paired_cluster_interval(always_rows, gate_rows, 20590000 + pair)
        gates[f"pair{pair}_radio_noninferiority"] = ci[0] >= protocol["frozen_gates"]["paired_radio_ci_lower_at_least_pp"]
        gates[f"pair{pair}_ai_gain"] = ai_gain_pct >= protocol["frozen_gates"]["ai_gain_at_least_percent"]
        comparisons.append({
            "pair": pair,
            "always_correct": always["correct_cells"],
            "gate_correct": gated["correct_cells"],
            "radio_difference_pp": radio_pp,
            "paired_release_bootstrap_95_ci_pp": ci,
            "always_ai_units": always["background_units"],
            "gate_ai_units": gated["background_units"],
            "ai_gain_percent": ai_gain_pct,
            "gate_skips": gated["gate_skips"],
            "feature_host_ms": gated["feature_host_ms"],
            "feature_gpu_ms": gated["feature_gpu_ms"],
        })
    report = {
        "schema": "softwall-confirm58-online-gate-abba-v1",
        "job": args.job,
        "execution_order": protocol["execution_order"],
        "comparisons": comparisons,
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": "One channel-gate ingredient with pinned endpoints and AI after radio; not the full strong combined baseline or joint policy. P150/D130 is synthetic.",
    }
    output = result / f"confirm58_online_gate_abba_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [name for name, passed in gates.items() if not passed],
                      "comparisons": comparisons}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
