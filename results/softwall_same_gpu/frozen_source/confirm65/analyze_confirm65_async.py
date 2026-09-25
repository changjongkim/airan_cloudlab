#!/usr/bin/env python3.11
"""Frozen paired audit of synchronous versus asynchronous pre-radio AI RPC."""

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
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm65_async_protocol.json")
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
        for path, digest in protocol["source_sha256_before_run"].items()
    )
    arms = {}
    observed_before_return = {}
    for mode in ("sync_one", "async_one"):
        prefix = f"confirm65_{mode}_job{args.job}"
        ran = read(base / "raw" / f"{prefix}_controller.json")
        workers = [read(base / "raw" / f"{prefix}_worker{cell}.json") for cell in range(2)]
        background = read(base / "raw" / f"{prefix}_background.json")
        requal = read(base / "raw" / f"{prefix}_requalification.json")
        rows = ran["records"]
        pre = [row for row in ran["background_records"]
               if row.get("phase") == "before_nrx_observation"]
        arms[mode] = ran
        observed_before_return[mode] = sum(
            any(
                radio["index"] == ai["release_index"]
                and radio["admitted"]
                and radio["nrx_dispatched_ns"] <= ai["admitted_ns"]
                and radio["nrx_observed_ns"] < ai["returned_ns"]
                for radio in rows
            ) for ai in pre
        )
        gates[f"{mode}_contract"] = (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and ran["gate_mode"] == "low_threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["early_mandatory"] == "on"
            and ran["ai_during_nrx"] == mode
            and ran["noise_reference"] == "pre_fading"
            and len(rows) == 2 * protocol["iterations"]
            and requal["iterations"] == 200
            and requal["deadline_misses"] == 0
        )
        gates[f"{mode}_provenance"] = (
            len({item["host"] for item in (ran, *workers, background, requal)}) == 1
            and all(str(item["slurm_job_id"]) == args.job
                    for item in (ran, *workers, background, requal))
            and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        )
        gates[f"{mode}_safety"] = (
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
            and all(item["outstanding"] == 0 for item in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(row["commit_kind"] in ("nrx", "conventional") for row in rows)
        )
        gates[f"{mode}_pre_ai"] = (
            len(pre) == ran["pre_radio_ai_units"]
            and len(pre) >= protocol["min_pre_ai_units_per_arm"]
            and all(
                not ai["fallback_guard_violation"]
                and ai["returned_ns"] + protocol["ai_guard_ns"] <= ai["earliest_fallback_start_ns"]
                and any(
                    radio["index"] == ai["release_index"]
                    and radio["admitted"]
                    and radio["nrx_dispatched_ns"] <= ai["admitted_ns"]
                    for radio in rows
                )
                for ai in pre
            )
        )
    sync, async_run = arms["sync_one"], arms["async_one"]
    gates["paired_trace"] = all(
        a["index"] == b["index"]
        and a["cell"] == b["cell"]
        and a["channel_seed"] == b["channel_seed"]
        and a["gate_skipped"] == b["gate_skipped"]
        for a, b in zip(sync["records"], async_run["records"])
    )
    gates["async_host_observation"] = (
        observed_before_return["sync_one"] == 0
        and observed_before_return["async_one"] >= protocol["min_async_observations_before_ai_return"]
    )
    gates["radio_noninferiority"] = (
        async_run["correct_cells"] - sync["correct_cells"]
        >= protocol["min_correct_cell_difference"]
    )
    report = {
        "schema": "softwall-confirm65-async-ai-v1",
        "job": args.job,
        "gates": gates,
        "all_pass": all(gates.values()),
        "comparison": {
            "sync_correct": sync["correct_cells"],
            "async_correct": async_run["correct_cells"],
            "sync_ai_units": sync["background_units"],
            "async_ai_units": async_run["background_units"],
            "ai_gain_percent": 100 * (async_run["background_units"] - sync["background_units"]) / sync["background_units"],
            "sync_pre_ai_units": sync["pre_radio_ai_units"],
            "async_pre_ai_units": async_run["pre_radio_ai_units"],
            "sync_observed_before_ai_return": observed_before_return["sync_one"],
            "async_observed_before_ai_return": observed_before_return["async_one"],
            "sync_commit_p99_ms": sync["commit_response_ms"]["p99"],
            "async_commit_p99_ms": async_run["commit_response_ms"]["p99"],
        },
        "interpretation": "Success-path diagnostic: asynchronous RPC lets host observe some NRx results before AI completion, but joins before conventional or the earliest fallback guard. Host observation is not GPU kernel overlap. One 100-release pair cannot qualify WCET, fault continuation, production MAC expiry, or joint-policy advantage over the strong baseline.",
    }
    output = base / f"confirm65_async_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "failed": [k for k, v in gates.items() if not v], "comparison": report["comparison"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
