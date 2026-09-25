#!/usr/bin/env python3.11
"""Gate mandatory cuPHY continuity during repeated Qwen reloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_c164_qwen_reload_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None,
                "p99": None, "max": None}
    ordered = sorted(values)
    pick = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": pick(.50), "p99": pick(.99), "max": ordered[-1]}


def overlaps(record: dict, episode: dict) -> bool:
    return (record["release_ns"] < episode["ready_ns"]
            and record["completed_ns"] > episode["launch_ns"])


def evaluate(protocol: dict, mandatory: dict, episodes: list[dict],
             inventory: list[dict], current_hashes: dict[str, str]) -> dict:
    records = mandatory.get("records", [])
    interval_rows = []
    episode_overlap_counts = []
    for episode in episodes:
        overlapping = [row for row in records if overlaps(row, episode)]
        episode_overlap_counts.append(len(overlapping))
        interval_rows.extend((episode["episode"], row) for row in overlapping)
    overlapping_ids = {row["index"] for _, row in interval_rows}
    overlap_records = [row for row in records if row["index"] in overlapping_ids]
    quiet_records = [row for row in records if row["index"] not in overlapping_ids]
    all_cells = [cell for row in records for cell in row["cell_results"]]
    same_identity = (
        mandatory.get("host")
        and all(episode.get("host") == mandatory.get("host") for episode in episodes)
        and all(episode.get("slurm_job_id") == mandatory.get("slurm_job_id")
                for episode in episodes)
    )
    expected_episodes = protocol.get("episodes")
    gates = {
        "frozen_source_hash_match": protocol.get("source_sha256") == current_hashes,
        "a100_mig_off_inventory": (
            len(inventory) == 4
            and all("A100" in row.get("name", "") for row in inventory)
            and all(row.get("mig.mode.current", "").lower() == "disabled"
                    for row in inventory)
        ),
        "node_not_excluded": mandatory.get("host") not in protocol.get("excluded_nodes", []),
        "same_host_job_clock_domain": (
            same_identity and mandatory.get("clock") == "time.perf_counter_ns"
            and all(episode.get("clock") == "time.perf_counter_ns" for episode in episodes)
        ),
        "all_reload_episodes_complete": (
            len(episodes) == expected_episodes
            and [row.get("episode") for row in episodes]
                == list(range(1, expected_episodes + 1))
            and all(row.get("all_pass") is True for row in episodes)
        ),
        "every_reload_overlaps_mandatory_releases": (
            len(episode_overlap_counts) == expected_episodes
            and all(count > 0 for count in episode_overlap_counts)
        ),
        "quiet_control_observed": bool(quiet_records),
        "optional_inference_admission_zero": all(
            row.get("qwen", {}).get("completed_units") == 0
            for row in episodes
        ),
        "mandatory_sample_complete": (
            mandatory.get("completed_iterations") == len(records)
            and len(records) >= 30
            and mandatory.get("stop_observed") is True
            and mandatory.get("error") is None
        ),
        "mandatory_correct_and_single_release_complete": (
            bool(records) and all(row.get("correct") is True for row in records)
            and mandatory.get("correct_releases") == len(records)
        ),
        "mandatory_deadline_d155": (
            mandatory.get("deadline_misses") == 0
            and all(row.get("deadline_miss") is False for row in records)
            and max((row["response_ms"] for row in records), default=1e9) <= 155
        ),
        "per_cell_component_bound25": (
            mandatory.get("component_bound_violations") == 0
            and bool(all_cells)
            and all(cell.get("component_bound_violation") is False
                    and cell.get("gpu_ms", 1e9) <= 25 for cell in all_cells)
        ),
    }
    return {
        "schema": "softwall-c164-qwen-reload-result-v1",
        "status": "C164_QWEN_RELOAD_MANDATORY_PASS" if all(gates.values()) else "C164_QWEN_RELOAD_MANDATORY_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "campaign": protocol.get("campaign"), "label": protocol.get("label"),
        "host": mandatory.get("host"),
        "slurm_job_id": mandatory.get("slurm_job_id"),
        "counts": {
            "reload_episodes": len(episodes),
            "mandatory_releases": len(records),
            "overlap_releases": len(overlap_records),
            "quiet_releases": len(quiet_records),
            "cell_decodes": len(all_cells),
            "deadline_misses": mandatory.get("deadline_misses"),
            "component_bound_violations": mandatory.get("component_bound_violations"),
            "optional_inference_units": sum(
                row.get("qwen", {}).get("completed_units", 0)
                for row in episodes
            ),
        },
        "episode_overlap_counts": episode_overlap_counts,
        "reload_ms": summarize([row["load_and_warmup_ms"] for row in episodes]),
        "response_ms": {
            "all": summarize([row["response_ms"] for row in records]),
            "reload_overlap": summarize([row["response_ms"] for row in overlap_records]),
            "quiet": summarize([row["response_ms"] for row in quiet_records]),
        },
        "component_gpu_ms": {
            "all": summarize([cell["gpu_ms"] for cell in all_cells]),
            "reload_overlap": summarize([
                cell["gpu_ms"] for row in overlap_records
                for cell in row["cell_results"]
            ]),
            "quiet": summarize([
                cell["gpu_ms"] for row in quiet_records
                for cell in row["cell_results"]
            ]),
        },
        "claim_boundary": (
            "Finite-sample mandatory four-cell continuity while fresh Qwen "
            "processes load and warm on the same MPS GPU. Optional inference "
            "remains closed. This is not optional-work availability, a Qwen "
            "reload bound, production d_MAC, or WCET qualification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mandatory", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    mandatory = load(args.mandatory)
    episodes = [load(Path(path)) for path in protocol["event_paths"]]
    with args.inventory.open(newline="") as handle:
        inventory = [{key.strip(): value.strip() for key, value in row.items()}
                     for row in csv.DictReader(handle)]
    value = evaluate(protocol, mandatory, episodes, inventory,
                     source_hashes(args.scripts_root, args.task1_root))
    value["artifact_sha256"] = {
        "protocol": sha256(args.protocol), "mandatory": sha256(args.mandatory),
        "inventory": sha256(args.inventory),
        "episodes": {path: sha256(Path(path)) for path in protocol["event_paths"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
