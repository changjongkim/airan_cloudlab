#!/usr/bin/env python3.11
"""Gate one physical C164 same-worker channel reconnect campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_c164_reconnect_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p99": None, "max": None}
    values = sorted(values)
    pick = lambda q: values[round((len(values) - 1) * q)]
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": pick(.5), "p99": pick(.99), "max": values[-1]}


def evaluate(protocol: dict, mandatory: dict, worker: dict, client: dict,
             inventory: list[dict], hashes: dict[str, str]) -> dict:
    token_rows = client.get("records", [])
    journals = worker.get("journal_records", [])
    mandatory_rows = mandatory.get("records", [])
    cells = [cell for row in mandatory_rows for cell in row["cell_results"]]
    expected = int(protocol["tokens"])
    expected_half = int(protocol["mode"]["expected_per_fault"])
    bounds = {int(k): float(v) for k, v in protocol["mode"]["ai_bounds_ms"].items()}
    prepare = [row for row in token_rows if row.get("fault_mode") == "drop_after_prepare"]
    fenced = [row for row in token_rows if row.get("fault_mode") == "drop_after_fence"]
    journal_by_token = {row["identity"]["token"]: row for row in journals}
    launch_intervals = [
        (row["record"]["launch_called_ns"], row["record"]["completed_ns"])
        for row in fenced
    ]
    overlap_releases = [
        row for row in mandatory_rows
        if any(row["started_ns"] < end and row["completed_ns"] > start
               for start, end in launch_intervals)
    ]
    pair_counts = {}
    for row in token_rows:
        key = f"{row['fault_mode']}:{row['context_length']}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    expected_quarter = int(protocol["mode"]["expected_per_context_fault"])
    gates = {
        "frozen_source_hash_match": protocol.get("source_sha256") == hashes,
        "a100_mig_off_inventory": (
            len(inventory) == 4
            and all("A100" in row.get("name", "") for row in inventory)
            and all(row.get("mig.mode.current", "").lower() == "disabled"
                    for row in inventory)
        ),
        "node_not_excluded": mandatory.get("host") not in protocol.get("excluded_nodes", []),
        "same_host_job_worker_epoch": (
            mandatory.get("host") == worker.get("host") == client.get("host")
            and mandatory.get("slurm_job_id") == worker.get("slurm_job_id")
            and worker.get("worker_epoch") == client.get("worker_epoch")
                == protocol.get("worker_epoch")
            and client.get("lifecycle_epoch") == protocol.get("lifecycle_epoch")
        ),
        "complete_balanced_token_product": (
            len(token_rows) == len(journals) == expected
            and len(prepare) == len(fenced) == expected_half
            and set(pair_counts.values()) == {expected_quarter}
            and len(pair_counts) == 4
        ),
        "every_fault_loses_channel_then_reconnects": (
            all(row.get("channel_loss_observed") is True for row in token_rows)
            and all(row.get("resolution") == "retire_terminal" for row in token_rows)
        ),
        "prepare_loss_never_launches_and_nonlaunch_fences": all(
            row["record"].get("stage") == "nonlaunch_fenced"
            and row["record"].get("journal_seq") == 2
            and row["record"].get("physical_launch_count") == 0
            and row["record"].get("physical_fence_count") == 1
            for row in prepare
        ),
        "post_fence_loss_has_one_launch_and_fence": all(
            row["record"].get("stage") == "fenced"
            and row["record"].get("journal_seq") == 3
            and row["record"].get("physical_launch_count") == 1
            and row["record"].get("physical_fence_count") == 1
            and row["record"].get("gpu_ms", 1e9) <= bounds[row["context_length"]]
            for row in fenced
        ),
        "aligned_launch_call_to_fence_spans_mandatory_release": all(
            row.get("aligned_release_ns") is not None
            and row["record"].get("launch_called_ns", 2**63)
                <= row["aligned_release_ns"]
                <= row["record"].get("completed_ns", -1)
            for row in fenced
        ),
        "duplicate_reconcile_is_idempotent": all(
            row.get("duplicate_reconcile_stable") is True for row in token_rows
        ),
        "client_worker_journal_exact_match": (
            len(journal_by_token) == expected
            and all(journal_by_token.get(row["identity"]["token"]) == row["record"]
                    for row in token_rows)
            and not worker.get("rejected")
        ),
        "qwen_launch_to_fence_and_mandatory_wall_overlap_observed": bool(overlap_releases),
        "mandatory_sample_complete": (
            len(mandatory_rows) == mandatory.get("completed_iterations")
            and len(mandatory_rows) >= 30
            and mandatory.get("stop_observed") is True
            and mandatory.get("error") is None
        ),
        "mandatory_correct_deadline_and_component_bound": (
            bool(mandatory_rows) and all(row.get("correct") is True for row in mandatory_rows)
            and mandatory.get("deadline_misses") == 0
            and mandatory.get("component_bound_violations") == 0
            and all(row.get("response_ms", 1e9) <= 155 for row in mandatory_rows)
            and all(cell.get("gpu_ms", 1e9) <= 25 for cell in cells)
        ),
    }
    return {
        "schema": "softwall-c164-reconnect-physical-result-v1",
        "status": "C164_RECONNECT_PHYSICAL_PASS" if all(gates.values()) else "C164_RECONNECT_PHYSICAL_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "campaign": protocol.get("campaign"), "label": protocol.get("label"),
        "host": mandatory.get("host"), "slurm_job_id": mandatory.get("slurm_job_id"),
        "counts": {
            "tokens": len(token_rows), "prepare_loss": len(prepare),
            "post_fence_loss": len(fenced),
            "physical_qwen_launches": sum(row["record"]["physical_launch_count"] for row in token_rows),
            "terminal_fences": sum(row["record"]["physical_fence_count"] for row in token_rows),
            "mandatory_releases": len(mandatory_rows), "cell_decodes": len(cells),
            "overlap_releases": len(overlap_releases),
            "deadline_misses": mandatory.get("deadline_misses"),
            "component_bound_violations": mandatory.get("component_bound_violations"),
        },
        "pair_counts": pair_counts,
        "qwen_gpu_ms": summarize([row["record"]["gpu_ms"] for row in fenced]),
        "mandatory_response_ms": summarize([row["response_ms"] for row in mandatory_rows]),
        "mandatory_component_gpu_ms": summarize([row["gpu_ms"] for row in cells]),
        "claim_boundary": (
            "Finite-sample same-process Qwen worker channel reconnect with "
            "fsync-backed prepared/fenced journal reconciliation and concurrent "
            "mandatory four-cell cuPHY. Overlap is launch-call-to-fence wall "
            "interval evidence, not a new kernel-overlap trace. Worker-process replacement, arbitrary "
            "crash windows, production d_MAC and WCET remain unqualified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mandatory", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.inventory.open(newline="") as handle:
        inventory = [{k.strip(): v.strip() for k, v in row.items()}
                     for row in csv.DictReader(handle)]
    value = evaluate(
        load(args.protocol), load(args.mandatory), load(args.worker),
        load(args.client), inventory,
        source_hashes(args.scripts_root, args.task1_root),
    )
    value["artifact_sha256"] = {
        name: sha256(path) for name, path in {
            "protocol": args.protocol, "mandatory": args.mandatory,
            "worker": args.worker, "client": args.client,
            "inventory": args.inventory,
        }.items()
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
