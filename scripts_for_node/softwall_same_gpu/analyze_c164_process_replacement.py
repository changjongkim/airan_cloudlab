#!/usr/bin/env python3.11
"""Gate one physical MPS process-replacement campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_c164_process_replacement_protocol import source_hashes


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    if not values: return {"count": 0, "mean": None, "p50": None, "p99": None, "max": None}
    values = sorted(values); pick = lambda q: values[round((len(values)-1)*q)]
    return {"count": len(values), "mean": sum(values)/len(values),
            "p50": pick(.5), "p99": pick(.99), "max": values[-1]}


def evaluate(protocol: dict, mandatory: dict, inventory: list[dict],
             hashes: dict[str, str]) -> dict:
    audited = []
    for spec in protocol["episodes"]:
        row = {name: load(spec[name]) for name in (
            "ready", "client", "journal", "before", "termination",
            "exit", "after", "certificate",
        )}
        row["spec"] = spec; audited.append(row)
    final_ready = load(protocol["final_worker"]["ready"])
    final_output = load(protocol["final_worker"]["output"])
    mandatory_rows = mandatory.get("records", [])
    cells = [cell for row in mandatory_rows for cell in row["cell_results"]]
    expected = len(protocol["episodes"])
    pair_counts = {}
    for row in audited:
        spec = row["spec"]
        key = f"{spec['fault_mode']}:{spec['context_length']}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    target_pids = [row["termination"]["target"]["pid"] for row in audited]
    servers = [row["termination"]["target"]["server_pid"] for row in audited]
    gates = {
        "frozen_source_hash_match": protocol.get("source_sha256") == hashes,
        "a100_mig_off_inventory": len(inventory) == 4
            and all("A100" in row.get("name", "") for row in inventory)
            and all(row.get("mig.mode.current", "").lower() == "disabled" for row in inventory),
        "node_not_excluded": mandatory.get("host") not in protocol.get("excluded_nodes", []),
        "complete_balanced_episode_product": len(audited) == expected
            and len(pair_counts) == 4
            and set(pair_counts.values()) == {protocol["mode"]["expected_per_pair"]},
        "same_host_job_and_epoch_chain": all(
            row["ready"].get("host") == mandatory.get("host")
            and row["ready"].get("slurm_job_id") == mandatory.get("slurm_job_id")
            and row["ready"].get("worker_epoch") == row["spec"]["old_worker_epoch"]
            and row["journal"].get("recovered_by_worker_epoch") == row["spec"]["new_worker_epoch"]
            for row in audited
        ),
        "each_replacement_worker_consumes_predecessor_terminal": all(
            (index == 0 or row["ready"].get("predecessor_journal_valid") is True)
            for index, row in enumerate(audited)
        ) and final_ready.get("predecessor_journal_valid") is True,
        "every_channel_loss_observed": all(
            row["client"].get("channel_loss_observed") is True for row in audited
        ),
        "every_quiescence_certificate_passes": all(
            row["certificate"].get("all_pass") is True for row in audited
        ),
        "every_mps_termination_returns_zero": all(
            row["termination"].get("all_pass") is True
            and row["termination"].get("result_code") == 0 for row in audited
        ),
        "old_process_exits_and_client_disappears": all(
            row["exit"].get("all_pass") is True
            and not row["after"].get("target_match") for row in audited
        ),
        "same_mps_server_and_mandatory_client_survive": all(
            row["certificate"]["evidence"].get("same_mps_server_survived") is True
            and row["certificate"]["evidence"].get("mandatory_client_preserved") is True
            for row in audited
        ),
        "journal_terminal_transition_exact": all(
            row["journal"].get("identity", {}).get("token") == row["spec"]["token"]
            and row["journal"].get("stage") == (
                "nonlaunch_fenced" if row["spec"]["fault_mode"] == "drop_after_prepare"
                else "quiescence_fenced"
            )
            and row["journal"].get("physical_launch_count") == int(
                row["spec"]["fault_mode"] == "drop_after_launch"
            )
            and row["journal"].get("quiescence_fence_count") == 1
            for row in audited
        ),
        "launched_branch_physically_submitted": all(
            row["spec"]["fault_mode"] != "drop_after_launch"
            or row["certificate"]["journal_before"].get("physical_submission_observed") is True
            for row in audited
        ),
        "mandatory_sample_correct_deadline_and_bound": (
            len(mandatory_rows) == mandatory.get("completed_iterations")
            and len(mandatory_rows) >= 30 and mandatory.get("stop_observed") is True
            and mandatory.get("error") is None and mandatory.get("deadline_misses") == 0
            and mandatory.get("component_bound_violations") == 0
            and all(row.get("correct") is True and row.get("response_ms", 1e9) <= 155
                    for row in mandatory_rows)
            and all(cell.get("gpu_ms", 1e9) <= 25 for cell in cells)
        ),
        "distinct_target_processes": len(set(target_pids)) == expected,
        "single_surviving_mps_server_epoch": len(set(servers)) == 1,
        "final_replacement_worker_starts_and_stops_cleanly": (
            final_ready.get("worker_epoch") == protocol["final_worker"]["epoch"]
            and final_ready.get("host") == mandatory.get("host")
            and final_output.get("clean_stop") is True
            and final_output.get("worker_epoch") == final_ready.get("worker_epoch")
        ),
    }
    return {
        "schema": "softwall-c164-process-replacement-result-v1",
        "status": "C164_PROCESS_REPLACEMENT_PHYSICAL_PASS" if all(gates.values()) else "C164_PROCESS_REPLACEMENT_PHYSICAL_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "campaign": protocol.get("campaign"), "label": protocol.get("label"),
        "host": mandatory.get("host"), "slurm_job_id": mandatory.get("slurm_job_id"),
        "counts": {"episodes": expected,
                   "prepared_replacements": sum(r["spec"]["fault_mode"] == "drop_after_prepare" for r in audited),
                   "launched_replacements": sum(r["spec"]["fault_mode"] == "drop_after_launch" for r in audited),
                   "terminate_client_success": sum(r["termination"].get("result_code") == 0 for r in audited),
                   "mandatory_releases": len(mandatory_rows), "cell_decodes": len(cells),
                   "deadline_misses": mandatory.get("deadline_misses"),
                   "component_bound_violations": mandatory.get("component_bound_violations")},
        "pair_counts": pair_counts,
        "terminate_ms": summarize([(r["termination"]["returned_ns"]-r["termination"]["started_ns"])/1e6 for r in audited]),
        "mandatory_response_ms": summarize([r["response_ms"] for r in mandatory_rows]),
        "mandatory_component_gpu_ms": summarize([c["gpu_ms"] for c in cells]),
        "claim_boundary": (
            "Finite-sample MPS-assisted Qwen worker process replacement while "
            "mandatory four-cell cuPHY continues. It relies on Legacy MPS v2 "
            "terminate_client success; arbitrary GPU/driver failure, production "
            "d_MAC, WCET and optional-service availability during replacement remain UQ."
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
    protocol, mandatory = load(args.protocol), load(args.mandatory)
    with args.inventory.open(newline="") as handle:
        inventory = [{k.strip(): v.strip() for k,v in row.items()}
                     for row in csv.DictReader(handle)]
    value = evaluate(protocol, mandatory, inventory,
                     source_hashes(args.scripts_root, args.task1_root))
    value["artifact_sha256"] = {"protocol": sha256(args.protocol),
                                "mandatory": sha256(args.mandatory),
                                "inventory": sha256(args.inventory)}
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
