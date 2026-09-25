#!/usr/bin/env python3.11
"""Combine C164 MPS-restart development and independent holdout results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(development: dict, holdout: dict,
            development_protocol: dict, holdout_protocol: dict) -> dict:
    dev_contract = development_protocol.get("c164_restart", {})
    hold_contract = holdout_protocol.get("c164_restart", {})
    count_keys = ("actual_nrx", "injected_recoveries", "qwen_units",
                  "radio_commits", "deadline_misses")
    counts = {
        key: development["counts"][key] + holdout["counts"][key]
        for key in count_keys
    }
    maxima = {
        key: max(development["maxima_ms"][key], holdout["maxima_ms"][key])
        for key in development["maxima_ms"]
    }
    gates = {
        "both_campaigns_pass": (
            development.get("all_pass") is True
            and holdout.get("all_pass") is True
        ),
        "development_holdout_roles": (
            development.get("campaign") == "development"
            and holdout.get("campaign") == "holdout"
        ),
        "independent_job_and_node": (
            development.get("host") != holdout.get("host")
            and development.get("slurm_job_id") != holdout.get("slurm_job_id")
        ),
        "frozen_restart_source_equality": (
            bool(dev_contract.get("source_sha256"))
            and dev_contract.get("source_sha256")
                == hold_contract.get("source_sha256")
        ),
        "frozen_base_source_equality": (
            development_protocol.get("source_sha256")
                == holdout_protocol.get("source_sha256")
        ),
        "independent_seed_and_reverse_order": (
            development_protocol.get("seed_base")
                != holdout_protocol.get("seed_base")
            and development_protocol.get("reverse_cases") is False
            and holdout_protocol.get("reverse_cases") is True
        ),
        "development_node_excluded_from_holdout": (
            development.get("host")
                in holdout_protocol.get("excluded_nodes", [])
        ),
        "same_restart_contract": (
            dev_contract.get("lifecycle")
                == hold_contract.get("lifecycle") == "mps_restart_first"
            and dev_contract.get("restart_scope")
                == hold_contract.get("restart_scope")
        ),
        "both_physical_restart_proofs_pass": (
            development.get("gates", {}).get("physical_restart_marker_pass") is True
            and holdout.get("gates", {}).get("physical_restart_marker_pass") is True
        ),
        "combined_safety": (
            counts["deadline_misses"] == 0
            and counts["radio_commits"] == counts["actual_nrx"]
            and all(development.get("gates", {}).values())
            and all(holdout.get("gates", {}).values())
        ),
    }
    return {
        "schema": "softwall-c164-mps-restart-two-node-result-v1",
        "status": "C164_MPS_RESTART_TWO_NODE_PASS" if all(gates.values()) else "C164_MPS_RESTART_TWO_NODE_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "nodes": [development["host"], holdout["host"]],
        "jobs": [development["slurm_job_id"], holdout["slurm_job_id"]],
        "rounds": development_protocol["iterations"] + holdout_protocol["iterations"],
        "counts": counts, "maxima_ms": maxima,
        "restart_identity": {
            development["host"]: development.get("restart_identity"),
            holdout["host"]: holdout.get("restart_identity"),
        },
        "claim_boundary": (
            "Two-node finite-sample first-RAN-request boundary after a proven "
            "quiescent MPS daemon restart and full fresh-client requalification. "
            "Optional service is closed during restart. Restart availability, full "
            "six-class lifecycle qualification, production d_MAC, cross-family, "
            "and WCET remain UQ."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--holdout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = combine(
        load(args.development), load(args.holdout),
        load(args.development_protocol), load(args.holdout_protocol),
    )
    value["artifact_sha256"] = {
        "development": sha256(args.development),
        "holdout": sha256(args.holdout),
        "development_protocol": sha256(args.development_protocol),
        "holdout_protocol": sha256(args.holdout_protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
