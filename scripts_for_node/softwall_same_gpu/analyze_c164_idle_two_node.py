#!/usr/bin/env python3.11
"""Combine the frozen C164 idle30 development and independent holdout."""

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
    dev_c164 = development_protocol.get("c164", {})
    hold_c164 = holdout_protocol.get("c164", {})
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
    dev_node = development["host"]
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
        "frozen_c164_source_equality": (
            bool(dev_c164.get("source_sha256"))
            and dev_c164.get("source_sha256") == hold_c164.get("source_sha256")
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
            dev_node in holdout_protocol.get("excluded_nodes", [])
        ),
        "same_lifecycle_contract": (
            dev_c164.get("lifecycle") == hold_c164.get("lifecycle")
                == "idle_30s_first"
            and dev_c164.get("required_idle_s")
                == hold_c164.get("required_idle_s") == 30.0
        ),
        "both_idle_intervals_observed": (
            development.get("observed_idle_s", 0) >= 30
            and holdout.get("observed_idle_s", 0) >= 30
        ),
        "combined_safety": (
            counts["deadline_misses"] == 0
            and counts["radio_commits"] == counts["actual_nrx"]
            and all(development["gates"].values())
            and all(holdout["gates"].values())
        ),
    }
    return {
        "schema": "softwall-c164-idle30-two-node-result-v1",
        "status": "C164_IDLE30_TWO_NODE_PASS" if all(gates.values()) else "C164_IDLE30_TWO_NODE_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "nodes": [development["host"], holdout["host"]],
        "jobs": [development["slurm_job_id"], holdout["slurm_job_id"]],
        "rounds": (
            development_protocol["iterations"] + holdout_protocol["iterations"]
        ),
        "observed_idle_s": [development["observed_idle_s"],
                            holdout["observed_idle_s"]],
        "counts": counts, "maxima_ms": maxima,
        "claim_boundary": (
            "Two-node finite-sample first-request-after-30s-idle boundary "
            "qualification. Full six-class lifecycle vector, restart, longer "
            "idle, production d_MAC, cross-family, and WCET remain UQ."
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
