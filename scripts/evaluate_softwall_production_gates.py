#!/usr/bin/env python3.11
"""Fail-closed P1--P4 production-readiness gate for SoftWall."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_p1(contract: dict | None) -> tuple[bool, list[str]]:
    failures = []
    if contract is None:
        return False, ["missing target-DU timing contract"]
    if contract.get("schema") != "softwall-du-timing-contract-v2":
        failures.append("wrong timing-contract schema")
    if contract.get("contract_kind") != "production-du":
        failures.append("contract_kind is not production-du")
    clock = contract.get("clock", {})
    if clock.get("synchronized") is not True:
        failures.append("clock is not synchronized")
    expiry = contract.get("expiry_contract", {})
    if expiry.get("synthetic") is not False:
        failures.append("expiry is synthetic or unspecified")
    if expiry.get("derived_from_ue_k2_n2") is not False:
        failures.append("expiry incorrectly derives from UE K2/N2")
    records = contract.get("records")
    if not isinstance(records, list) or not records:
        failures.append("no target-DU timing records")
    else:
        required = (
            "iq_ready_ns", "phy_submit_ns", "crc_visible_ns",
            "fapi_publish_ns", "mac_consume_ns", "expiry_ns",
        )
        for index, record in enumerate(records):
            if record.get("radio_commit_count") != 1:
                failures.append(f"record {index} does not have one radio commit")
                break
            values = [record.get(key) for key in required]
            if any(not isinstance(value, int) for value in values):
                failures.append(f"record {index} has missing timestamps")
                break
            if values != sorted(values):
                failures.append(f"record {index} timestamps or expiry are unordered")
                break
    return not failures, failures


def evaluate_p2(result: dict | None, contract_hash: str | None) -> tuple[bool, list[str]]:
    failures = []
    if result is None:
        return False, ["missing production fast-path result"]
    iterations = result.get("iterations")
    counts = result.get("deadline_counts", {})
    correctness = result.get("correctness", {})
    if not isinstance(iterations, int) or iterations <= 0:
        failures.append("invalid iteration count")
    elif counts.get("parallel_pair_wall_le_deadline") != iterations:
        failures.append("fast path has late pair completions")
    if correctness.get("neural_correct") != iterations:
        failures.append("NeuralRx correctness is incomplete")
    if correctness.get("conventional_correct") != iterations:
        failures.append("conventional correctness is incomplete")
    if contract_hash is None:
        failures.append("P1 contract hash is unavailable")
    elif result.get("timing_contract_sha256") != contract_hash:
        failures.append("fast path is not bound to the P1 timing contract")
    if result.get("campaign_role") != "independent-holdout":
        failures.append("fast path is not an independent holdout")
    return not failures, failures


def evaluate_p3(result: dict | None) -> tuple[bool, list[str]]:
    failures = []
    if result is None:
        return False, ["missing external-channel holdout"]
    if result.get("schema") != "softwall-external-channel-holdout-v1":
        failures.append("wrong external-channel schema")
    if result.get("frozen_before_execution") is not True:
        failures.append("external-channel holdout was not frozen")
    if result.get("disjoint_from_development") is not True:
        failures.append("external-channel seeds are not disjoint")
    if result.get("model_channel_contract_declared") is not True:
        failures.append("model has no declared channel contract")
    checks = result.get("checks", {})
    for key in (
        "conventional_pipeline_operational",
        "neural_pipeline_operational",
        "bler_grid_complete",
        "observable_only_policy",
    ):
        if checks.get(key) is not True:
            failures.append(f"external-channel check failed: {key}")
    if result.get("all_pass") is not True:
        failures.append("external-channel holdout did not pass")
    return not failures, failures


def evaluate_p4(
    result: dict | None,
    contract_hash: str | None,
    p2_hash: str | None,
    p3_hash: str | None,
) -> tuple[bool, list[str]]:
    failures = []
    if result is None:
        return False, ["missing integrated production requalification"]
    if result.get("schema") != "softwall-integrated-production-holdout-v1":
        failures.append("wrong integrated-holdout schema")
    references = result.get("input_sha256", {})
    expected = {
        "p1_timing_contract": contract_hash,
        "p2_fast_path": p2_hash,
        "p3_channel_holdout": p3_hash,
    }
    for key, value in expected.items():
        if value is None or references.get(key) != value:
            failures.append(f"integrated result does not bind {key}")
    checks = result.get("checks", {})
    for key in (
        "deadline_misses_zero",
        "single_commit_violations_zero",
        "credit_violations_zero",
        "correlated_all_fail_exercised",
        "bounded_qwen_exercised",
        "independent_nodes_pass",
    ):
        if checks.get(key) is not True:
            failures.append(f"integrated check failed: {key}")
    if result.get("all_pass") is not True:
        failures.append("integrated production holdout did not pass")
    return not failures, failures


def evaluate(paths: dict[str, Path | None]) -> dict:
    artifacts = {key: load(path) for key, path in paths.items()}
    hashes = {
        key: sha256(path) if path is not None and path.is_file() else None
        for key, path in paths.items()
    }
    p1, p1_failures = evaluate_p1(artifacts["p1"])
    p2, p2_failures = evaluate_p2(artifacts["p2"], hashes["p1"] if p1 else None)
    p3, p3_failures = evaluate_p3(artifacts["p3"])
    p4, p4_failures = evaluate_p4(
        artifacts["p4"],
        hashes["p1"] if p1 else None,
        hashes["p2"] if p2 else None,
        hashes["p3"] if p3 else None,
    )
    stages = {
        "P1_LIVE_DU_CLOCK": {"pass": p1, "failures": p1_failures},
        "P2_FAST_PATH": {"pass": p2, "failures": p2_failures},
        "P3_CHANNEL_COMPATIBILITY": {"pass": p3, "failures": p3_failures},
        "P4_INTEGRATED_REQUALIFICATION": {"pass": p4, "failures": p4_failures},
    }
    all_pass = all(item["pass"] for item in stages.values())
    first_blocker = next((name for name, item in stages.items() if not item["pass"]), None)
    return {
        "schema": "softwall-production-gates-v1",
        "all_pass": all_pass,
        "status": "PRODUCTION_QUALIFIED" if all_pass else "PRODUCTION_NOT_QUALIFIED",
        "first_blocker": first_blocker,
        "stages": stages,
        "artifact_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("p1", "p2", "p3", "p4"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate({name: getattr(args, name) for name in ("p1", "p2", "p3", "p4")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
