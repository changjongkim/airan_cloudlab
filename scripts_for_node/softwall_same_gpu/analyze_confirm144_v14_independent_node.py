#!/usr/bin/env python3
"""Combine independent-node V14 AI45 and three-point fault requalification."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(protocol_path, ai_path, fault_paths):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    root = protocol_path.parents[2]
    ai = json.loads(Path(ai_path).read_text())
    faults = [json.loads(Path(path).read_text()) for path in fault_paths]
    nodes = set()
    for home in ai["homes"]:
        nodes.add(json.loads(Path(home["controller"]).read_text())["host"])
    for result in faults:
        for home in result["homes"]:
            nodes.add(json.loads(Path(home["controller"]).read_text())["host"])
    source_mismatches = []
    for relative, digest in protocol["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            source_mismatches.append(relative)
    operations = [result["crash_operation"] for result in faults]
    gates = {
        "one_independent_node": len(nodes) == 1
            and nodes.isdisjoint(set(protocol["excluded_nodes"])),
        "ai45_arm_pass": ai["all_pass"],
        "ai45_exchange_observed": ai["candidate_evidence"]["exchanges"] > 0,
        "three_fault_points": sorted(operations)
            == ["commit", "complete", "prepare"],
        "all_fault_arms_pass": all(result["all_pass"] for result in faults),
        "all_radio_continues_after_fault": all(
            all(value > 0 for value in result["radio_records_after_fault_detection"])
            for result in faults
        ),
        "no_source_mismatch": not source_mismatches,
    }
    return {
        "schema": "softwall-confirm144-v14-independent-node-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "nodes": sorted(nodes),
        "ai45_result": str(Path(ai_path).resolve()),
        "fault_results": [str(Path(path).resolve()) for path in fault_paths],
        "totals": {
            "radio_records": sum(
                home["summary"]["records"] for home in ai["homes"]
            ) + sum(
                home["summary"]["records"]
                for result in faults for home in result["homes"]
            ),
            "ai45_exchanges": ai["candidate_evidence"]["exchanges"],
            "radio_after_fault": sum(
                sum(result["radio_records_after_fault_detection"])
                for result in faults
            ),
            "fault_arms": len(faults),
        },
        "source_hash_mismatches": source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Independent A100-node finite-sample requalification of one V14 AI45 "
            "conditional-exchange arm and one post-apply fault arm per broker "
            "control point. This is same-family, not cross-family or WCET evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--ai", required=True)
    parser.add_argument("--faults", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.protocol, args.ai, args.faults)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("C144 V14 independent-node gate failed")


if __name__ == "__main__":
    main()
