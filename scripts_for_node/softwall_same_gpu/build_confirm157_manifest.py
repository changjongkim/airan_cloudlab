#!/usr/bin/env python3.11
"""Build a transitive artifact manifest for the C157 actual-NRx campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_actual_nrx_integrated_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    scripts = root / "scripts_for_node" / "softwall_same_gpu"
    task1 = root / "scripts_for_node" / "task1"
    results = root / "results" / "softwall_multigpu"
    first_label = "confirm157a_actual_nrx_attempt7_job58854900"
    second_label = "confirm157b_actual_nrx_holdout_job58855194"
    first_protocol_path = results / f"{first_label}_protocol.json"
    second_protocol_path = results / f"{second_label}_protocol.json"
    combined_path = results / "confirm157_actual_nrx_two_node.json"
    first_protocol = load(first_protocol_path)
    second_protocol = load(second_protocol_path)
    combined = load(combined_path)
    failures = [results / f"confirm157a_attempt{index}_failure.json"
                for index in range(1, 7)]

    files = set(failures + [first_protocol_path, second_protocol_path, combined_path])
    for label in (first_label, second_label):
        files.update(results.glob(f"{label}*"))
        files.update((results / "raw").glob(f"{label}*"))
    files.update({
        root / "docs" / "current" / "SOFTWALL_CONFIRM157_ACTUAL_NRX_RESULT_KO.md",
        scripts / "actual_nrx_shared_recovery_plan_v1.py",
        scripts / "actual_nrx_integrated_owner.py",
        scripts / "actual_nrx_multi_peer_worker.py",
        scripts / "actual_nrx_integrated_coordinator.py",
        scripts / "build_actual_nrx_integrated_protocol.py",
        scripts / "analyze_actual_nrx_integrated.py",
        scripts / "analyze_confirm157_actual_nrx_two_node.py",
        scripts / "run_actual_nrx_integrated.sh",
        scripts / "test_actual_nrx_shared_recovery_plan_v1.py",
    })
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    files = sorted((path for path in files if path.is_file()), key=str)
    current_hashes = source_hashes(scripts, task1)
    gates = {
        "two_node_combined_pass": combined.get("all_pass") is True,
        "six_failures_preserved": all(path.is_file() for path in failures),
        "passing_source_still_matches": (
            current_hashes == first_protocol.get("source_sha256")
            == second_protocol.get("source_sha256")
        ),
        "distinct_job_node_seed": combined.get("gates", {}).get(
            "distinct_job_node_seed"
        ) is True,
        "no_missing_artifact": not missing,
    }
    value = {
        "schema": "softwall-confirm157-artifact-manifest-v1",
        "status": (
            "TWO_NODE_ACTUAL_NRX_TRANSITION_PASS_WARM_SYNTHETIC"
            if all(gates.values()) else "MANIFEST_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "file_count": len(files),
        "passing_jobs": ["58854900", "58855194"],
        "passing_nodes": ["nid001109", "nid001085"],
        "preserved_failure_attempts": 6,
        "missing": missing,
        "claim_scope": combined["claim_scope"],
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "file_count": value["file_count"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
