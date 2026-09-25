#!/usr/bin/env python3.11
"""Build the transitive artifact manifest for C161 phase-1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c161_phase1_protocol import SCRIPT_SOURCES, TASK1_SOURCES, source_hashes


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
    scripts = root / "scripts_for_node/softwall_same_gpu"
    task1 = root / "scripts_for_node/task1"
    results = root / "results/softwall_multigpu"
    raw = results / "raw"
    development = "c161p1a_dev_j58859044"
    holdout = "c161p1b_holdout_j58859145"
    combined_path = results / "c161_phase1_two_node.json"
    report_path = root / "docs/current/SOFTWALL_C161_PHASE1_RESULT_KO.md"
    c160_path = results / "c160_fault_state_model_v1.json"
    combined = load(combined_path)
    c160 = load(c160_path)
    protocols = [load(results / f"{label}_protocol.json") for label in (development, holdout)]
    files: set[Path] = {
        combined_path,
        report_path,
        c160_path,
        results / "c160_fault_state_model_manifest.json",
        root / "docs/current/SOFTWALL_C160_FAULT_MODEL_RESULT_KO.md",
        scripts / "analyze_c161_phase1_two_node.py",
        scripts / "build_c161_phase1_manifest.py",
    }
    for label in (development, holdout):
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    files.update(scripts / name for name in SCRIPT_SOURCES)
    files.update(task1 / name for name in TASK1_SOURCES)
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    files = {path for path in files if path.is_file()}
    current_hashes = source_hashes(scripts, task1)
    gates = {
        "c160_semantics_pass": c160.get("all_pass") is True,
        "two_node_combined_pass": combined.get("all_pass") is True,
        "passing_source_still_matches": (
            current_hashes == protocols[0].get("source_sha256")
            == protocols[1].get("source_sha256")
        ),
        "distinct_job_node_seed": combined.get("gates", {}).get(
            "distinct_job_node_seed"
        ) is True,
        "reverse_arm_order": combined.get("gates", {}).get(
            "reverse_arm_order"
        ) is True,
        "sample_180_rounds_720_nrx": (
            combined.get("summary", {}).get("rounds") == 180
            and combined.get("summary", {}).get("actual_nrx_requests") == 720
            and combined.get("summary", {}).get("radio_commits") == 720
        ),
        "correlated_recovery_exact": (
            combined.get("summary", {}).get("correlated_recoveries") == 240
        ),
        "physical_nonlaunch_exercised": (
            combined.get("summary", {}).get("guarded_nonlaunches", 0) > 0
        ),
        "zero_deadline_miss": (
            combined.get("summary", {}).get("deadline_misses") == 0
        ),
        "phase2_faults_explicitly_unqualified": all(
            token in combined.get("scope", "")
            for token in ("Response loss", "worker crash", "missing completion fence")
        ),
        "no_missing_selected_artifact": not missing,
    }
    value = {
        "schema": "softwall-c161-phase1-artifact-manifest-v1",
        "status": "C161_PHASE1_TWO_NODE_PASS" if all(gates.values()) else "C161_PHASE1_MANIFEST_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "passing_jobs": ["58859044", "58859145"],
        "passing_nodes": ["nid001824", "nid001025"],
        "passing_labels": [development, holdout],
        "claim_scope": combined["scope"],
        "missing": missing,
        "file_count": len(files),
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files, key=str)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "file_count": value["file_count"], "missing": missing,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
