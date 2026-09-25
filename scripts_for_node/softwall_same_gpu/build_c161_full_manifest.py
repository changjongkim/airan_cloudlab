#!/usr/bin/env python3.11
"""Build transitive manifest for C160/C161 full fault qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c161_phase2_protocol import PHASE2_SOURCES, TASK1_SOURCES, source_hashes


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
    development = "c161p2d_dev_j58859872"
    holdout = "c161p2e_holdout_j58859986"
    full_path = results / "c161_full_fault_qualification.json"
    phase2_path = results / "c161_phase2_two_node.json"
    report_path = root / "docs/current/SOFTWALL_C161_PHASE2_RESULT_KO.md"
    full = load(full_path)
    phase2 = load(phase2_path)
    protocols = [load(results / f"{label}_protocol.json") for label in (development, holdout)]
    failures = {
        results / "c161p2a_quarantine_sentinel_failure.json",
        results / "c161p2b_marker_deadline_audit_failure.json",
        results / "c161p2c_first_round_nrx_bound_failure.json",
    }
    files: set[Path] = {
        full_path, phase2_path, report_path,
        results / "c160_fault_state_model_v1.json",
        results / "c160_fault_state_model_manifest.json",
        results / "c161_phase1_two_node.json",
        results / "c161_phase1_manifest.json",
        root / "docs/current/SOFTWALL_C160_FAULT_MODEL_RESULT_KO.md",
        root / "docs/current/SOFTWALL_C161_PHASE1_RESULT_KO.md",
        scripts / "build_c161_full_qualification.py",
        scripts / "build_c161_full_manifest.py",
    } | failures
    for label in (
        "c161p2a_dev_j58859663", "c161p2b_dev_j58859663",
        "c161p2c_dev_j58859663", development, holdout,
    ):
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    for protocol in protocols:
        for arm in protocol["arms"]:
            files.update(Path(arm["completion_dir"]).glob("*.json"))
    files.update(scripts / name for name in PHASE2_SOURCES)
    files.update(task1 / name for name in TASK1_SOURCES)
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    files = {path for path in files if path.is_file()}
    current = source_hashes(scripts, task1)
    phase1_manifest = load(results / "c161_phase1_manifest.json")
    c160_manifest = load(results / "c160_fault_state_model_manifest.json")
    gates = {
        "c160_manifest_pass": c160_manifest.get("all_pass") is True,
        "phase1_manifest_pass": phase1_manifest.get("all_pass") is True,
        "phase2_two_node_pass": phase2.get("all_pass") is True,
        "full_a0_a6_pass": full.get("all_pass") is True,
        "passing_phase2_source_still_matches": (
            current == protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
        ),
        "distinct_node_reverse_order": (
            phase2["gates"]["distinct_job_node_seed"] is True
            and phase2["gates"]["reverse_arm_order"] is True
        ),
        "failure_history_preserved": all(path.is_file() for path in failures),
        "node_bound_failure_scoped_uq": (
            full.get("gates", {}).get("node_lifecycle_failure_not_promoted") is True
        ),
        "zero_combined_deadline_miss": full["summary"]["deadline_misses"] == 0,
        "no_missing_selected_artifact": not missing,
    }
    value = {
        "schema": "softwall-c161-full-fault-manifest-v1",
        "status": "C161_FULL_QUALIFIED_NODE_PASS" if all(gates.values()) else "C161_FULL_MANIFEST_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "passing_jobs": ["58859044", "58859145", "58859872", "58859986"],
        "passing_nodes": ["nid001824", "nid001025", "nid001145", "nid001069"],
        "unqualified_observed_node": "nid001044",
        "failure_artifacts": [str(path.relative_to(root)) for path in sorted(failures)],
        "claim_scope": full["claim_scope"],
        "missing": missing,
        "file_count": len(files),
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
            for path in sorted(files, key=str)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "file_count": value["file_count"], "missing": missing},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
