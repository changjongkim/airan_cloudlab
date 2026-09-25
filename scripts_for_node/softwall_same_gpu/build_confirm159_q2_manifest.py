#!/usr/bin/env python3.11
"""Build the transitive artifact manifest for C159-Q2 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm159_q2_protocol import SCRIPT_SOURCES, TASK1_SOURCES, source_hashes


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
    raw = results / "raw"
    development = "confirm159q2f_batch_dev_j58857672"
    holdout = "confirm159q2g_batch_holdout_j58858194"
    combined_path = results / "confirm159_q2_variable_two_node.json"
    report_path = root / "docs" / "current" / "SOFTWALL_CONFIRM159_Q2_VARIABLE_RESULT_KO.md"
    combined = load(combined_path)
    dev_protocol = load(results / f"{development}_protocol.json")
    holdout_protocol = load(results / f"{holdout}_protocol.json")
    labels = (
        "confirm159q2a_pilot_j58857672",
        "confirm159q2b_dev_j58857672",
        "confirm159q2c_guarded_pilot_j58857672",
        "confirm159q2d_guarded_dev_j58857672",
        "confirm159q2e_batch_pilot_j58857672",
        development,
        holdout,
    )
    files: set[Path] = {
        combined_path,
        report_path,
        results / "confirm159_experiment_prespec_v1.json",
        results / "confirm159q2b_launch_control_failure.json",
        results / "confirm159q2d_sequential_revalidation_failure.json",
        root / "data" / "current" / "softwall_c159_calibration_burst_windows_v1.json",
        root / "docs" / "current" / "SOFTWALL_CONFIRM159_TRACE_PROTOCOL_KO.md",
    }
    for label in labels:
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    files.update(scripts / name for name in SCRIPT_SOURCES)
    files.update(task1 / name for name in TASK1_SOURCES)
    files.update({
        scripts / "analyze_confirm159_q2_two_node.py",
        scripts / "build_confirm159_q2_manifest.py",
    })
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    files = set(path for path in files if path.is_file())
    current_hashes = source_hashes(scripts, task1)
    gates = {
        "two_node_combined_pass": combined.get("all_pass") is True,
        "passing_source_still_matches": (
            current_hashes == dev_protocol.get("source_sha256")
            == holdout_protocol.get("source_sha256")
        ),
        "distinct_job_node_seed": combined.get("gates", {}).get(
            "distinct_job_node_seed"
        ) is True,
        "combined_sample_1200_rounds_4800_nrx": (
            combined.get("summary", {}).get("rounds") == 1200
            and combined.get("summary", {}).get("actual_nrx_requests") == 4800
        ),
        "all_six_classes_qualified_per_node": combined.get("gates", {}).get(
            "each_class_offered_200_and_qualified_per_node"
        ) is True,
        "zero_safety_transport_contract_violation": all(
            combined.get("summary", {}).get(key) == 0 for key in (
                "deadline_misses", "input_roundtrip_violations",
                "response_echo_violations", "recovery_contract_violations",
            )
        ),
        "failure_history_preserved": all(
            (results / name).is_file() for name in (
                "confirm159q2b_launch_control_failure.json",
                "confirm159q2d_sequential_revalidation_failure.json",
            )
        ),
        "burstgpt_holdout_still_unmaterialized": combined.get("gates", {}).get(
            "burstgpt_holdout_still_unmaterialized"
        ) is True,
        "no_missing_selected_artifact": not missing,
    }
    value = {
        "schema": "softwall-confirm159-q2-artifact-manifest-v1",
        "status": (
            "C159_Q2_TWO_NODE_VARIABLE_P180_PASS"
            if all(gates.values()) else "C159_Q2_MANIFEST_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "passing_jobs": ["58857672", "58858194"],
        "passing_nodes": ["nid001204", "nid001632"],
        "passing_labels": [development, holdout],
        "selected_labels": list(labels),
        "artifact_history_limitation": (
            "Q2a was a 24-epoch pre-correction pilot. Q2b completed 600 epochs "
            "but failed one stale atomic lease after a 23.530143 ms control tail. "
            "Q2c verified bounded revalidation and the physical latest-start gate. "
            "Q2d stopped after 349 epochs because retry replay exposed an artificial "
            "sequential partial-outcome state. Q2e verified atomic observed-success "
            "batch transition. Only Q2f and the independent-node Q2g holdout are "
            "authoritative passing qualification campaigns."
        ),
        "missing": missing,
        "file_count": len(files),
        "claim_scope": combined["claim_scope"],
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
        "status": value["status"],
        "gates": gates,
        "file_count": value["file_count"],
        "missing": missing,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
