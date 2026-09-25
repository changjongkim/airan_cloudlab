#!/usr/bin/env python3.11
"""Build the transitive artifact manifest for C159-Q1 P180 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm159_q1_protocol import SCRIPT_SOURCES, TASK1_SOURCES, source_hashes


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
    development = "confirm159q1a6_dev_j58857317"
    holdout = "confirm159q1b_holdout_j58857402"
    combined_path = results / "confirm159_q1_p180_two_node.json"
    report_path = root / "docs" / "current" / "SOFTWALL_CONFIRM159_Q1_P180_RESULT_KO.md"
    combined = load(combined_path)
    dev_protocol = load(results / f"{development}_protocol.json")
    hold_protocol = load(results / f"{holdout}_protocol.json")

    development_labels = (
        "confirm159q1_prestaged_attempt1_job58857067",
        "confirm159q1_prestaged_attempt2_pilot_job58857117",
        "confirm159q1_prestaged_attempt3_development_job58857165",
        "confirm159q1a4_j58857203",
        "confirm159q1a5_pilot_j58857317",
        "confirm159q1a6_dev_j58857317",
    )
    files = {
        combined_path,
        report_path,
        results / "confirm159_experiment_prespec_v1.json",
        results / "confirm159q1_attempt1_prelaunch_failure.json",
        results / "confirm159q1_attempt3_prelaunch_failure.json",
        root / "data" / "current" / "softwall_c159_calibration_burst_windows_v1.json",
        root / "docs" / "current" / "SOFTWALL_CONFIRM159_TRACE_PROTOCOL_KO.md",
    }
    for label in development_labels + (holdout,):
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    files.update(scripts / name for name in SCRIPT_SOURCES)
    files.update(task1 / name for name in TASK1_SOURCES)
    files.update({
        scripts / "analyze_confirm159_q1_two_node.py",
        scripts / "build_confirm159_q1_manifest.py",
    })
    missing = sorted(str(path.relative_to(root)) for path in files if not path.is_file())
    files = sorted((path for path in files if path.is_file()), key=str)
    current_hashes = source_hashes(scripts, task1)
    gates = {
        "two_node_combined_pass": combined.get("all_pass") is True,
        "passing_source_still_matches": (
            current_hashes == dev_protocol.get("source_sha256")
            == hold_protocol.get("source_sha256")
        ),
        "distinct_job_node_seed": combined.get("gates", {}).get(
            "distinct_job_node_seed"
        ) is True,
        "combined_sample_2000": combined.get("summary", {}).get(
            "actual_nrx_requests"
        ) == 2000,
        "zero_safety_transport_contract_violation": all(
            combined.get("summary", {}).get(key) == 0 for key in (
                "deadline_misses", "input_roundtrip_violations",
                "response_echo_violations", "recovery_contract_violations",
            )
        ),
        "no_missing_selected_artifact": not missing,
    }
    value = {
        "schema": "softwall-confirm159-q1-artifact-manifest-v1",
        "status": (
            "C159_Q1_TWO_NODE_P180_PASS"
            if all(gates.values()) else "C159_Q1_MANIFEST_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "passing_jobs": ["58857317", "58857402"],
        "passing_nodes": ["nid001177", "nid001308"],
        "passing_labels": [development, holdout],
        "selected_development_labels": list(development_labels),
        "artifact_history_limitation": (
            "Attempt1 failed before launch because the peer descriptor schema "
            "was incompatible; attempt2 was a 10-epoch development pilot; "
            "attempt3 failed before launch because its AF_UNIX path was too "
            "long; attempt4 completed 250 epochs but failed one warm-recovery "
            "bound sample because only the receiver, rather than the complete "
            "P2P/window/cuPHY/response path, was preflighted. Attempt5 verified "
            "the lifecycle correction on 20 epochs. Only attempt6 and the "
            "independent holdout are authoritative passing campaigns."
        ),
        "missing": missing,
        "file_count": len(files),
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
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "file_count": value["file_count"], "missing": missing,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
