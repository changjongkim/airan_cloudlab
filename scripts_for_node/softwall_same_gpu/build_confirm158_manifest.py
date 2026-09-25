#!/usr/bin/env python3.11
"""Build the transitive artifact manifest for C158 repeated qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm158_repeated_protocol import SCRIPT_SOURCES, TASK1_SOURCES, source_hashes


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
    development = "confirm158a_repeated_attempt8_job58855711"
    holdout = "confirm158b_repeated_holdout_job58856474"
    combined_path = results / "confirm158_repeated_actual_nrx_two_node.json"
    report_path = root / "docs" / "current" / "SOFTWALL_CONFIRM158_REPEATED_QUALIFICATION_KO.md"
    combined = load(combined_path)
    dev_protocol = load(results / f"{development}_protocol.json")
    hold_protocol = load(results / f"{holdout}_protocol.json")

    development_labels = (
        "confirm158a_repeated_pilot1_job58855711",
        "confirm158a_repeated_job58855711",
        "confirm158a_repeated_attempt2_job58855711",
        "confirm158a_repeated_attempt3_job58855711",
        "confirm158a_repeated_attempt4_job58855711",
        "confirm158a_repeated_attempt5_job58855711",
        "confirm158a_repeated_attempt6_job58855711",
        "confirm158a_repeated_attempt7_job58855711",
        "confirm158a_repeated_attempt8_job58855711",
        "confirm158a_repeated_pass_job58855711",
        "confirm158a_repeated_final_job58855711",
    )
    files = {combined_path, report_path}
    for label in development_labels + (holdout,):
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    files.update(scripts / name for name in SCRIPT_SOURCES)
    files.update(task1 / name for name in TASK1_SOURCES)
    files.update({
        scripts / "analyze_confirm158_two_node.py",
        scripts / "build_confirm158_manifest.py",
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
        "schema": "softwall-confirm158-artifact-manifest-v1",
        "status": (
            "C158_TWO_NODE_REPEATED_ACTUAL_NRX_PASS"
            if all(gates.values()) else "C158_MANIFEST_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "passing_jobs": ["58855711", "58856474"],
        "passing_nodes": ["nid002288", "nid002100"],
        "passing_labels": [development, holdout],
        "selected_development_labels": list(development_labels),
        "artifact_history_limitation": (
            "One early failed development label, "
            "confirm158a_repeated_final_job58855711, was accidentally reused; "
            "its earlier raw version was overwritten. The authoritative "
            "attempt8 development and independent holdout labels are unique."
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
