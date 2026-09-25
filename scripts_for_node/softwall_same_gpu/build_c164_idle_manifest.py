#!/usr/bin/env python3.11
"""Build a transitive manifest for C164 idle30 failures and passing campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts_for_node/softwall_same_gpu"
    task1 = root / "scripts_for_node/task1"
    results = root / "results/softwall_multigpu"
    raw = results / "raw"
    labels = (
        "c164a3_idle30_dev_j58861564",
        "c164b_idle30_holdout_j58861662",
    )
    protocols = [load(results / f"{label}_protocol.json") for label in labels]
    combined = load(results / "c164_idle30_two_node.json")
    model = load(results / "c164_lifecycle_model_v1.json")
    failures = [
        load(results / "c164a_attempt1_allocation_entry_failure.json"),
        load(results / "c164a_attempt2_schedule_timeout_failure.json"),
    ]
    files: set[Path] = set()
    source_mismatches = []

    def register_source(key: str, expected: str) -> None:
        if key.startswith("scripts/"):
            path = scripts / key.removeprefix("scripts/")
        elif key.startswith("task1/"):
            path = task1 / key.removeprefix("task1/")
        else:
            raise ValueError(f"unknown source key {key}")
        files.add(path)
        if not path.is_file() or sha256(path) != expected:
            source_mismatches.append(key)

    for protocol in protocols:
        for key, expected in protocol["source_sha256"].items():
            register_source(key, expected)
        for key, expected in protocol["c164"]["source_sha256"].items():
            register_source(key, expected)

    extras = (
        "scripts_for_node/softwall_same_gpu/analyze_c164_idle_two_node.py",
        "scripts_for_node/softwall_same_gpu/test_analyze_c164_idle_two_node.py",
        "scripts_for_node/softwall_same_gpu/build_c164_idle_manifest.py",
        "docs/current/SOFTWALL_C164_IDLE30_RESULT_KO.md",
        "results/softwall_multigpu/c164_lifecycle_model_v1.json",
        "results/softwall_multigpu/c164_idle30_two_node.json",
        "results/softwall_multigpu/c164a_attempt1_allocation_entry_failure.json",
        "results/softwall_multigpu/c164a_attempt2_schedule_timeout_failure.json",
    )
    files.update(root / name for name in extras)
    for label in labels:
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    failure_reference_mismatches = []
    for failure in failures:
        for name, expected in failure["artifact_sha256"].items():
            path = root / name
            files.add(path)
            if not path.is_file() or sha256(path) != expected:
                failure_reference_mismatches.append(name)
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    paths = sorted(files)
    checks = {
        "combined_two_node_pass": combined.get("all_pass") is True,
        "lifecycle_model_pass": model.get("all_pass") is True,
        "passing_source_equality": (
            protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
            and protocols[0]["c164"]["source_sha256"]
                == protocols[1]["c164"]["source_sha256"]
        ),
        "current_passing_source_hashes_match": not source_mismatches,
        "failure_artifacts_preserved": (
            all(item.get("qualification_sample") is False for item in failures)
            and not failure_reference_mismatches
        ),
        "all_manifest_files_exist": not missing,
        "nonempty_manifest": len(paths) >= 40,
    }
    value = {
        "schema": "softwall-c164-idle30-manifest-v1",
        "status": "C164_IDLE30_MANIFEST_PASS" if all(checks.values()) else "C164_IDLE30_MANIFEST_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "file_count": len(paths),
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in paths
            if path.is_file()
        },
        "diagnostics": {
            "missing": missing, "source_mismatches": source_mismatches,
            "failure_reference_mismatches": failure_reference_mismatches,
        },
    }
    output = results / "c164_idle30_manifest.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": value["status"], "checks": checks,
                      "file_count": len(paths)}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
