#!/usr/bin/env python3.11
"""Build a transitive manifest for the C164 two-node MPS-restart campaign."""

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
        "c164c_mpsrestart_dev_j58861897",
        "c164d_mpsrestart_holdout_j58861911",
    )
    protocols = [load(results / f"{label}_protocol.json") for label in labels]
    combined = load(results / "c164_mps_restart_two_node.json")
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
        for key, expected in protocol["c164_restart"]["source_sha256"].items():
            register_source(key, expected)

    extras = (
        "scripts_for_node/softwall_same_gpu/analyze_c164_restart_two_node.py",
        "scripts_for_node/softwall_same_gpu/test_analyze_c164_restart_two_node.py",
        "scripts_for_node/softwall_same_gpu/build_c164_restart_manifest.py",
        "docs/current/SOFTWALL_C164_IDLE30_RESULT_KO.md",
        "results/softwall_multigpu/c164_mps_restart_two_node.json",
    )
    files.update(root / name for name in extras)
    for label in labels:
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    paths = sorted(files)
    checks = {
        "combined_two_node_pass": combined.get("all_pass") is True,
        "passing_source_equality": (
            protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
            and protocols[0]["c164_restart"]["source_sha256"]
                == protocols[1]["c164_restart"]["source_sha256"]
        ),
        "current_passing_source_hashes_match": not source_mismatches,
        "both_restart_markers_included": all(
            raw / f"{label}_mps_restart_marker.json" in files for label in labels
        ),
        "all_manifest_files_exist": not missing,
        "nonempty_manifest": len(paths) >= 40,
    }
    value = {
        "schema": "softwall-c164-mps-restart-manifest-v1",
        "status": "C164_MPS_RESTART_MANIFEST_PASS" if all(checks.values()) else "C164_MPS_RESTART_MANIFEST_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "file_count": len(paths),
        "files_sha256": {
            str(path.relative_to(root)): sha256(path) for path in paths
            if path.is_file()
        },
        "diagnostics": {"missing": missing,
                        "source_mismatches": source_mismatches},
    }
    output = results / "c164_mps_restart_manifest.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": value["status"], "checks": checks,
                      "file_count": len(paths)}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
