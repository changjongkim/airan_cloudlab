#!/usr/bin/env python3.11
"""Build a transitive manifest for the C164 reconnect campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts_for_node/softwall_same_gpu"
    task1 = root / "scripts_for_node/task1"
    results = root / "results/softwall_multigpu"
    raw = results / "raw"
    state = root / "run_state/softwall_multigpu"
    passing = ("c164g3_reconnect_dev_j58862429",
               "c164h_reconnect_holdout_j58862501")
    failures = ("c164g_reconnect_dev_j58862429",
                "c164g2_reconnect_dev_j58862429")
    protocols = [load(results / f"{label}_protocol.json") for label in passing]
    combined = load(results / "c164_reconnect_two_node.json")
    failure_results = [load(results / f"{label}_result.json") for label in failures]
    files: set[Path] = set()
    source_mismatches = []

    for protocol in protocols:
        for key, expected in protocol["source_sha256"].items():
            if key.startswith("scripts/"):
                path = scripts / key.removeprefix("scripts/")
            elif key.startswith("task1/"):
                path = task1 / key.removeprefix("task1/")
            else:
                raise ValueError(key)
            files.add(path)
            if not path.is_file() or sha256(path) != expected:
                source_mismatches.append(key)
    extras = (
        "scripts_for_node/softwall_same_gpu/analyze_c164_reconnect_two_node.py",
        "scripts_for_node/softwall_same_gpu/test_analyze_c164_reconnect_two_node.py",
        "scripts_for_node/softwall_same_gpu/build_c164_reconnect_manifest.py",
        "docs/current/SOFTWALL_C164_IDLE30_RESULT_KO.md",
        "results/softwall_multigpu/c164_reconnect_model_v1.json",
        "results/softwall_multigpu/c164_reconnect_two_node.json",
    )
    files.update(root / name for name in extras)
    for label in passing + failures:
        files.update(results.glob(f"{label}*"))
        files.update(raw.glob(f"{label}*"))
        journal = state / label / "journal"
        if journal.is_dir():
            files.update(journal.glob("*.json"))
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    paths = sorted(files)
    checks = {
        "combined_two_node_pass": combined.get("all_pass") is True,
        "passing_source_equality": protocols[0]["source_sha256"]
            == protocols[1]["source_sha256"],
        "current_passing_source_hashes_match": not source_mismatches,
        "two_development_failures_preserved": (
            len(failure_results) == 2
            and all(item.get("all_pass") is False for item in failure_results)
            and all(item.get("counts", {}).get("overlap_releases") == 0
                    for item in failure_results)
        ),
        "all_manifest_files_exist": not missing,
        "durable_journal_files_included": sum(
            1 for path in paths if path.parent.name == "journal"
        ) == 240,
        "nonempty_manifest": len(paths) >= 280,
    }
    value = {
        "schema": "softwall-c164-reconnect-manifest-v1",
        "status": "C164_RECONNECT_MANIFEST_PASS" if all(checks.values()) else "C164_RECONNECT_MANIFEST_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "file_count": len(paths),
        "files_sha256": {str(path.relative_to(root)): sha256(path)
                         for path in paths if path.is_file()},
        "diagnostics": {"missing": missing,
                        "source_mismatches": source_mismatches},
    }
    output = results / "c164_reconnect_manifest.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks,
                      "file_count": len(paths)}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
