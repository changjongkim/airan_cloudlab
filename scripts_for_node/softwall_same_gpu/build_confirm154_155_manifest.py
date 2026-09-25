#!/usr/bin/env python3.11
"""Build the transitive manifest for the C154/C155 two-node holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm154_holdout_protocol import SCRIPT_SOURCES, TASK1_SOURCES


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = (
    "results/softwall_multigpu/confirm154_integrated_holdout_job58853453_result.json",
    "results/softwall_multigpu/confirm155_integrated_holdout_job_result.json",
)
PROTOCOLS = (
    "results/softwall_multigpu/confirm154_integrated_holdout_job58853453_protocol.json",
    "results/softwall_multigpu/confirm155_integrated_holdout_job_protocol.json",
)
COMBINED = "results/softwall_multigpu/confirm154_155_controlled_integrated_holdout.json"
EXTRA_CODE = (
    "scripts_for_node/softwall_same_gpu/test_integrated_shared_recovery_plan_v1.py",
    "scripts_for_node/softwall_same_gpu/test_integrated_shared_recovery_holdout_plan_v1.py",
    "scripts_for_node/softwall_same_gpu/analyze_confirm154_155_combined.py",
    "scripts_for_node/softwall_same_gpu/build_confirm154_155_manifest.py",
    "scripts_for_node/softwall_same_gpu/test_build_confirm154_155_manifest.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def code_paths() -> tuple[str, ...]:
    values = [
        f"scripts_for_node/softwall_same_gpu/{name}" for name in SCRIPT_SOURCES
    ]
    values.extend(f"scripts_for_node/task1/{name}" for name in TASK1_SOURCES)
    values.extend(EXTRA_CODE)
    return tuple(values)


def collect_files() -> tuple[str, ...]:
    files = set(code_paths())
    files.add(COMBINED)
    for campaign_name, protocol_name in zip(CAMPAIGNS, PROTOCOLS):
        files.update((campaign_name, protocol_name))
        campaign = json.loads((ROOT / campaign_name).read_text(encoding="utf-8"))
        for arm_name in campaign["arm_results"]:
            arm_path = Path(arm_name)
            files.add(relative(arm_path))
            arm = json.loads(arm_path.read_text(encoding="utf-8"))
            files.add(relative(Path(arm["peer_spec"])))
            files.update(relative(Path(value)) for value in arm["artifact_sha256"])
    missing = [value for value in sorted(files) if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return tuple(sorted(files))


def build_manifest() -> dict:
    protocols = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in PROTOCOLS]
    current = {}
    for name in SCRIPT_SOURCES:
        current[f"scripts/{name}"] = sha256(
            ROOT / "scripts_for_node/softwall_same_gpu" / name
        )
    for name in TASK1_SOURCES:
        current[f"task1/{name}"] = sha256(ROOT / "scripts_for_node/task1" / name)
    if not all(value["source_sha256"] == current for value in protocols):
        raise RuntimeError("holdout source hash no longer matches current sealed source")
    combined = json.loads((ROOT / COMBINED).read_text(encoding="utf-8"))
    if not combined["all_pass"]:
        raise RuntimeError("combined holdout did not pass")
    files = collect_files()
    return {
        "schema": "softwall-confirm154-155-integrated-holdout-manifest-v1",
        "status": combined["status"],
        "combined_all_pass": combined["all_pass"],
        "nodes": combined["nodes"],
        "job_ids": combined["job_ids"],
        "file_count": len(files),
        "files": [
            {
                "path": value,
                "bytes": (ROOT / value).stat().st_size,
                "sha256": sha256(ROOT / value),
            }
            for value in files
        ],
        "claim_scope": combined["claim_scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": value["status"],
        "file_count": value["file_count"],
        "nodes": value["nodes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

