#!/usr/bin/env python3.11
"""Build the transitive manifest for C156 attempts and passing trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm156_timeline_protocol import (
    SCRIPT_SOURCES,
    TASK1_SOURCES,
    source_hashes,
)


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT1 = "confirm156_timeline_job58853926"
ATTEMPT2 = "confirm156_timeline_attempt2_job58853926"
PASSING = "confirm156_timeline_attempt3_job58853926"
HOLDOUT = "confirm156b_timeline_holdout_job58854401"
COMBINED = "results/softwall_multigpu/confirm156_156b_two_node_gpu_timeline.json"
EXTRA_CODE = (
    "scripts_for_node/softwall_same_gpu/test_integrated_shared_recovery_launch_plan_v1.py",
    "scripts_for_node/softwall_same_gpu/test_analyze_confirm156_timeline.py",
    "scripts_for_node/softwall_same_gpu/test_build_confirm156_timeline_protocol.py",
    "scripts_for_node/softwall_same_gpu/analyze_confirm156_two_node.py",
    "scripts_for_node/softwall_same_gpu/test_analyze_confirm156_two_node.py",
    "scripts_for_node/softwall_same_gpu/build_confirm156_manifest.py",
    "scripts_for_node/softwall_same_gpu/test_build_confirm156_manifest.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_paths() -> set[str]:
    values = {
        f"scripts_for_node/softwall_same_gpu/{name}" for name in SCRIPT_SOURCES
    }
    values.update(f"scripts_for_node/task1/{name}" for name in TASK1_SOURCES)
    values.update(EXTRA_CODE)
    return values


def label_paths(label: str, include_result: bool) -> set[str]:
    base = "results/softwall_multigpu"
    raw = f"{base}/raw"
    paths = {
        f"{base}/{label}_protocol.json",
        f"{base}/{label}_peer_spec.json",
        f"{raw}/{label}_gpu_inventory.csv",
        f"{raw}/{label}_worker.json",
        f"{raw}/{label}_qwen.json",
        f"{raw}/{label}_worker_profile.nsys-rep",
        f"{raw}/{label}_worker_profile.sqlite",
        f"{raw}/{label}_qwen_profile.nsys-rep",
        f"{raw}/{label}_qwen_profile.sqlite",
    }
    peer_spec = json.loads(
        (ROOT / f"{base}/{label}_peer_spec.json").read_text(encoding="utf-8")
    )
    paths.update(
        str(Path(row["owner_output"]).resolve().relative_to(ROOT.resolve()))
        for row in peer_spec["peers"]
    )
    if include_result:
        paths.add(f"{base}/{label}_result.json")
    return paths


def collect_files() -> tuple[str, ...]:
    files = code_paths()
    files.update(label_paths(ATTEMPT1, include_result=False))
    files.update(label_paths(ATTEMPT2, include_result=True))
    files.update(label_paths(PASSING, include_result=True))
    files.update(label_paths(HOLDOUT, include_result=True))
    files.add(COMBINED)
    files.add("results/softwall_multigpu/confirm156_attempt1_failure.json")
    missing = [value for value in sorted(files) if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return tuple(sorted(files))


def build_manifest() -> dict:
    attempt2 = json.loads((
        ROOT / f"results/softwall_multigpu/{ATTEMPT2}_result.json"
    ).read_text(encoding="utf-8"))
    passed = json.loads((
        ROOT / f"results/softwall_multigpu/{PASSING}_result.json"
    ).read_text(encoding="utf-8"))
    holdout = json.loads((
        ROOT / f"results/softwall_multigpu/{HOLDOUT}_result.json"
    ).read_text(encoding="utf-8"))
    protocols = [json.loads((
        ROOT / f"results/softwall_multigpu/{PASSING}_protocol.json"
    ).read_text(encoding="utf-8")), json.loads((
        ROOT / f"results/softwall_multigpu/{HOLDOUT}_protocol.json"
    ).read_text(encoding="utf-8"))]
    combined = json.loads((ROOT / COMBINED).read_text(encoding="utf-8"))
    current = source_hashes(
        ROOT / "scripts_for_node/softwall_same_gpu",
        ROOT / "scripts_for_node/task1",
    )
    if not all(protocol["source_sha256"] == current for protocol in protocols):
        raise RuntimeError("passing C156 protocols do not match current source")
    if (attempt2["all_pass"] or not passed["all_pass"]
            or not holdout["all_pass"] or not combined["all_pass"]):
        raise RuntimeError("unexpected C156 attempt outcomes")
    files = collect_files()
    return {
        "schema": "softwall-confirm156-gpu-timeline-manifest-v1",
        "status": combined["status"],
        "physical_runs": 4,
        "failed_attempts_preserved": 2,
        "passing_job": passed["job_id"],
        "passing_node": passed["node"],
        "passing_label": PASSING,
        "passing_jobs": combined["jobs"],
        "passing_nodes": combined["nodes"],
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
        "passing_job": value["passing_job"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
