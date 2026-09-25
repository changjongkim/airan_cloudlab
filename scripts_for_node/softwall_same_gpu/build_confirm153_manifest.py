#!/usr/bin/env python3.11
"""Build the immutable C153 development-campaign manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "scripts_for_node/softwall_same_gpu/integrated_shared_recovery_plan_v1.py",
    "scripts_for_node/softwall_same_gpu/test_integrated_shared_recovery_plan_v1.py",
    "scripts_for_node/softwall_same_gpu/integrated_shared_recovery_owner.py",
    "scripts_for_node/softwall_same_gpu/integrated_shared_recovery_worker.py",
    "scripts_for_node/softwall_same_gpu/shared_recovery_certificate_v1.py",
    "scripts_for_node/softwall_same_gpu/shared_conventional_owner.py",
    "scripts_for_node/softwall_same_gpu/shared_conventional_worker.py",
    "scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py",
    "scripts_for_node/softwall_same_gpu/dual_receiver_phy.py",
    "scripts_for_node/softwall_same_gpu/trace_qwen_worker.py",
    "scripts_for_node/softwall_same_gpu/build_confirm153_protocol.py",
    "scripts_for_node/softwall_same_gpu/analyze_confirm153_integrated_shared_recovery.py",
    "scripts_for_node/softwall_same_gpu/build_confirm153_manifest.py",
    "scripts_for_node/softwall_same_gpu/test_build_confirm153_manifest.py",
    "scripts_for_node/softwall_same_gpu/run_confirm153_integrated_shared_recovery.sh",
    "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
)
ATTEMPT1 = (
    "results/softwall_multigpu/confirm153_attempt1_failure.json",
    "results/softwall_multigpu/confirm153_integrated_shared_recovery_job58852725_protocol.json",
    "results/softwall_multigpu/raw/confirm153_integrated_shared_recovery_job58852725_gpu_inventory.csv",
)
COMPLETE_LABELS = (
    "confirm153_integrated_shared_recovery_job58852876",
    "confirm153_integrated_shared_recovery_job58852924",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files() -> tuple[str, ...]:
    files = list(CODE) + list(ATTEMPT1)
    root = ROOT / "results/softwall_multigpu"
    for label in COMPLETE_LABELS:
        files.extend((
            f"results/softwall_multigpu/{label}_protocol.json",
            f"results/softwall_multigpu/{label}_result.json",
            f"results/softwall_multigpu/raw/{label}_gpu_inventory.csv",
            f"results/softwall_multigpu/raw/{label}_owner0.json",
            f"results/softwall_multigpu/raw/{label}_owner1.json",
            f"results/softwall_multigpu/raw/{label}_worker.json",
            f"results/softwall_multigpu/raw/{label}_qwen.json",
        ))
    if len(files) != len(set(files)):
        raise RuntimeError("duplicate C153 manifest path")
    missing = [value for value in files if not (ROOT / value).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return tuple(files)


def build_manifest() -> dict:
    failed = json.loads((
        ROOT / "results/softwall_multigpu/"
        "confirm153_integrated_shared_recovery_job58852876_result.json"
    ).read_text(encoding="utf-8"))
    passed = json.loads((
        ROOT / "results/softwall_multigpu/"
        "confirm153_integrated_shared_recovery_job58852924_result.json"
    ).read_text(encoding="utf-8"))
    if failed["all_pass"] or not passed["all_pass"]:
        raise RuntimeError("unexpected C153 attempt outcomes")
    files = collect_files()
    return {
        "schema": "softwall-confirm153-development-manifest-v1",
        "status": passed["status"],
        "development_attempts": 3,
        "failed_attempts_preserved": 2,
        "passing_job": "58852924",
        "passing_node": "nid001348",
        "file_count": len(files),
        "files": [
            {
                "path": value,
                "bytes": (ROOT / value).stat().st_size,
                "sha256": sha256(ROOT / value),
            }
            for value in files
        ],
        "claim_scope": passed["scope"],
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
