#!/usr/bin/env python3.11
"""Build the transitive manifest for C151/C152 shared-conventional evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "scripts_for_node/softwall_same_gpu/shared_conventional_owner.py",
    "scripts_for_node/softwall_same_gpu/shared_conventional_worker.py",
    "scripts_for_node/softwall_same_gpu/build_confirm149_protocol.py",
    "scripts_for_node/softwall_same_gpu/analyze_confirm149_shared_conventional.py",
    "scripts_for_node/softwall_same_gpu/analyze_confirm151_152_shared_conventional.py",
    "scripts_for_node/softwall_same_gpu/run_v17_shared_conventional_canary.sh",
    "scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py",
    "scripts_for_node/softwall_same_gpu/dual_receiver_phy.py",
    "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
)
LABELS = (
    "confirm151_shared_conv_qualified_job58851949",
    "confirm152_shared_conv_holdout_job58852015",
)
COMBINED = (
    "results/softwall_multigpu/confirm151_152_shared_conventional_qualification.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT.resolve()))


def collect_files():
    files = list(CODE) + [COMBINED]
    result_root = ROOT / "results/softwall_multigpu"
    for label in LABELS:
        result_path = result_root / f"{label}_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        files.extend((
            relative(result_path),
            relative(result_root / f"{label}_protocol.json"),
            relative(result_root / f"{label}.log"),
            relative(Path(result["worker"])),
            relative(Path(result["inventory"])),
        ))
        files.extend(relative(Path(path)) for path in result["owners"])
    if len(files) != len(set(files)):
        raise RuntimeError("duplicate manifest path")
    return tuple(files)


def build_manifest():
    combined = json.loads((ROOT / COMBINED).read_text(encoding="utf-8"))
    files = collect_files()
    return {
        "schema": "softwall-confirm151-152-shared-conventional-manifest-v1",
        "status": combined["status"],
        "claim_scope": combined["claim_scope"],
        "combined_all_pass": combined["all_pass"],
        "file_count": len(files),
        "files": [
            {
                "path": value,
                "bytes": (ROOT / value).stat().st_size,
                "sha256": sha256(ROOT / value),
            }
            for value in files
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "file_count": manifest["file_count"],
        "combined_all_pass": manifest["combined_all_pass"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
