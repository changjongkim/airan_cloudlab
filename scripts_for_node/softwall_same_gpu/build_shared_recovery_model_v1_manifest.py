#!/usr/bin/env python3.11
"""Build a deterministic manifest for the V17 shared-recovery model candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "scripts_for_node/softwall_same_gpu/shared_recovery_certificate_v1.py",
    "scripts_for_node/softwall_same_gpu/test_shared_recovery_certificate_v1.py",
    "scripts_for_node/softwall_same_gpu/verify_shared_recovery_certificate_v1.py",
    "scripts_for_node/softwall_same_gpu/test_verify_shared_recovery_certificate_v1.py",
    "results/softwall_multigpu/softwall_shared_recovery_model_v1.json",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest():
    result_path = ROOT / FILES[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = []
    for relative in FILES:
        path = ROOT / relative
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    return {
        "schema": "softwall-shared-recovery-model-manifest-v1",
        "status": "MODEL_PASS_PHYSICAL_UQ",
        "claim_scope": result["claim_scope"],
        "result_all_pass": result["all_pass"],
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "files": len(manifest["files"]),
        "result_all_pass": manifest["result_all_pass"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
