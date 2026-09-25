#!/usr/bin/env python3.11
"""Augment a frozen C162 protocol with quiescent MPS-restart gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


C164_RESTART_SOURCES = (
    "c164_mps_epoch_probe.py",
    "c164_mps_epoch_snapshot.py",
    "build_c164_mps_restart_marker.py",
    "build_c164_restart_protocol.py",
    "analyze_c164_restart.py",
    "test_c164_mps_restart.py",
    "run_c164_mps_restart.sh",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restart_source_hashes(scripts_root: Path) -> dict[str, str]:
    paths = {
        f"scripts/{name}": scripts_root / name
        for name in C164_RESTART_SOURCES
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing C164 restart source: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.protocol.read_text())
    if value.get("schema") != "softwall-c162-boundary-protocol-v1":
        parser.error("base C162 protocol schema mismatch")
    if value.get("status") != "frozen-before-run":
        parser.error("base protocol is not frozen")
    value["c164_restart"] = {
        "schema": "softwall-c164-mps-restart-protocol-v1",
        "lifecycle": "mps_restart_first",
        "restart_scope": "quiescent daemon restart followed by full fresh-client requalification before the first RAN request",
        "availability_scope": "optional NRx and AI remain closed during restart; uninterrupted service across restart is not claimed",
        "old_token_rule": "the previous lifecycle epoch is stale; no optional admission until new readiness and full-path preflight complete",
        "physical_restart_proof": (
            "four-GPU CUDA probe in the old epoch, complete control/server absence, "
            "distinct control socket/control PID/server PID, then a four-GPU CUDA "
            "probe in the new epoch"
        ),
        "qualification_level": "C162 six-case boundary subset after restart; full six-class lifecycle vector remains UQ",
        "source_sha256": restart_source_hashes(args.scripts_root),
    }
    temporary = args.protocol.with_suffix(args.protocol.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.protocol)
    print(json.dumps(value["c164_restart"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
