#!/usr/bin/env python3.11
"""Augment a frozen C162 boundary protocol with C164 lifecycle gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


C164_SOURCES = (
    "c164_lifecycle_qualification_v1.py",
    "test_c164_lifecycle_qualification_v1.py",
    "run_c164_lifecycle_model_v1.py",
    "c164_idle_coordinator.py",
    "build_c164_idle_protocol.py",
    "analyze_c164_idle.py",
    "run_c164_idle30.sh",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c164_source_hashes(scripts_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in C164_SOURCES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing C164 source: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--required-idle-s", type=float, required=True)
    parser.add_argument("--release-lead-ms", type=float, required=True)
    args = parser.parse_args()
    if args.required_idle_s <= 0:
        parser.error("required idle must be positive")
    if args.release_lead_ms < 2000:
        parser.error("release lead must leave two seconds after schedule publication")
    value = json.loads(args.protocol.read_text())
    if value.get("schema") != "softwall-c162-boundary-protocol-v1":
        parser.error("base C162 protocol schema mismatch")
    if value.get("status") != "frozen-before-run":
        parser.error("base protocol is not frozen")
    if float(value["mode"]["release_lead_ms"]) != args.release_lead_ms:
        parser.error("release lead differs from frozen base protocol")
    value["c164"] = {
        "schema": "softwall-c164-idle-boundary-protocol-v1",
        "lifecycle": "idle_30s_first",
        "required_idle_s": args.required_idle_s,
        "release_lead_ms": args.release_lead_ms,
        "scope": "first physical request after all ready persistent clients idle before schedule publication",
        "qualification_level": "boundary-subset; full six-class lifecycle vector remains UQ",
        "gate": (
            "same-host coordinator records at least the required idle after all "
            "readiness and before schedule publication, then all frozen C162 boundary "
            "and physical gates pass"
        ),
        "source_sha256": c164_source_hashes(args.scripts_root),
    }
    temporary = args.protocol.with_suffix(args.protocol.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.protocol)
    print(json.dumps(value["c164"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
