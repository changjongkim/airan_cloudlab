#!/usr/bin/env python3
"""Fail closed on a malformed C166 Python-to-native fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    manifest_path = args.fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema") == "softwall-c166-native-fixture-v1",
        "not_qualification": "no latency" in manifest.get("role", "").lower(),
        "both_reference_decoders_correct": (
            manifest["fixture_decode"]["conventional_correct"] is True
            and manifest["fixture_decode"]["neural_correct"] is True
        ),
    }
    for value in manifest["buffers"].values():
        path = args.fixture / value["file"]
        checks[f"exists_{value['file']}"] = path.is_file()
        checks[f"bytes_{value['file']}"] = path.stat().st_size == value["bytes"]
        checks[f"sha256_{value['file']}"] = sha256(path) == value["sha256"]

    shape = tuple(manifest["buffers"]["complex_fortran"]["shape"])
    complex_path = args.fixture / manifest["buffers"]["complex_fortran"]["file"]
    p2p_path = args.fixture / manifest["buffers"]["p2p_real_imag"]["file"]
    direct = np.fromfile(complex_path, dtype=np.complex64).reshape(shape, order="F")
    planes = np.fromfile(p2p_path, dtype=np.float32)
    elements = int(np.prod(shape))
    rebuilt = (
        planes[:elements].reshape(shape, order="C")
        + 1j * planes[elements:].reshape(shape, order="C")
    ).astype(np.complex64)
    checks["p2p_and_fortran_logical_values_match"] = np.array_equal(direct, rebuilt)
    all_pass = all(checks.values())
    result = {
        "schema": "softwall-c166-native-fixture-validation-v1",
        "fixture": str(args.fixture),
        "checks": checks,
        "all_pass": all_pass,
    }
    print(json.dumps(result, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
