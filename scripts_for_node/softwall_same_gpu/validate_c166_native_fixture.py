#!/usr/bin/env python3
"""Fail closed on a malformed C166 Python-to-native fixture."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import sys


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
    if sys.byteorder != "little":
        raise RuntimeError("C166 fixture validator currently requires little endian")
    direct = array.array("f")
    direct.frombytes(complex_path.read_bytes())
    planes = array.array("f")
    planes.frombytes(p2p_path.read_bytes())
    n0, n1, n2 = shape
    elements = n0 * n1 * n2
    values_match = len(direct) == 2 * elements and len(planes) == 2 * elements
    if values_match:
        for first in range(n0):
            for second in range(n1):
                for third in range(n2):
                    fortran_index = first + n0 * (second + n1 * third)
                    c_index = (first * n1 + second) * n2 + third
                    if (
                        direct[2 * fortran_index] != planes[c_index]
                        or direct[2 * fortran_index + 1] != planes[elements + c_index]
                    ):
                        values_match = False
                        break
                if not values_match:
                    break
            if not values_match:
                break
    checks["p2p_and_fortran_logical_values_match"] = values_match
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
