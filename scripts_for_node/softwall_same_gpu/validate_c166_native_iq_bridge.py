#!/usr/bin/env python3
"""Validate the native CUDA IQ bridge against the frozen C166 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

import _softwall_native_iq


SHAPE = (3276, 14, 4)
ELEMENTS = int(np.prod(SHAPE))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    planes_path = args.fixture / manifest["buffers"]["p2p_real_imag"]["file"]
    complex_path = args.fixture / manifest["buffers"]["complex_fortran"]["file"]
    planes_cpu = np.fromfile(planes_path, dtype=np.float32)
    expected_cpu = np.fromfile(complex_path, dtype=np.complex64).reshape(
        SHAPE, order="F"
    )
    planes = cp.asarray(planes_cpu)
    actual = cp.empty(SHAPE, dtype=cp.complex64, order="F")
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        _softwall_native_iq.assemble_iq(planes, actual, int(stream.ptr))
    stream.synchronize()
    expected = cp.asarray(expected_cpu, order="F")
    actual_words = actual.ravel(order="F").view(cp.uint32)
    expected_words = expected.ravel(order="F").view(cp.uint32)
    bitwise_equal = bool(cp.array_equal(actual_words, expected_words))

    rejected_c_order = False
    try:
        bad = cp.empty(SHAPE, dtype=cp.complex64, order="C")
        _softwall_native_iq.assemble_iq(planes, bad, int(stream.ptr))
    except ValueError:
        rejected_c_order = True

    checks = {
        "fixture_hashes_match_manifest": (
            digest(planes_path) == manifest["buffers"]["p2p_real_imag"]["sha256"]
            and digest(complex_path) == manifest["buffers"]["complex_fortran"]["sha256"]
        ),
        "element_count": planes_cpu.size == 2 * ELEMENTS,
        "native_output_bitwise_equal": bitwise_equal,
        "c_order_destination_rejected": rejected_c_order,
    }
    result = {
        "schema": "softwall-c166-native-iq-bridge-validation-v1",
        "analysis_role": "Exact bridge validation only; no timing qualification.",
        "complex_elements": ELEMENTS,
        "checks": checks,
        "all_pass": all(checks.values()),
        "sha256": {
            "manifest": digest(manifest_path),
            "planes": digest(planes_path),
            "complex_fortran": digest(complex_path),
            "native_module": digest(args.module),
        },
        "qualification": "forbidden",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
