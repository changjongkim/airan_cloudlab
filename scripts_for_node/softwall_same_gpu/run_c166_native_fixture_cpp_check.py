#!/usr/bin/env python3
"""Build and invoke the dependency-light C++ C166 fixture checker."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().with_name("native_fast_path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.fixture / "manifest.json").read_text(encoding="utf-8"))
    build = ROOT / "run_state" / "c166_native_fixture_cpp_build"
    subprocess.run(
        ["cmake", "-S", str(SOURCE), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
    command = [
        str(build / "softwall_native_fixture_check"),
        "--fixture", str(args.fixture),
        "--complex-sha", manifest["buffers"]["complex_fortran"]["sha256"],
        "--p2p-sha", manifest["buffers"]["p2p_real_imag"]["sha256"],
        "--tb-sha", manifest["buffers"]["reference_tb"]["sha256"],
        "--tb-bytes", str(manifest["buffers"]["reference_tb"]["bytes"]),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    result["fixture"] = str(args.fixture)
    result["qualification"] = "forbidden"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
