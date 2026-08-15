#!/usr/bin/env python3
"""Compile DART endpoint profiles strictly from regular full-day artifacts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--guard-pct", type=float, required=True)
    parser.add_argument("--endpoints", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.guard_pct < 0 or args.endpoints <= 0:
        parser.error("guard non-negative and endpoints positive")
    root = Path(args.root).resolve()
    nrx_paths = sorted((root / "02_nrx_stack").glob("trial_*.json"))
    p2p_paths = sorted((root / "05_fiveway_compute/p2p").glob("load_0.50/trial_*/*.json"))
    if len(nrx_paths) != 5 or len(p2p_paths) != 5:
        raise RuntimeError(
            f"regular sources incomplete nrx={len(nrx_paths)} p2p={len(p2p_paths)}"
        )
    services = []
    controls = []
    for path in nrx_paths:
        value = json.loads(path.read_text())
        variant = value["variants"]["N04_cuda_graph"]["metrics"]
        services.append(float(variant["gpu_ms"]["p99"]) * 1e6)
        controls.append(float(variant["enqueue_us"]["p99"]) * 1e3)
    forwards = []
    backwards = []
    for path in p2p_paths:
        value = json.loads(path.read_text())
        forwards.extend(float(item["fwd_us"]) * 1e3 for item in value["raw"])
        backwards.extend(float(item["bwd_us"]) * 1e3 for item in value["raw"])
    service_ns = round(statistics.median(services))
    control_ns = round(statistics.median(controls))
    forward_ns = round(float(np.percentile(forwards, 99)))
    backward_ns = round(float(np.percentile(backwards, 99)))
    unguarded = service_ns + control_ns + forward_ns + backward_ns
    error_ns = round(unguarded * args.guard_pct / 100.0)
    profiles = []
    for index in range(args.endpoints):
        profiles.append({
            "endpoint_id": f"p2p{index}", "tensor_class": 1, "graph_id": 1,
            "background_mode": "isolated", "forward_ns": forward_ns,
            "service_ns": service_ns, "backward_ns": backward_ns,
            "control_ns": control_ns, "positive_error_ns": error_ns,
            "bound_ns": unguarded + error_ns, "transport": "cuda_p2p",
        })
    result = {
        "schema": "dart-profile-full-v1",
        "scope": "regular 5-trial NRx graph and low-load same-boundary P2P",
        "quantile": "median of per-trial p99 for NRx; pooled p99 for P2P",
        "guard_pct": args.guard_pct,
        "current_payload": {"forward_bytes": 1_415_232, "backward_bytes": 314_496},
        "aggregate": {
            "service_ns": service_ns, "control_ns": control_ns,
            "forward_ns": forward_ns, "backward_ns": backward_ns,
            "unguarded_ns": unguarded, "bound_ns": unguarded + error_ns,
        },
        "profiles": profiles,
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in nrx_paths + p2p_paths
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"[DART-FULL-PROFILE] PASS guard={args.guard_pct}% "
        f"bound={(unguarded+error_ns)/1e6:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
