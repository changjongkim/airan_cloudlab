#!/usr/bin/env python3
"""Compile conservative DART profiles from measured NRx/P2P/GDR artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def find_direct_graph(root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("direct_graph.json")
        if "nsys_nrx_direct_wrapper" in str(path)
    ]
    if not candidates:
        candidates = [
            path
            for path in root.rglob("*.json")
            if (data := load_json(path)).get("cuda_graph") is True
            and "direct" in data
        ]
    if not candidates:
        raise FileNotFoundError("no direct CUDA-Graph NRx artifact")
    return sorted(candidates)[-1]


def p2p_samples(root: Path):
    records = []
    sources = []
    for path in root.rglob("p2p_overlap_*.json"):
        data = load_json(path)
        if data.get("mode") != "p2p":
            continue
        metrics = data.get("metrics", {})
        try:
            records.append({
                "forward_ns": round(metrics["fwd_copy_us"]["p99"] * 1000),
                "backward_ns": round(metrics["bwd_copy_us"]["p99"] * 1000),
                "transport_ns": round(metrics["transport_us"]["p99"] * 1000),
            })
            sources.append(path)
        except (KeyError, TypeError):
            continue
    if not records:
        raise FileNotFoundError("no P2P current-payload artifacts")
    return records, sources


def transport_summary(root: Path):
    """Read optional validated transport summary without inventing GDR values."""
    for path in root.rglob("TRANSPORT_SUMMARY.csv"):
        import csv

        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return path, rows
    return None, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-class", type=int, default=1)
    parser.add_argument("--graph-id", type=int, default=1)
    parser.add_argument("--endpoint-count", type=int, default=3)
    parser.add_argument("--error-guard-pct", type=float, default=10.0)
    args = parser.parse_args()
    if args.endpoint_count <= 0 or args.error_guard_pct < 0:
        parser.error("invalid endpoint count or error guard")

    root = Path(args.results_root).resolve()
    output = Path(args.output).resolve()
    direct_path = find_direct_graph(root)
    direct = load_json(direct_path)
    service_ns = round(direct["direct"]["gpu_ms"]["p99"] * 1_000_000)
    control_ns = round(direct["direct"]["enqueue_us"]["p99"] * 1000)

    p2p, p2p_paths = p2p_samples(root)
    forward_ns = round(statistics.median(item["forward_ns"] for item in p2p))
    backward_ns = round(statistics.median(item["backward_ns"] for item in p2p))
    base_ns = forward_ns + service_ns + backward_ns + control_ns
    error_ns = round(base_ns * args.error_guard_pct / 100.0)

    profiles = []
    for index in range(args.endpoint_count):
        profiles.append({
            "endpoint_id": f"p2p{index}",
            "tensor_class": args.tensor_class,
            "graph_id": args.graph_id,
            "background_mode": "isolated",
            "forward_ns": forward_ns,
            "service_ns": service_ns,
            "backward_ns": backward_ns,
            "control_ns": control_ns,
            "positive_error_ns": error_ns,
            "bound_ns": base_ns + error_ns,
            "transport": "p2p",
        })

    summary_path, summary_rows = transport_summary(root)
    gdr_note = (
        "available but not auto-mapped: direction-specific current-payload rows required"
        if summary_rows
        else "not available; GDR profile intentionally omitted"
    )
    sources = [{
        "path": str(direct_path),
        "sha256": sha256(direct_path),
        "role": "nrx_service_and_control",
    }]
    for path in p2p_paths:
        sources.append({
            "path": str(path),
            "sha256": sha256(path),
            "role": "p2p_current_payload",
        })
    if summary_path:
        sources.append({
            "path": str(summary_path),
            "sha256": sha256(summary_path),
            "role": "optional_transport_summary",
        })

    result = {
        "schema": "dart-profile-v0",
        "scope": "measured profile compiler; GDR omitted unless direction-safe",
        "quantile": "p99",
        "error_guard_pct": args.error_guard_pct,
        "current_payload": {"forward_bytes": 1_415_232, "backward_bytes": 314_496},
        "aggregate": {
            "nrx_service_ns": service_ns,
            "control_ns": control_ns,
            "p2p_forward_ns": forward_ns,
            "p2p_backward_ns": backward_ns,
            "p2p_samples": len(p2p),
            "gdr": gdr_note,
        },
        "profiles": profiles,
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"[DART-PROFILE] OK service={service_ns/1e6:.3f}ms "
        f"p2p={forward_ns/1e3:.1f}+{backward_ns/1e3:.1f}us "
        f"bound={(base_ns+error_ns)/1e6:.3f}ms output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
