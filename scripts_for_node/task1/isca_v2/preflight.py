#!/usr/bin/env python3
"""Read-only CloudLab preflight with strict topology and provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


def command(argv, required=True):
    result = subprocess.run(argv, text=True, capture_output=True)
    item = {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if required and result.returncode:
        raise RuntimeError(f"command failed: {argv}: {result.stderr.strip()}")
    return item


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-root", default="/mydata/aerial-cuda-accelerated-ran/pyaerial"
    )
    parser.add_argument("--allow-running-containers", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    checks = {}
    checks["nvidia_smi_l"] = command(["nvidia-smi", "-L"])
    topology = checks["nvidia_smi_l"]["stdout"]
    checks["gpu_query"] = command([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,persistence_mode,mig.mode.current",
        "--format=csv,noheader,nounits",
    ])
    checks["compute_processes"] = command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], required=False)
    checks["topo"] = command(["nvidia-smi", "topo", "-m"])
    checks["docker_ps"] = command([
        "docker", "ps", "--format", "{{.Names}} {{.Status}}"
    ])
    checks["docker_images"] = command([
        "docker", "image", "inspect", "airan:25-3-final", "airan:25-3-rdma",
        "--format", "{{.RepoTags}} {{.Id}}",
    ])
    checks["ib_state"] = command([
        "bash", "-c",
        "cat /sys/class/infiniband/mlx5_0/ports/1/state; "
        "cat /sys/class/infiniband/mlx5_0/ports/1/rate; "
        "cat /sys/class/infiniband/mlx5_0/ports/1/gids/3; "
        "cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/3; "
        "cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/3",
    ])
    checks["ip_link"] = command(["ip", "-brief", "address", "show", "enp161s0np0"])
    checks["modules"] = command([
        "bash", "-c", "lsmod | grep -E '^(nvidia_peermem|mlx5_ib|ib_uverbs) '",
    ])
    checks["disk"] = command(["df", "-h", "/mydata"])

    failures = []
    if topology.count("NVIDIA A100-SXM4-40GB") != 4:
        failures.append("expected four A100-SXM4-40GB GPUs")
    if topology.count("MIG 4g.20gb") != 1 or topology.count("MIG 3g.20gb") != 1:
        failures.append("expected fixed GPU0 4g+3g topology")
    if checks["compute_processes"]["stdout"].strip():
        failures.append("unexpected GPU compute process exists")
    running = checks["docker_ps"]["stdout"].strip()
    if running and not args.allow_running_containers:
        failures.append("unexpected running container exists")
    ib = checks["ib_state"]["stdout"]
    for expected in ("ACTIVE", "100 Gb/sec", "RoCE v2", "enp161s0np0"):
        if expected not in ib:
            failures.append(f"RDMA preflight missing {expected!r}")
    if "192.168.99.1" not in checks["ip_link"]["stdout"]:
        failures.append("RoCE loopback IPv4 address missing")

    tracked = []
    for relative in (
        "nrx_trt_direct.py",
        "p2p_overlap_bench.py",
        "qwen7b_gated.py",
        "isca_v2/dart_runtime.py",
        "isca_v2/test_dart_runtime.py",
    ):
        path = source_root / relative
        if not path.is_file():
            failures.append(f"missing source {path}")
            continue
        tracked.append({"path": str(path), "sha256": sha256(path)})

    result = {
        "schema": "dart-day1-preflight-v0",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "uid": os.getuid(),
        "utc_ns": time.time_ns(),
        "source_root": str(source_root),
        "tracked_sources": tracked,
        "checks": checks,
        "failures": failures,
        "pass": not failures,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        f"[DART-PREFLIGHT] {'PASS' if result['pass'] else 'FAIL'} "
        f"failures={len(failures)} output={output}", flush=True
    )
    for failure in failures:
        print(f"[ALERT] {failure}", flush=True)
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
