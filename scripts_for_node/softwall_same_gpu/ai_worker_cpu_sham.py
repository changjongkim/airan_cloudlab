#!/usr/bin/env python3
"""Socket-compatible retirement control that never creates a CUDA context."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    print(f"[CPU-SHAM] rpc ready {args.socket}", flush=True)
    completed = 0
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            while line := channel.readline():
                request = json.loads(line)
                if request.get("op") == "stop":
                    channel.write(b'{"ok":true,"stopped":true}\n')
                    break
                if request.get("op") == "run":
                    completed += 1
                    channel.write(b'{"ok":true,"unit":1,"gpu_ms":0.0}\n')
                else:
                    channel.write(b'{"ok":false,"error":"unknown operation"}\n')
    finally:
        report = {
            "schema": "softwall-cpu-sham-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "completed_units": completed,
            "cuda_context_created": False,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
