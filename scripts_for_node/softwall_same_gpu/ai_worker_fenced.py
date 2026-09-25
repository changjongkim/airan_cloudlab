#!/usr/bin/env python3
"""AI worker with a GPU-synchronized completion marker for RPC fault trials."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import cupy as cp

from softwall_phy import summary


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class WorkUnit:
    def __init__(
        self,
        kind: str,
        matrix_size: int,
        hbm_mb: int,
        repeats: int,
        engine: str | None,
    ) -> None:
        self.kind = kind
        self.repeats = repeats
        self.nrx = None
        if kind == "nrx":
            from nrx_trt_direct import DirectNrx

            if not engine:
                raise RuntimeError("NeuralRx requires a TensorRT engine")
            self.nrx = DirectNrx(engine)
            self.nrx.capture_graph()
            self.stream = self.nrx.stream
            return

        self.stream = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            if kind == "gemm":
                self.left = cp.random.standard_normal(
                    (matrix_size, matrix_size), dtype=cp.float32)
                self.right = cp.random.standard_normal(
                    (matrix_size, matrix_size), dtype=cp.float32)
                self.output = cp.empty_like(self.left)
            else:
                elements = hbm_mb * 1024 * 1024 // 4
                self.source = cp.random.standard_normal(elements, dtype=cp.float32)
                self.destination = cp.empty_like(self.source)
        self.stream.synchronize()

    def run(self) -> float:
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        with self.stream:
            begin.record()
            for _ in range(self.repeats):
                if self.kind == "gemm":
                    cp.matmul(self.left, self.right, out=self.output)
                elif self.kind == "hbm":
                    cp.copyto(self.destination, self.source)
                    cp.copyto(self.source, self.destination)
                else:
                    self.nrx.launch(use_graph=True)
            end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(begin, end))


def report(args: argparse.Namespace, records: list[dict], wall_s: float) -> dict:
    completed = len(records)
    return {
        "schema": "softwall-ai-unit-v1",
        "kind": args.kind,
        "mode": args.mode,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
        "visible_sm_count": int(
            cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        ),
        "matrix_size": args.matrix_size,
        "hbm_mb": args.hbm_mb,
        "repeats": args.repeats,
        "pre_ready_gpu_warmup_units": 10,
        "engine": args.engine if args.kind == "nrx" else None,
        "completion_marker": args.completion_marker,
        "joint_response_delay_ms": args.joint_response_delay_ms,
        "completed_units": completed,
        "wall_s": wall_s,
        "units_per_s": completed / wall_s if wall_s > 0 else 0.0,
        "gpu_ms": summary([x["gpu_ms"] for x in records]),
        "records": records,
    }


def run_continuous(args: argparse.Namespace, unit: WorkUnit) -> None:
    if args.ready_file:
        Path(args.ready_file).touch()
    if args.start_file:
        while not Path(args.start_file).exists():
            time.sleep(0.001)
    records = []
    started = time.perf_counter()
    while time.perf_counter() - started < args.duration_s:
        if args.stop_file and Path(args.stop_file).exists():
            break
        gpu_ms = unit.run()
        records.append({"completed_ns": time.perf_counter_ns(), "gpu_ms": gpu_ms})
    result = report(args, records, time.perf_counter() - started)
    atomic_json(Path(args.output), result)
    print(
        f"[AI-WORKER] {args.kind} continuous units={len(records)} "
        f"rate={result['units_per_s']:.3f}/s max={result['gpu_ms']['max']:.3f}ms",
        flush=True,
    )


def run_rpc(args: argparse.Namespace, unit: WorkUnit) -> None:
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    durations = []
    started = time.perf_counter()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(f"[AI-WORKER] rpc ready {socket_path}", flush=True)
    quiesced = False
    joint_response_delayed = False
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            while True:
                try:
                    line = channel.readline()
                except (ConnectionError, OSError):
                    return
                if not line:
                    break
                request = json.loads(line)
                if request.get("op") == "stop":
                    channel.write(b'{"ok":true,"stopped":true}\n')
                    break
                if request.get("op") == "quiesce":
                    quiesced = True
                    channel.write(b'{"ok":true,"quiesced":true}\n')
                    continue
                if quiesced:
                    channel.write(b'{"ok":false,"error":"quiesced"}\n')
                    continue
                if request.get("op") != "run":
                    channel.write(b'{"ok":false,"error":"unknown operation"}\n')
                    continue
                if 0 <= args.fault_after_units <= len(durations):
                    print(
                        f"[AI-WORKER] injected disconnect after {len(durations)} units",
                        flush=True,
                    )
                    return
                duration_ms = unit.run()
                durations.append(duration_ms)
                completed_ns = time.perf_counter_ns()
                if request.get("joint"):
                    if not args.completion_marker or not request.get("lease_id"):
                        raise RuntimeError("joint request requires a completion marker and lease id")
                    # unit.run() synchronized its CUDA end event. The atomic
                    # rename makes this a physical-completion fence that can
                    # be checked independently of a delayed RPC response.
                    atomic_json(Path(args.completion_marker), {
                        "lease_id": request["lease_id"],
                        "unit": len(durations),
                        "completed_ns": completed_ns,
                        "gpu_ms": duration_ms,
                        "host": platform.node(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    })
                response = json.dumps({
                    "ok": True,
                    "unit": len(durations),
                    "gpu_ms": duration_ms,
                    "completed_ns": completed_ns,
                }).encode() + b"\n"
                if args.response_delay_ms:
                    time.sleep(args.response_delay_ms / 1000.0)
                if (request.get("joint") and not joint_response_delayed
                        and args.joint_response_delay_ms):
                    joint_response_delayed = True
                    time.sleep(args.joint_response_delay_ms / 1000.0)
                try:
                    channel.write(response)
                except BrokenPipeError:
                    return
    finally:
        wall_s = time.perf_counter() - started
        records = [
            {"completed_ns": 0, "gpu_ms": value}
            for value in durations
        ]
        atomic_json(Path(args.output), report(args, records, wall_s))
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("gemm", "hbm", "nrx"), required=True)
    parser.add_argument("--mode", choices=("continuous", "rpc"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--socket")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--hbm-mb", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--engine", default=os.environ.get("SOFTWALL_NRX_ENGINE"))
    parser.add_argument("--ready-file")
    parser.add_argument("--start-file")
    parser.add_argument("--stop-file")
    parser.add_argument("--fault-after-units", type=int, default=-1)
    parser.add_argument("--response-delay-ms", type=float, default=0.0)
    parser.add_argument("--completion-marker")
    parser.add_argument("--joint-response-delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == "rpc" and not args.socket:
        parser.error("--socket is required in rpc mode")
    if min(args.matrix_size, args.hbm_mb, args.repeats) <= 0:
        parser.error("work-unit sizes must be positive")
    if min(args.response_delay_ms, args.joint_response_delay_ms) < 0:
        parser.error("response delay must be non-negative")
    cp.cuda.runtime.setDevice(0)
    unit = WorkUnit(
        args.kind, args.matrix_size, args.hbm_mb, args.repeats, args.engine
    )
    for _ in range(10):
        unit.run()
    if args.mode == "continuous":
        run_continuous(args, unit)
    else:
        run_rpc(args, unit)


if __name__ == "__main__":
    main()
