#!/usr/bin/env python3
"""Qwen inference work units for uncontrolled and S0 SoftWall experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


class QwenWorkUnit:
    def __init__(
        self, model_name: str, phase: str, context_length: int, batch_size: int
    ) -> None:
        self.phase = phase
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            local_files_only=False,
        )
        self.model.eval()
        vocab_limit = min(int(self.model.config.vocab_size), 32_000)
        self.context = torch.randint(
            0, vocab_limit, (batch_size, context_length), device="cuda:0"
        )
        self.next_token = torch.randint(
            0, vocab_limit, (batch_size, 1), device="cuda:0"
        )
        self.past_key_values = None
        self.sink = None
        if phase == "decode":
            with torch.inference_mode():
                output = self.model(self.context, use_cache=True)
                self.past_key_values = output.past_key_values
                self.sink = output.logits[:, -1, :]
        torch.cuda.synchronize()

    def run(self) -> float:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.inference_mode():
            if self.phase == "prefill":
                output = self.model(self.context, use_cache=False)
            else:
                output = self.model(
                    self.next_token,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
                self.past_key_values = output.past_key_values
            self.sink = output.logits[:, -1, :]
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end))


def report(args: argparse.Namespace, records: list[dict], wall_s: float) -> dict:
    completed = len(records)
    return {
        "schema": "softwall-qwen-unit-v1",
        "kind": "qwen",
        "mode": args.mode,
        "phase": args.phase,
        "model": args.model,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
        "visible_sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "completed_units": completed,
        "wall_s": wall_s,
        "units_per_s": completed / wall_s if wall_s > 0 else 0.0,
        "gpu_ms": summarize([item["gpu_ms"] for item in records]),
        "records": records,
    }


def run_continuous(args: argparse.Namespace, unit: QwenWorkUnit) -> None:
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
    wall_s = time.perf_counter() - started
    result = report(args, records, wall_s)
    atomic_json(Path(args.output), result)
    print(
        f"[QWEN-WORKER] {args.phase} continuous units={len(records)} "
        f"rate={result['units_per_s']:.3f}/s max={result['gpu_ms']['max']:.3f}ms",
        flush=True,
    )


def run_rpc(args: argparse.Namespace, unit: QwenWorkUnit) -> None:
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    records = []
    started = time.perf_counter()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(f"[QWEN-WORKER] rpc ready {socket_path}", flush=True)
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            while True:
                line = channel.readline()
                if not line:
                    break
                request = json.loads(line)
                if request.get("op") == "stop":
                    channel.write(b'{"ok":true,"stopped":true}\n')
                    break
                if request.get("op") != "run":
                    channel.write(b'{"ok":false,"error":"unknown operation"}\n')
                    continue
                gpu_ms = unit.run()
                record = {
                    "completed_ns": time.perf_counter_ns(),
                    "gpu_ms": gpu_ms,
                }
                records.append(record)
                channel.write(
                    json.dumps(
                        {
                            "ok": True,
                            "unit": len(records),
                            "gpu_ms": gpu_ms,
                            "completed_ns": record["completed_ns"],
                        }
                    ).encode()
                    + b"\n"
                )
    finally:
        atomic_json(
            Path(args.output), report(args, records, time.perf_counter() - started)
        )
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("continuous", "rpc"), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--socket")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--ready-file")
    parser.add_argument("--start-file")
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    if args.mode == "rpc" and not args.socket:
        parser.error("--socket is required in rpc mode")
    if args.context_length <= 0 or args.batch_size <= 0:
        parser.error("context length and batch size must be positive")
    torch.cuda.set_device(0)
    unit = QwenWorkUnit(
        args.model, args.phase, args.context_length, args.batch_size
    )
    for _ in range(3):
        unit.run()
    if args.mode == "continuous":
        run_continuous(args, unit)
    else:
        run_rpc(args, unit)


if __name__ == "__main__":
    main()
