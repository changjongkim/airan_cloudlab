#!/usr/bin/env python3
"""Variable-length Qwen worker with a physical latest-start launch gate.

The legacy qwen_worker.py intentionally stays unchanged because it is part of
the frozen sources for Confirm97--105.  This worker accepts only a prospectively
declared set of context lengths and records the request identity and physical
GPU completion time for every RPC.
"""

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


def parse_lengths(text: str) -> tuple[int, ...]:
    try:
        lengths = tuple(sorted({int(item) for item in text.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("context lengths must be integers") from error
    if not lengths or lengths[0] <= 0:
        raise argparse.ArgumentTypeError("context lengths must be positive")
    return lengths


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


class TraceQwenPrefill:
    def __init__(
        self, model_name: str, allowed_lengths: tuple[int, ...], batch_size: int
    ) -> None:
        self.allowed_lengths = allowed_lengths
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            local_files_only=False,
        )
        self.model.eval()
        vocab_limit = min(int(self.model.config.vocab_size), 32_000)
        self.context = torch.randint(
            0, vocab_limit, (batch_size, allowed_lengths[-1]), device="cuda:0"
        )
        self.sink = None
        torch.cuda.synchronize()

    @staticmethod
    def memory_mib() -> dict[str, float]:
        scale = 1024 * 1024
        return {
            "allocated": torch.cuda.memory_allocated() / scale,
            "reserved": torch.cuda.memory_reserved() / scale,
            "max_allocated": torch.cuda.max_memory_allocated() / scale,
            "max_reserved": torch.cuda.max_memory_reserved() / scale,
        }

    def run(self, context_length: int) -> float:
        if context_length not in self.allowed_lengths:
            raise ValueError(f"context length {context_length} is outside allowed set")
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.inference_mode():
            # Autoregressive prefill consumes only the final prompt token's
            # logits.  AutoModelForCausalLM otherwise materializes logits for
            # every prompt token; at length 512 that is a large, unused
            # [batch, sequence, vocabulary] allocation which can evict the
            # four-cell cuPHY mode.  Run the full transformer over the prompt,
            # then apply the unchanged language-model head to the final hidden
            # state only.
            hidden = self.model.model(
                self.context[:, :context_length],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1:, :]
            self.sink = self.model.lm_head(hidden).squeeze(1)
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--allowed-context-lengths", type=parse_lengths, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-per-length", type=int, default=3)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup_per_length < 0:
        parser.error("batch size must be positive and warmup count nonnegative")

    torch.cuda.set_device(0)
    unit = TraceQwenPrefill(
        args.model, args.allowed_context_lengths, args.batch_size
    )
    memory_after_load = unit.memory_mib()
    warmup = []
    for length in args.allowed_context_lengths:
        for _ in range(args.warmup_per_length):
            warmup.append({"context_length": length, "gpu_ms": unit.run(length)})
    memory_after_warmup = unit.memory_mib()
    torch.cuda.empty_cache()
    memory_after_empty_cache = unit.memory_mib()

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    records = []
    rejected = []
    launch_guard_rejections = []
    started = time.perf_counter()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(f"[TRACE-QWEN] rpc ready {socket_path}", flush=True)
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            while True:
                line = channel.readline()
                if not line:
                    break
                try:
                    request = json.loads(line)
                    if request.get("op") == "stop":
                        channel.write(b'{"ok":true,"stopped":true}\n')
                        break
                    if request.get("op") != "run":
                        raise ValueError("unknown operation")
                    context_length = int(request["context_length"])
                    request_id = str(request["request_id"])
                    latest_start_ns = int(request["latest_start_ns"])
                    accepted_ns = time.perf_counter_ns()
                    if accepted_ns > latest_start_ns:
                        row = {
                            "request_id": request_id,
                            "context_length": context_length,
                            "accepted_ns": accepted_ns,
                            "latest_start_ns": latest_start_ns,
                            "lateness_ms": (accepted_ns - latest_start_ns) / 1e6,
                            "reason": "latest_start_expired_before_gpu_launch",
                        }
                        launch_guard_rejections.append(row)
                        response = {
                            "ok": True,
                            "launched": False,
                            **row,
                        }
                        channel.write(json.dumps(response).encode() + b"\n")
                        continue
                    gpu_ms = unit.run(context_length)
                    completed_ns = time.perf_counter_ns()
                    record = {
                        "request_id": request_id,
                        "context_length": context_length,
                        "latest_start_ns": latest_start_ns,
                        "accepted_ns": accepted_ns,
                        "completed_ns": completed_ns,
                        "gpu_ms": gpu_ms,
                    }
                    records.append(record)
                    response = {
                        "ok": True, "launched": True,
                        "unit": len(records), **record,
                    }
                except (KeyError, TypeError, ValueError) as error:
                    response = {"ok": False, "error": str(error)}
                    rejected.append({
                        "received_ns": time.perf_counter_ns(),
                        "request": request if isinstance(request, dict) else None,
                        "error": str(error),
                    })
                channel.write(json.dumps(response).encode() + b"\n")
    finally:
        result = {
            "schema": "softwall-trace-qwen-prefill-v1",
            "model": args.model,
            "phase": "prefill",
            "allowed_context_lengths": args.allowed_context_lengths,
            "batch_size": args.batch_size,
            "warmup_per_length": args.warmup_per_length,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mps_active_thread_percentage": os.environ.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            ),
            "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
            "visible_sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "implementation": "full-transformer-final-token-lm-head",
            "memory_mib": {
                "after_load": memory_after_load,
                "after_warmup": memory_after_warmup,
                "after_empty_cache": memory_after_empty_cache,
                "at_exit": unit.memory_mib(),
            },
            "wall_s": time.perf_counter() - started,
            "warmup": warmup,
            "gpu_ms": summarize([record["gpu_ms"] for record in records]),
            "completed_units": len(records),
            "launch_guard_rejections": launch_guard_rejections,
            "rejected": rejected,
            "records": records,
        }
        atomic_json(args.output, result)
        server.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
