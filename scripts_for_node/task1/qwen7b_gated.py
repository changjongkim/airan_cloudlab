#!/usr/bin/env python3
"""Resident Qwen workload with cooperative enqueue gating.

The model remains loaded while a shared one-byte gate controls whether another
forward pass may be submitted.  Every submitted pass is synchronized so the
gate response is measured at a real GPU work boundary rather than at CPU enqueue
time.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/mydata/hf_cache")

import numpy as np
import torch
from transformers import AutoModelForCausalLM


STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def read_gate(path):
    try:
        return path.read_text(encoding="utf-8").strip() == "1"
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-file",
        help="optional cooperative state file (initializing/idle/busy/stopped)",
    )
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--mode", choices=("full_forward", "decode"), default="full_forward")
    parser.add_argument("--decode-steps-before-reset", type=int, default=128)
    parser.add_argument("--poll-ms", type=float, default=0.25)
    parser.add_argument(
        "--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B"))
    args = parser.parse_args()
    if (
        args.duration <= 0
        or args.sequence_length <= 0
        or args.poll_ms <= 0
        or args.decode_steps_before_reset <= 0
    ):
        parser.error("duration, sequence length, and poll interval must be positive")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    gate_path = Path(args.gate_file)
    ready_path = Path(args.ready_file)
    output_path = Path(args.output)
    state_path = Path(args.state_file) if args.state_file else None
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_state = None

    def publish_state(state):
        nonlocal last_state
        if state_path is None or state == last_state:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            f"{state} {time.monotonic_ns()}\n", encoding="utf-8")
        last_state = state

    publish_state("initializing")

    print(f"[QWEN-GATED] loading {args.model}", flush=True)
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    dummy = torch.randint(
        0, 32000, (1, args.sequence_length), device="cuda:0")
    past_key_values = None
    next_token = None
    decode_step = 0

    def prefill():
        nonlocal past_key_values, next_token, decode_step
        output = model(dummy, use_cache=True)
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1:, :].argmax(dim=-1)
        decode_step = 0

    with torch.no_grad():
        if args.mode == "decode":
            prefill()
        else:
            for _ in range(3):
                model(dummy)
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info(0)
    ready_ns = time.monotonic_ns()
    ready_path.write_text(str(ready_ns), encoding="utf-8")
    publish_state("idle")
    print(
        f"[QWEN-GATED] ready HBM={(total-free)/1e9:.2f}/{total/1e9:.2f}GB "
        f"seq={args.sequence_length} mode={args.mode}",
        flush=True,
    )

    transitions = []
    iterations = []
    last_gate = None
    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + round(args.duration * 1e9)
    try:
        with torch.no_grad():
            while not STOP and time.monotonic_ns() < deadline_ns:
                enabled = read_gate(gate_path)
                if enabled != last_gate:
                    transition_ns = time.monotonic_ns()
                    transitions.append({
                        "enabled": enabled,
                        "monotonic_ns": transition_ns,
                    })
                    print(
                        f"[QWEN-GATED] gate={'on' if enabled else 'off'} "
                        f"t={(transition_ns-start_ns)/1e9:.6f}s",
                        flush=True,
                    )
                    last_gate = enabled
                if not enabled:
                    publish_state("idle")
                    time.sleep(args.poll_ms / 1000.0)
                    continue
                iteration_start_ns = time.monotonic_ns()
                publish_state("busy")
                iteration_kind = args.mode
                if args.mode == "decode":
                    if decode_step >= args.decode_steps_before_reset:
                        iteration_kind = "prefill"
                        prefill()
                    else:
                        output = model(
                            next_token,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = output.past_key_values
                        next_token = output.logits[:, -1:, :].argmax(dim=-1)
                        decode_step += 1
                else:
                    model(dummy)
                torch.cuda.synchronize()
                iteration_end_ns = time.monotonic_ns()
                iterations.append({
                    "start_ns": iteration_start_ns,
                    "end_ns": iteration_end_ns,
                    "latency_ms": (iteration_end_ns - iteration_start_ns) / 1e6,
                    "kind": iteration_kind,
                })
    finally:
        end_ns = time.monotonic_ns()
        publish_state("stopped")
        result = {
            "model": args.model,
            "sequence_length": args.sequence_length,
            "mode": args.mode,
            "decode_steps_before_reset": args.decode_steps_before_reset,
            "resident_model": True,
            "synchronize_each_iteration": True,
            "ready_ns": ready_ns,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "wall_s": (end_ns - start_ns) / 1e9,
            "transitions": transitions,
            "iterations": iterations,
            "iteration_latency_ms": summarize([
                item["latency_ms"] for item in iterations]),
        }
        output_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"[QWEN-GATED] done iterations={len(iterations)} "
            f"output={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
