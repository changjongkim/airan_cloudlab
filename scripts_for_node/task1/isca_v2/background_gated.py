#!/usr/bin/env python3
"""Cooperatively gated, architecture-realistic background GPU workloads.

This is an interference/lease benchmark, not a model-quality benchmark.
ResNet, BERT, and Whisper execute their real framework graphs with synthetic
inputs and randomly initialized weights.  Qwen uses the separately cached,
pretrained workload runner.  Synchronizing each work unit gives DART-L an
observable, bounded drain point.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch


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


def publish(path, state):
    if path is not None:
        path.write_text(f"{state} {time.monotonic_ns()}\n", encoding="utf-8")


def build_workload(name):
    """Return (callable, metadata) for one bounded GPU work unit."""
    if name == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None).cuda().half().eval()
        value = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float16)

        def run():
            model(value)

        metadata = {
            "architecture": "torchvision.resnet50",
            "batch": 32,
            "input_shape": list(value.shape),
        }
    elif name == "bert_base":
        from transformers import BertConfig, BertModel

        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        model = BertModel(config).cuda().half().eval()
        tokens = torch.randint(0, config.vocab_size, (8, 128), device="cuda")
        mask = torch.ones_like(tokens)

        def run():
            model(input_ids=tokens, attention_mask=mask)

        metadata = {
            "architecture": "transformers.BertModel/base",
            "batch": 8,
            "sequence_length": 128,
        }
    elif name == "whisper_base":
        from transformers import WhisperConfig, WhisperModel

        config = WhisperConfig(
            d_model=512,
            encoder_layers=6,
            encoder_attention_heads=8,
            encoder_ffn_dim=2048,
            decoder_layers=6,
            decoder_attention_heads=8,
            decoder_ffn_dim=2048,
        )
        model = WhisperModel(config).encoder.cuda().half().eval()
        features = torch.randn(
            1, config.num_mel_bins, config.max_source_positions * 2,
            device="cuda", dtype=torch.float16,
        )

        def run():
            model(input_features=features)

        metadata = {
            "architecture": "transformers.WhisperEncoder/base",
            "batch": 1,
            "input_shape": list(features.shape),
        }
    else:
        raise ValueError(name)
    metadata.update({
        "weights": "synthetic_random",
        "inputs": "synthetic",
        "purpose": "GPU interference and lease-drain characterization only",
    })
    return run, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", choices=("resnet50", "bert_base", "whisper_base"),
        required=True,
    )
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--poll-ms", type=float, default=0.25)
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.poll_ms <= 0:
        parser.error("duration/poll-ms must be positive and warmup non-negative")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    gate = Path(args.gate_file)
    state = Path(args.state_file)
    ready = Path(args.ready_file)
    output = Path(args.output)
    for path in (gate, state, ready, output):
        path.parent.mkdir(parents=True, exist_ok=True)

    publish(state, "initializing")
    torch.cuda.set_device(0)
    run, metadata = build_workload(args.workload)
    with torch.inference_mode():
        for _ in range(args.warmup):
            run()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    ready_ns = time.monotonic_ns()
    ready.write_text(str(ready_ns), encoding="utf-8")
    publish(state, "idle")
    print(
        f"[BACKGROUND] ready workload={args.workload} "
        f"resident={(total_bytes-free_bytes)/1e9:.3f}GB",
        flush=True,
    )

    transitions = []
    units = []
    last_gate = None
    last_state = "idle"
    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + round(args.duration * 1e9)
    try:
        with torch.inference_mode():
            while not STOP and time.monotonic_ns() < deadline_ns:
                enabled = read_gate(gate)
                if enabled != last_gate:
                    transitions.append({
                        "enabled": enabled,
                        "monotonic_ns": time.monotonic_ns(),
                    })
                    last_gate = enabled
                if not enabled:
                    if last_state != "idle":
                        publish(state, "idle")
                        last_state = "idle"
                    time.sleep(args.poll_ms / 1000.0)
                    continue
                begin_ns = time.monotonic_ns()
                if last_state != "busy":
                    publish(state, "busy")
                    last_state = "busy"
                run()
                torch.cuda.synchronize()
                end_ns = time.monotonic_ns()
                units.append({
                    "start_ns": begin_ns,
                    "end_ns": end_ns,
                    "latency_ms": (end_ns - begin_ns) / 1e6,
                })
    finally:
        end_ns = time.monotonic_ns()
        publish(state, "stopped")
        result = {
            "schema": "dart-background-gated-v0",
            "workload": args.workload,
            "metadata": metadata,
            "resident": True,
            "cooperative_unit_boundary": True,
            "synchronize_each_unit": True,
            "ready_ns": ready_ns,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "wall_s": (end_ns - start_ns) / 1e9,
            "transitions": transitions,
            "units": units,
            "unit_latency_ms": summarize([
                item["latency_ms"] for item in units
            ]),
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(
            f"[BACKGROUND] done workload={args.workload} units={len(units)} "
            f"output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
