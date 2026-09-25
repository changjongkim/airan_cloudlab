#!/usr/bin/env python3
"""Validate both receiver branches on one paired PUSCH input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_receiver_phy import PairedDualReceiver
from softwall_phy import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20270401)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    for _ in range(args.warmup):
        receiver.run_conventional()
        receiver.run_neural()
    conventional = []
    neural = []
    for _ in range(args.iterations):
        conventional.append(receiver.run_conventional())
        neural.append(receiver.run_neural())
    result = {
        "schema": "softwall-dual-receiver-validation-v1",
        "iterations": args.iterations,
        "seed": args.seed,
        "conventional": {
            "gpu_ms": summary([item[0] for item in conventional]),
            "correct": sum(item[1] for item in conventional),
            "crc_failures": sum(item[2] for item in conventional),
            "payload_mismatches": sum(item[3] for item in conventional),
        },
        "neural": {
            "gpu_ms": summary([item[0] for item in neural]),
            "correct": sum(item[1] for item in neural),
            "crc_failures": sum(item[2] for item in neural),
            "payload_mismatches": sum(item[3] for item in neural),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["conventional"]["correct"] != args.iterations:
        raise SystemExit("conventional receiver failed paired correctness")
    if result["neural"]["correct"] != args.iterations:
        raise SystemExit("NeuralRx failed paired correctness")


if __name__ == "__main__":
    main()
