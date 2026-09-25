#!/usr/bin/env python3
"""Profile the paired NeuralRx pipeline stage by stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_receiver_phy import PairedDualReceiver
from softwall_phy import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20270402)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    for _ in range(args.warmup):
        receiver.run_neural()
    records = [receiver.profile_neural_once() for _ in range(args.iterations)]
    stage_names = list(records[0]["stages_gpu_ms"])
    result = {
        "schema": "softwall-dual-receiver-profile-v1",
        "iterations": args.iterations,
        "correct": sum(item["correct"] for item in records),
        "total_gpu_ms": summary([item["total_gpu_ms"] for item in records]),
        "stages_gpu_ms": {
            name: summary([item["stages_gpu_ms"][name] for item in records])
            for name in stage_names
        },
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
