#!/usr/bin/env python3
"""Repeat selected transition traces to separate PHY instability from scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver


def outcome(value: tuple[float, bool, int, int]) -> dict:
    return {
        "gpu_ms": value[0],
        "correct": value[1],
        "crc_failures": value[2],
        "payload_mismatches": value[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, default=-8.5)
    parser.add_argument("--payload-seed-base", type=int, default=20270500)
    parser.add_argument("--channel-seed-base", type=int, default=202609200)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trace", action="append", required=True,
                        help="round:index")
    args = parser.parse_args()
    traces: dict[int, list[int]] = {}
    for item in args.trace:
        round_text, index_text = item.split(":", 1)
        traces.setdefault(int(round_text), []).append(int(index_text))

    results = []
    for round_index, indices in sorted(traces.items()):
        receiver = PairedDualReceiver(
            args.engine, seed=args.payload_seed_base + round_index
        )
        for _ in range(10):
            receiver.run_neural()
            receiver.run_conventional()
        for index in indices:
            channel_seed = (
                args.channel_seed_base + round_index * 100000 + index
            )
            receiver.apply_rayleigh_awgn(args.snr_db, channel_seed)
            frozen = cp.array(receiver.rx_slot, order="F", copy=True)
            repetitions = []
            for repeat in range(args.repeats):
                receiver.rx_slot = cp.array(frozen, order="F", copy=True)
                conventional_only = receiver.run_conventional()
                receiver.rx_slot = cp.array(frozen, order="F", copy=True)
                nrx_only = receiver.run_neural()
                receiver.rx_slot = cp.array(frozen, order="F", copy=True)
                neural_first = receiver.run_neural()
                mutated_by_neural = not bool(
                    cp.array_equal(receiver.rx_slot, frozen)
                )
                conventional_after_neural = receiver.run_conventional()
                mutated_by_conventional = not bool(
                    cp.array_equal(receiver.rx_slot, frozen)
                )
                repetitions.append({
                    "repeat": repeat,
                    "conventional_only": outcome(conventional_only),
                    "nrx_only": outcome(nrx_only),
                    "neural_first": outcome(neural_first),
                    "conventional_after_neural": outcome(
                        conventional_after_neural
                    ),
                    "rx_mutated_by_neural": mutated_by_neural,
                    "rx_mutated_after_both": mutated_by_conventional,
                })
            results.append({
                "round": round_index,
                "index": index,
                "payload_seed": args.payload_seed_base + round_index,
                "channel_seed": channel_seed,
                "counts": {
                    key: sum(item[key]["correct"] for item in repetitions)
                    for key in (
                        "conventional_only",
                        "nrx_only",
                        "neural_first",
                        "conventional_after_neural",
                    )
                },
                "rx_mutation_observed": any(
                    item["rx_mutated_by_neural"]
                    or item["rx_mutated_after_both"]
                    for item in repetitions
                ),
                "repetitions": repetitions,
            })
    report = {
        "schema": "softwall-receiver-stability-v1",
        "snr_db": args.snr_db,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in results:
        print(
            f"r{item['round']} i{item['index']} counts={item['counts']} "
            f"rx_mutated={item['rx_mutation_observed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
