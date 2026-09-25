#!/usr/bin/env python3
"""Measure conventional and NeuralRx utility on identical fading/noisy slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_receiver_phy import PairedDualReceiver
from softwall_phy import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--snr-db", default="-8,-6,-4,-2,0,2,4")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20270600)
    parser.add_argument(
        "--noise-reference", choices=("post_fading", "pre_fading"),
        default="post_fading",
    )
    parser.add_argument("--record-observed-features", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snrs = [float(item) for item in args.snr_db.split(",")]
    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    receiver.restore_clean_slot()
    for _ in range(10):
        receiver.run_conventional()
        receiver.run_neural()

    results = []
    for snr_index, snr_db in enumerate(snrs):
        records = []
        for trial in range(args.trials):
            channel_seed = args.seed + snr_index * 100_000 + trial
            receiver.apply_rayleigh_awgn(
                snr_db, channel_seed, noise_reference=args.noise_reference
            )
            features = (
                receiver.observed_channel_features()
                if args.record_observed_features else None
            )
            conventional = receiver.run_conventional()
            neural = receiver.run_neural()
            record = {
                "trial": trial,
                "channel_seed": channel_seed,
                "conventional_correct": conventional[1],
                "neural_correct": neural[1],
                "conventional_gpu_ms": conventional[0],
                "neural_gpu_ms": neural[0],
                "conventional_crc_failures": conventional[2],
                "neural_crc_failures": neural[2],
                "conventional_payload_mismatches": conventional[3],
                "neural_payload_mismatches": neural[3],
            }
            if features is not None:
                record["observed_features"] = features
            records.append(record)
        both = sum(
            item["conventional_correct"] and item["neural_correct"]
            for item in records
        )
        conventional_only = sum(
            item["conventional_correct"] and not item["neural_correct"]
            for item in records
        )
        neural_only = sum(
            item["neural_correct"] and not item["conventional_correct"]
            for item in records
        )
        neither = len(records) - both - conventional_only - neural_only
        results.append({
            "snr_db": snr_db,
            "trials": args.trials,
            "conventional_correct": both + conventional_only,
            "neural_correct": both + neural_only,
            "both_correct": both,
            "conventional_only_correct": conventional_only,
            "neural_only_correct": neural_only,
            "neither_correct": neither,
            "oracle_union_correct": both + conventional_only + neural_only,
            "conventional_gpu_ms": summary(
                [item["conventional_gpu_ms"] for item in records]
            ),
            "neural_gpu_ms": summary([item["neural_gpu_ms"] for item in records]),
            "records": records,
        })
        print(
            f"[SNR] {snr_db:+.1f}dB conv={both + conventional_only}/{args.trials} "
            f"nrx={both + neural_only}/{args.trials} union={both + conventional_only + neural_only}/{args.trials} "
            f"conv_only={conventional_only} nrx_only={neural_only}",
            flush=True,
        )
    document = {
        "schema": "softwall-dual-receiver-snr-v2",
        "channel": "independent per-slot block Rayleigh coefficient per RX antenna plus complex AWGN",
        "noise_reference": args.noise_reference,
        "snr_definition": (
            "mean post-fading resource-grid power divided by mean complex noise power"
            if args.noise_reference == "post_fading" else
            "mean clean transmit-grid power divided by mean complex noise power"
        ),
        "observed_features_recorded": args.record_observed_features,
        "seed": args.seed,
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
