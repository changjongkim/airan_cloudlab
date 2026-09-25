#!/usr/bin/env python3.11
"""Exploratory PHY complementarity screen with train-frozen feature bins.

This does not qualify an online policy; its purpose is to test whether a
channel feature contains information about NeuralRx's *incremental* decode
success over the conventional receiver on independent channel seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"][0]["records"]
    if not rows or any("observed_features" not in row for row in rows):
        raise ValueError(f"missing feature records in {path}")
    return rows


def wilson_95(success: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    p = success / count
    den = 1 + z * z / count
    mid = (p + z * z / (2 * count)) / den
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return [mid - half, mid + half]


def summarize(rows: list[dict], bounds: list[float]) -> list[dict]:
    bins = [[] for _ in range(len(bounds) + 1)]
    for row in rows:
        x = float(row["observed_features"]["channel_estimate_power"])
        index = next((i for i, edge in enumerate(bounds) if x < edge), len(bounds))
        bins[index].append(row)
    result = []
    for index, members in enumerate(bins):
        n = len(members)
        neural_only = sum(bool(x["neural_correct"]) and not bool(x["conventional_correct"]) for x in members)
        conventional_only = sum(bool(x["conventional_correct"]) and not bool(x["neural_correct"]) for x in members)
        both = sum(bool(x["conventional_correct"]) and bool(x["neural_correct"]) for x in members)
        neither = n - neural_only - conventional_only - both
        result.append({
            "bin": index,
            "feature_lower_inclusive": None if index == 0 else bounds[index - 1],
            "feature_upper_exclusive": None if index == len(bounds) else bounds[index],
            "n": n,
            "neural_only": neural_only,
            "conventional_only": conventional_only,
            "both": both,
            "neither": neither,
            "incremental_neural_success_rate": neural_only / n if n else None,
            "incremental_neural_success_wilson_95": wilson_95(neural_only, n),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train = load_records(args.train)
    test = load_records(args.test)
    train_seeds = {row["channel_seed"] for row in train}
    test_seeds = {row["channel_seed"] for row in test}
    if train_seeds & test_seeds:
        raise ValueError("train and test channel seeds overlap")
    powers = sorted(float(row["observed_features"]["channel_estimate_power"]) for row in train)
    bounds = [powers[int(len(powers) * i / 5)] for i in range(1, 5)]
    report = {
        "schema": "softwall-conditional-phy-value-exploratory-v1",
        "train": str(args.train),
        "test": str(args.test),
        "feature": "channel_estimate_power",
        "target": "NeuralRx correct AND conventional incorrect",
        "bin_edges_train_only": bounds,
        "train_bins": summarize(train, bounds),
        "test_bins": summarize(test, bounds),
        "limits": "Exploratory synthetic fixed-noise Rayleigh analysis; bin edges fixed from train. Sparse incremental successes and no online calibration, production-channel or deadline claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"edges": bounds, "train": [(x["n"], x["neural_only"]) for x in report["train_bins"]], "test": [(x["n"], x["neural_only"]) for x in report["test_bins"]]}))


if __name__ == "__main__":
    main()
