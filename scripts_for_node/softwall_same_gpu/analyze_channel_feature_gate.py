#!/usr/bin/env python3
"""Held-out qualification of an observable conventional-success gate.

This is a baseline ingredient, not a SoftWall novelty or deadline result.
The threshold is chosen only on the first payload/channel seed and evaluated
unchanged on the independent second seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FEATURE = "channel_estimate_power"


def load(path: Path, noise_reference: str, seed: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data["schema"] != "softwall-dual-receiver-snr-v2"
        or data["noise_reference"] != noise_reference
        or data["seed"] != seed
        or not data["observed_features_recorded"]
        or len(data["results"]) != 1
        or data["results"][0]["snr_db"] != -8.5
    ):
        raise ValueError(f"calibration input contract mismatch: {path}")
    records = data["results"][0]["records"]
    if len(records) != 500 or sorted(row["trial"] for row in records) != list(range(500)):
        raise ValueError(f"incomplete trials: {path}")
    for row in records:
        feature = row["observed_features"]
        if any(
            not math.isfinite(feature[name]) or feature[name] < 0
            for name in (FEATURE, "received_grid_power", "gpu_ms")
        ):
            raise ValueError(f"invalid observable feature: {path}")
    return records


def auc(rows: list[dict]) -> float | None:
    """AUC for larger observable channel power predicting conventional CRC."""

    ordered = sorted(
        (row["observed_features"][FEATURE], bool(row["conventional_correct"]))
        for row in rows
    )
    positives = sum(label for _, label in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        rank_sum += mean_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def gate(rows: list[dict], threshold: float) -> dict:
    skipped = [row for row in rows if row["observed_features"][FEATURE] >= threshold]
    neural_only_lost = sum(
        row["neural_correct"] and not row["conventional_correct"] for row in skipped
    )
    return {
        "skipped_nrx": len(skipped),
        "skipped_fraction": len(skipped) / len(rows),
        "neural_only_lost": neural_only_lost,
        "radio_loss_fraction": neural_only_lost / len(rows),
    }


def select_threshold(train: list[dict]) -> tuple[float, dict]:
    # The frozen train-side allowance is at most 2/500 radio successes lost.
    # Evaluate every distinct feature value plus the never-skip sentinel.
    candidates = sorted({row["observed_features"][FEATURE] for row in train})
    candidates.append(math.nextafter(candidates[-1], math.inf))
    feasible = [(gate(train, threshold)["skipped_nrx"], threshold) for threshold in candidates
                if gate(train, threshold)["neural_only_lost"] <= 2]
    _, threshold = max(feasible, key=lambda item: (item[0], item[1]))
    return threshold, gate(train, threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pre", type=Path, required=True)
    parser.add_argument("--train-post", type=Path, required=True)
    parser.add_argument("--test-pre", type=Path, required=True)
    parser.add_argument("--test-post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (20356001, 20356002)
    train = load(args.train_pre, "pre_fading", seeds[0])
    train_post = load(args.train_post, "post_fading", seeds[0])
    test = load(args.test_pre, "pre_fading", seeds[1])
    test_post = load(args.test_post, "post_fading", seeds[1])
    train_channels = {row["channel_seed"] for row in train}
    test_channels = {row["channel_seed"] for row in test}
    overlap = train_channels & test_channels
    if overlap:
        raise ValueError(
            f"train/test channel realizations are not independent: "
            f"{len(overlap)} overlapping seeds"
        )
    for pre, post in ((train, train_post), (test, test_post)):
        if any(
            (a["trial"], a["channel_seed"]) != (b["trial"], b["channel_seed"])
            for a, b in zip(pre, post)
        ):
            raise ValueError("pre/post noise conventions lack paired channel seeds")
    threshold, train_gate = select_threshold(train)
    test_gate = gate(test, threshold)
    test_neural_only = sum(
        row["neural_correct"] and not row["conventional_correct"] for row in test
    )
    max_feature_ms = max(
        row["observed_features"]["gpu_ms"]
        for rows in (train, train_post, test, test_post) for row in rows
    )
    report = {
        "schema": "softwall-channel-feature-gate-v1",
        "feature": FEATURE,
        "train_seed": seeds[0],
        "test_seed": seeds[1],
        "snr_db": -8.5,
        "trials_per_arm": 500,
        "train_auc_conventional_correct": auc(train),
        "test_auc_conventional_correct": auc(test),
        "post_fading_auc_train": auc(train_post),
        "post_fading_auc_test": auc(test_post),
        "threshold_selected_on_train": threshold,
        "train_gate": train_gate,
        "test_gate": test_gate,
        "test_neural_only_correct": test_neural_only,
        "max_feature_gpu_ms": max_feature_ms,
        "diagnostic_gates": {
            "train_loss_at_most_two": train_gate["neural_only_lost"] <= 2,
            "heldout_skip_at_least_ten_percent": test_gate["skipped_nrx"] >= 50,
            "heldout_radio_loss_at_most_one_percent": test_gate["neural_only_lost"] <= 5,
            "heldout_has_neural_only_cases": test_neural_only >= 50,
            "feature_gpu_max_under_five_ms": max_feature_ms < 5.0,
        },
        "evidence_scope": "offline paired-branch calibration only; no online admission, AI throughput, PHY expiry or WCET claim",
    }
    report["diagnostic_pass"] = all(report["diagnostic_gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["diagnostic_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
