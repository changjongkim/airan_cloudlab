#!/usr/bin/env python3.11
"""Train-only bins for incremental NeuralRx rescue value on disjoint traces."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def rows(raw: dict, trials: int, seed: int) -> list[dict]:
    if (raw["noise_reference"] != "pre_fading"
            or not raw["observed_features_recorded"]
            or not raw["observed_features_prewarmed"]
            or raw["seed"] != seed or len(raw["results"]) != 1
            or raw["results"][0]["trials"] != trials):
        raise ValueError("raw PHY trace does not match frozen contract")
    result = raw["results"][0]
    if result["snr_db"] != -8.5 or len(result["records"]) != trials:
        raise ValueError("SNR or record count mismatch")
    answer = []
    for index, row in enumerate(result["records"]):
        feature = row["observed_features"]["channel_estimate_power"]
        if (row["trial"] != index or row["channel_seed"] != seed + index
                or not math.isfinite(feature)):
            raise ValueError("trial, channel seed or feature mismatch")
        answer.append({
            "x": feature,
            "rescue": int(row["neural_correct"] and not row["conventional_correct"]),
            "neural_success": int(row["neural_correct"]),
            "conventional_success": int(row["conventional_correct"]),
        })
    if (sum(row["rescue"] for row in answer) != result["neural_only_correct"]
            or sum(row["neural_success"] for row in answer) != result["neural_correct"]):
        raise ValueError("per-record outcomes disagree with raw aggregate")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm74_phy_value_protocol.json")
    raw = base / "raw"
    prefix = protocol["prefix"]
    manifest = read(raw / f"{prefix}_manifest.json")
    train = rows(read(raw / f"{prefix}_train.json"),
                 protocol["trials"], protocol["train_seed"])
    test = rows(read(raw / f"{prefix}_test.json"),
                protocol["trials"], protocol["test_seed"])
    train_features = sorted(row["x"] for row in train)
    edges = [train_features[round(k * len(train) / protocol["bins"])]
             for k in range(1, protocol["bins"])]
    train_bins = [[] for _ in range(protocol["bins"])]
    test_bins = [[] for _ in range(protocol["bins"])]
    for row in train:
        train_bins[bisect.bisect_right(edges, row["x"])].append(row)
    for row in test:
        test_bins[bisect.bisect_right(edges, row["x"])].append(row)
    bins = []
    predictions = []
    for index, (training, heldout) in enumerate(zip(train_bins, test_bins)):
        n_train, n_test = len(training), len(heldout)
        rescue_train = sum(row["rescue"] for row in training)
        rescue_test = sum(row["rescue"] for row in heldout)
        p_hat = (rescue_train + 0.5) / (n_train + 1)
        neural_hat = (sum(row["neural_success"] for row in training) + 0.5) / (n_train + 1)
        predictions.extend((p_hat, row["rescue"]) for row in heldout)
        bins.append({
            "index": index,
            "feature_lower_inclusive": None if index == 0 else edges[index - 1],
            "feature_upper_exclusive": None if index == protocol["bins"] - 1 else edges[index],
            "train_n": n_train, "test_n": n_test,
            "train_rescue": rescue_train, "test_rescue": rescue_test,
            "train_incremental_rescue_probability": p_hat,
            "train_neural_success_probability": neural_hat,
            "test_rescue_rate": rescue_test / n_test,
            "test_rescue_wilson95": wilson(rescue_test, n_test),
        })
    prevalence_train = sum(row["rescue"] for row in train) / len(train)
    brier_model = sum((p - y) ** 2 for p, y in predictions) / len(predictions)
    brier_constant = sum((prevalence_train - y) ** 2 for _, y in predictions) / len(predictions)
    weighted_mace = sum(
        item["test_n"] * abs(item["train_incremental_rescue_probability"]
                             - item["test_rescue_rate"])
        for item in bins
    ) / len(test)
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "provenance": (
            str(manifest["slurm_job_id"]) == protocol["job"]
            and manifest["trials"] == protocol["trials"]
            and manifest["train_seed"] == protocol["train_seed"]
            and manifest["test_seed"] == protocol["test_seed"]
            and manifest["host"] == protocol["node"]
        ),
        "disjoint": (
            abs(protocol["train_seed"] - protocol["test_seed"]) >= protocol["trials"]
        ),
        "sample_size": (
            all(item["train_n"] >= protocol["min_bin_n"]
                and item["test_n"] >= protocol["min_bin_n"] for item in bins)
            and sum(row["rescue"] for row in train) >= protocol["min_rescue_events"]
            and sum(row["rescue"] for row in test) >= protocol["min_rescue_events"]
        ),
        "calibration": weighted_mace <= protocol["max_weighted_mace"],
        "predictive_value": brier_model < brier_constant,
    }
    report = {
        "schema": "softwall-confirm74-phy-value-calibration-v1",
        "job": protocol["job"], "node": manifest["host"],
        "all_pass": all(gates.values()), "gates": gates,
        "train_rescue": sum(row["rescue"] for row in train),
        "test_rescue": sum(row["rescue"] for row in test),
        "train_only_edges": edges, "bins": bins,
        "weighted_mace": weighted_mace,
        "brier_model": brier_model,
        "brier_train_prevalence_constant": brier_constant,
        "interpretation": "Disjoint 5,000/5,000 synthetic fixed-pre-fading-noise PHY trials. Train-only feature bins estimate incremental NeuralRx rescue, with held-out calibration and Brier comparison. No online scheduling, q_use timing, TDL/DU generalization, or production guarantee.",
    }
    path = base / f"confirm74_phy_value_job{protocol['job']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_pass": report["all_pass"],
        "failed": [key for key, ok in gates.items() if not ok],
        "train_rescue": report["train_rescue"],
        "test_rescue": report["test_rescue"],
        "weighted_mace": weighted_mace,
        "brier_model": brier_model,
        "brier_constant": brier_constant,
    }, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
