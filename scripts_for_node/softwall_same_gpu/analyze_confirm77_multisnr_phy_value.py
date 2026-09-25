#!/usr/bin/env python3.11
"""Train-only 2D observable-feature bins across a held-out SNR mixture."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def samples(raw: dict, seed: int, trials: int, snrs: list[float]) -> list[dict]:
    if (raw["seed"] != seed or raw["noise_reference"] != "pre_fading"
            or not raw["observed_features_recorded"]
            or not raw["observed_features_prewarmed"]
            or len(raw["results"]) != len(snrs)):
        raise ValueError("mixed-SNR raw trace contract mismatch")
    output = []
    for snr_index, (snr, result) in enumerate(zip(snrs, raw["results"])):
        if (result["snr_db"] != snr or result["trials"] != trials
                or len(result["records"]) != trials):
            raise ValueError("SNR, trial count or records mismatch")
        rescue_count = 0
        for trial, row in enumerate(result["records"]):
            feature = row["observed_features"]
            x = feature["channel_estimate_power"]
            y = feature["received_grid_power"]
            if (row["trial"] != trial
                    or row["channel_seed"] != seed + snr_index * 100_000 + trial
                    or not math.isfinite(x) or not math.isfinite(y)):
                raise ValueError("trial, seed or observable feature invalid")
            rescue = int(row["neural_correct"] and not row["conventional_correct"])
            rescue_count += rescue
            output.append({"x": x, "y": y, "rescue": rescue,
                           "neural_success": int(row["neural_correct"]),
                           "snr_db_evaluation_only": snr,
                           "channel_seed": row["channel_seed"]})
        if rescue_count != result["neural_only_correct"]:
            raise ValueError("raw aggregate rescue disagrees with records")
    return output


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((pred - truth) ** 2 for pred, truth in pairs) / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm77_multisnr_protocol.json")
    raw = base / "raw"
    prefix = protocol["prefix"]
    manifest = read(raw / f"{prefix}_manifest.json")
    train = samples(read(raw / f"{prefix}_train.json"),
                    protocol["train_seed"], protocol["trials_per_snr"], protocol["snrs_db"])
    test = samples(read(raw / f"{prefix}_test.json"),
                   protocol["test_seed"], protocol["trials_per_snr"], protocol["snrs_db"])
    xs = sorted(row["x"] for row in train)
    x_edges = [xs[round(k * len(xs) / protocol["x_bins"])]
               for k in range(1, protocol["x_bins"])]
    training_xbins = [[] for _ in range(protocol["x_bins"])]
    for row in train:
        training_xbins[bisect.bisect_right(x_edges, row["x"])].append(row)
    y_edges = [sorted(row["y"] for row in group)[len(group) // 2]
               for group in training_xbins]
    leaves_train = [[] for _ in range(2 * protocol["x_bins"])]
    leaves_test = [[] for _ in range(2 * protocol["x_bins"])]

    def index(row: dict) -> tuple[int, int]:
        xi = bisect.bisect_right(x_edges, row["x"])
        return xi, 2 * xi + int(row["y"] >= y_edges[xi])

    for row in train:
        leaves_train[index(row)[1]].append(row)
    for row in test:
        leaves_test[index(row)[1]].append(row)
    x_predictions = [
        (sum(row["rescue"] for row in group) + 0.5) / (len(group) + 1)
        for group in training_xbins
    ]
    leaves = []
    q_predictions = []
    x_only_predictions = []
    for leaf_index, (training, heldout) in enumerate(zip(leaves_train, leaves_test)):
        n_train, n_test = len(training), len(heldout)
        rescue_train = sum(row["rescue"] for row in training)
        rescue_test = sum(row["rescue"] for row in heldout)
        q_hat = (rescue_train + 0.5) / (n_train + 1)
        leaves.append({
            "index": leaf_index,
            "x_bin": leaf_index // 2,
            "y_half": leaf_index % 2,
            "train_n": n_train, "test_n": n_test,
            "train_rescue": rescue_train, "test_rescue": rescue_test,
            "train_q": q_hat,
            "train_p_neural": (sum(row["neural_success"] for row in training) + 0.5)
                / (n_train + 1),
            "test_q": rescue_test / n_test,
        })
        q_predictions.extend((q_hat, row["rescue"]) for row in heldout)
        x_only_predictions.extend((x_predictions[leaf_index // 2], row["rescue"])
                                  for row in heldout)
    base_rate = sum(row["rescue"] for row in train) / len(train)
    constant_pairs = [(base_rate, row["rescue"]) for row in test]
    mace = sum(row["test_n"] * abs(row["train_q"] - row["test_q"])
               for row in leaves) / len(test)
    train_seeds = {row["channel_seed"] for row in train}
    test_seeds = {row["channel_seed"] for row in test}
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "provenance": (
            manifest["host"] == protocol["node"]
            and str(manifest["slurm_job_id"]) == protocol["job"]
            and manifest["trials"] == protocol["trials_per_snr"]
            and manifest["train_seed"] == protocol["train_seed"]
            and manifest["test_seed"] == protocol["test_seed"]
        ),
        "disjoint": (
            len(train_seeds) == len(train)
            and len(test_seeds) == len(test)
            and not train_seeds.intersection(test_seeds)
        ),
        "sample_size": (
            len(train) == len(test) == protocol["trials_per_snr"] * len(protocol["snrs_db"])
            and all(row["train_n"] >= protocol["min_leaf_n"]
                    and row["test_n"] >= protocol["min_leaf_n"] for row in leaves)
            and sum(row["rescue"] for row in train) >= protocol["min_rescue_events"]
            and sum(row["rescue"] for row in test) >= protocol["min_rescue_events"]
        ),
        "calibration": mace <= protocol["max_weighted_mace"],
        "predictive_value": brier(q_predictions) < brier(constant_pairs),
    }
    per_snr = []
    for snr in protocol["snrs_db"]:
        selected_train = [row for row in train if row["snr_db_evaluation_only"] == snr]
        selected_test = [row for row in test if row["snr_db_evaluation_only"] == snr]
        per_snr.append({"snr_db": snr,
                        "train_rescue": sum(row["rescue"] for row in selected_train),
                        "test_rescue": sum(row["rescue"] for row in selected_test)})
    report = {
        "schema": "softwall-confirm77-mixed-snr-phy-value-v1",
        "job": protocol["job"], "all_pass": all(gates.values()), "gates": gates,
        "model_inputs": ["channel_estimate_power", "received_grid_power"],
        "snr_is_online_input": False,
        "train_only_x_edges": x_edges, "train_only_y_medians": y_edges,
        "leaves": leaves,
        "train_rescue": sum(row["rescue"] for row in train),
        "test_rescue": sum(row["rescue"] for row in test),
        "weighted_mace": mace,
        "brier_2d": brier(q_predictions),
        "brier_x_only": brier(x_only_predictions),
        "brier_train_prevalence_constant": brier(constant_pairs),
        "per_snr_evaluation_only": per_snr,
        "interpretation": "Train-only ten-leaf histogram based on two observable features across four synthetic SNR levels. Held-out evaluation does not use true SNR online. No DU/TDL generalization, q_use timing, policy performance or hard real-time proof.",
    }
    output = base / f"confirm77_multisnr_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [key for key, ok in gates.items() if not ok],
                      "train_rescue": report["train_rescue"],
                      "test_rescue": report["test_rescue"],
                      "weighted_mace": mace,
                      "brier_2d": report["brier_2d"],
                      "brier_x_only": report["brier_x_only"],
                      "brier_constant": report["brier_train_prevalence_constant"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
