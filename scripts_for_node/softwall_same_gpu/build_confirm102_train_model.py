#!/usr/bin/env python3.11
"""Select a finer observable PHY model using Confirm77 training data only."""

from __future__ import annotations

import bisect
import hashlib
import json
import statistics
from pathlib import Path

from analyze_confirm77_multisnr_phy_value import samples
from build_confirm74_train_model import wilson


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"
CANDIDATES = ((5, 2), (5, 4), (5, 5), (8, 4), (8, 5),
              (10, 4), (10, 5), (16, 2), (16, 4),
              (20, 2), (20, 4), (20, 5))
FOLDS = 5
MIN_CV_LEAF = 100


def quantile_edges(values: list[float], groups: int) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, round(k * len(ordered) / groups))]
            for k in range(1, groups)]


def fit(rows: list[dict], x_bins: int, y_bins: int) -> dict:
    x_edges = quantile_edges([row["x"] for row in rows], x_bins)
    x_groups = [[] for _ in range(x_bins)]
    for row in rows:
        x_groups[bisect.bisect_right(x_edges, row["x"])].append(row)
    y_edges = [quantile_edges([row["y"] for row in group], y_bins)
               for group in x_groups]
    leaves = [[] for _ in range(x_bins * y_bins)]
    for row in rows:
        x_bin = bisect.bisect_right(x_edges, row["x"])
        y_bin = bisect.bisect_right(y_edges[x_bin], row["y"])
        leaves[x_bin * y_bins + y_bin].append(row)
    return {"x_edges": x_edges, "y_edges": y_edges, "leaves": leaves,
            "x_bins": x_bins, "y_bins": y_bins}


def leaf_index(model: dict, row: dict) -> int:
    x_bin = bisect.bisect_right(model["x_edges"], row["x"])
    y_bin = bisect.bisect_right(model["y_edges"][x_bin], row["y"])
    return x_bin * model["y_bins"] + y_bin


def probabilities(model: dict) -> list[float]:
    return [(sum(row["rescue"] for row in group) + 0.5) / (len(group) + 1)
            for group in model["leaves"]]


def main() -> None:
    protocol_path = BASE / "confirm77_multisnr_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    raw_path = BASE / "raw" / f"{protocol['prefix']}_train.json"
    raw_bytes = raw_path.read_bytes()
    training = samples(json.loads(raw_bytes), protocol["train_seed"],
                       protocol["trials_per_snr"], protocol["snrs_db"])
    candidates = []
    for x_bins, y_bins in CANDIDATES:
        fold_brier = []
        fold_min_leaf = []
        for fold in range(FOLDS):
            train = [row for row in training if row["channel_seed"] % FOLDS != fold]
            validation = [row for row in training if row["channel_seed"] % FOLDS == fold]
            model = fit(train, x_bins, y_bins)
            probs = probabilities(model)
            fold_min_leaf.append(min(len(group) for group in model["leaves"]))
            fold_brier.append(statistics.mean(
                (row["rescue"] - probs[leaf_index(model, row)]) ** 2
                for row in validation
            ))
        candidates.append({
            "x_bins": x_bins, "y_bins": y_bins,
            "fold_brier": fold_brier,
            "mean_cv_brier": statistics.mean(fold_brier),
            "stdev_cv_brier": statistics.stdev(fold_brier),
            "minimum_fold_leaf_n": min(fold_min_leaf),
            "eligible": min(fold_min_leaf) >= MIN_CV_LEAF,
        })
    eligible = [row for row in candidates if row["eligible"]]
    chosen = min(eligible, key=lambda row: (
        row["mean_cv_brier"], row["x_bins"] * row["y_bins"]
    ))
    fitted = fit(training, chosen["x_bins"], chosen["y_bins"])
    xs = [row["x"] for row in training]
    ys = [row["y"] for row in training]
    exported = {
        "schema": "softwall-confirm102-train-only-cv-grid-phy-model-v1",
        "source": str(raw_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_source": "Confirm77 training split only",
        "test_or_confirm102_records_accessed": False,
        "features": ["channel_estimate_power", "received_grid_power"],
        "fold_assignment": "channel_seed modulo 5",
        "selection_rule": (
            "minimum mean five-fold rescue Brier among the fixed grid; "
            "require at least 100 fitting samples in every leaf of every fold"
        ),
        "candidate_grid": candidates,
        "selected_x_bins": chosen["x_bins"],
        "selected_y_bins": chosen["y_bins"],
        "x_support": [min(xs), max(xs)],
        "y_support": [min(ys), max(ys)],
        "x_upper_edges": fitted["x_edges"],
        "y_upper_edges_by_x_bin": fitted["y_edges"],
        "leaves": [],
        "policy_limit": (
            "Synthetic independent Rayleigh/pre-fading-AWGN mixture only. "
            "Confirm102 must independently evaluate calibration before this "
            "model supports a policy result. Wilson limits are sampling "
            "uncertainty, not a hard radio guarantee."
        ),
    }
    for index, group in enumerate(fitted["leaves"]):
        rescue = sum(row["rescue"] for row in group)
        neural = sum(row["neural_success"] for row in group)
        q_lower, q_upper = wilson(rescue, len(group))
        exported["leaves"].append({
            "index": index,
            "x_bin": index // fitted["y_bins"],
            "y_bin": index % fitted["y_bins"],
            "train_n": len(group),
            "rescue_count": rescue,
            "neural_success_count": neural,
            "q_rescue_smoothed": (rescue + 0.5) / (len(group) + 1),
            "q_rescue_wilson95_lower": q_lower,
            "q_rescue_wilson95_upper": q_upper,
            "p_neural_success_smoothed": (neural + 0.5) / (len(group) + 1),
        })
    output = BASE / "confirm102_train_only_cv_grid_model.json"
    output.write_text(json.dumps(exported, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "selected": [chosen["x_bins"], chosen["y_bins"]],
        "mean_cv_brier": chosen["mean_cv_brier"],
        "minimum_fold_leaf_n": chosen["minimum_fold_leaf_n"],
        "full_minimum_leaf_n": min(len(group) for group in fitted["leaves"]),
        "test_or_confirm102_records_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
