#!/usr/bin/env python3.11
"""Export the Confirm77 observable-feature rescue model from training data only."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

from analyze_confirm77_multisnr_phy_value import samples
from build_confirm74_train_model import wilson


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"


def main() -> None:
    protocol_path = BASE / "confirm77_multisnr_protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    raw_path = BASE / "raw" / f"{protocol['prefix']}_train.json"
    raw_bytes = raw_path.read_bytes()
    training = samples(json.loads(raw_bytes), protocol["train_seed"],
                       protocol["trials_per_snr"], protocol["snrs_db"])
    if len({row["channel_seed"] for row in training}) != len(training):
        raise ValueError("duplicate training channel seed")
    xs = sorted(row["x"] for row in training)
    x_edges = [xs[round(k * len(xs) / protocol["x_bins"])]
               for k in range(1, protocol["x_bins"])]
    x_groups = [[] for _ in range(protocol["x_bins"])]
    for row in training:
        x_groups[bisect.bisect_right(x_edges, row["x"])].append(row)
    y_medians = [sorted(row["y"] for row in group)[len(group) // 2]
                 for group in x_groups]
    leaves = [[] for _ in range(2 * protocol["x_bins"])]
    for row in training:
        x_bin = bisect.bisect_right(x_edges, row["x"])
        leaves[2 * x_bin + int(row["y"] >= y_medians[x_bin])].append(row)
    if any(len(group) < protocol["min_leaf_n"] for group in leaves):
        raise ValueError("training support below frozen minimum")

    model = {
        "schema": "softwall-confirm77-train-only-mixed-snr-phy-rescue-model-v1",
        "source": str(raw_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "train_seed": protocol["train_seed"],
        "train_n": len(training),
        "training_snrs_db": protocol["snrs_db"],
        "noise_reference": "pre_fading",
        "online_features": ["channel_estimate_power", "received_grid_power"],
        "true_snr_is_online_input": False,
        "x_support": [xs[0], xs[-1]],
        "y_support": [min(row["y"] for row in training),
                      max(row["y"] for row in training)],
        "x_upper_edges": x_edges,
        "y_medians_by_x_bin": y_medians,
        "leaves": [],
        "test_data_accessed": False,
        "policy_use_limit": (
            "Only the synthetic independent-Rayleigh/pre-fading-AWGN mixture "
            "with trained support. Wilson limits express sampling uncertainty, "
            "not a hard PHY or timing guarantee. q_use and DU/TDL transfer "
            "remain unqualified. Out-of-support features fail closed."
        ),
    }
    for index, group in enumerate(leaves):
        n = len(group)
        rescue = sum(row["rescue"] for row in group)
        neural = sum(row["neural_success"] for row in group)
        lower, upper = wilson(rescue, n)
        model["leaves"].append({
            "index": index, "x_bin": index // 2, "y_half": index % 2,
            "train_n": n, "rescue_count": rescue,
            "neural_success_count": neural,
            "q_rescue_smoothed": (rescue + 0.5) / (n + 1),
            "q_rescue_wilson95_lower": lower,
            "q_rescue_wilson95_upper": upper,
            "p_neural_success_smoothed": (neural + 0.5) / (n + 1),
        })
    output = BASE / "confirm77_train_only_q_model.json"
    output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train_n": len(training),
                      "rescues": [row["rescue_count"] for row in model["leaves"]],
                      "test_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
