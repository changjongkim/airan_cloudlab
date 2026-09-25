#!/usr/bin/env python3.11
"""Export an online-eligible PHY rescue model from Confirm74 train only."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> None:
    protocol = json.loads((BASE / "confirm74_phy_value_protocol.json").read_text())
    raw_path = BASE / "raw" / f"{protocol['prefix']}_train.json"
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    assert raw["seed"] == protocol["train_seed"]
    assert raw["noise_reference"] == "pre_fading"
    assert raw["observed_features_recorded"] and raw["observed_features_prewarmed"]
    samples = raw["results"][0]["records"]
    assert len(samples) == protocol["trials"]
    features = sorted(
        row["observed_features"]["channel_estimate_power"] for row in samples
    )
    edges = [features[round(k * len(features) / protocol["bins"])]
             for k in range(1, protocol["bins"])]
    groups = [[] for _ in range(protocol["bins"])]
    for index, row in enumerate(samples):
        assert row["trial"] == index
        assert row["channel_seed"] == protocol["train_seed"] + index
        groups[bisect.bisect_right(
            edges, row["observed_features"]["channel_estimate_power"]
        )].append(row)
    model = {
        "schema": "softwall-confirm74-train-only-phy-rescue-model-v1",
        "source": str(raw_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "feature": "channel_estimate_power",
        "train_seed": protocol["train_seed"],
        "train_n": protocol["trials"],
        "noise_reference": "pre_fading",
        "snr_db": -8.5,
        "support_min": features[0], "support_max": features[-1],
        "bin_upper_edges": edges,
        "bins": [],
        "test_data_accessed": False,
        "policy_use_limit": "Only synthetic -8.5 dB fixed-pre-fading-noise diagnostic. Use q upper bound to charge skipped NeuralRx radio-risk budget; use q lower bound when valuing speculative NRx. Out-of-support features must fail closed until independently qualified. This model does not include late-result q_use.",
    }
    for index, group in enumerate(groups):
        n = len(group)
        rescue = sum(row["neural_correct"] and not row["conventional_correct"]
                     for row in group)
        neural = sum(row["neural_correct"] for row in group)
        lower, upper = wilson(rescue, n)
        model["bins"].append({
            "index": index, "train_n": n,
            "rescue_count": rescue, "neural_success_count": neural,
            "q_rescue_smoothed": (rescue + 0.5) / (n + 1),
            "q_rescue_wilson95_lower": lower,
            "q_rescue_wilson95_upper": upper,
            "p_neural_success_smoothed": (neural + 0.5) / (n + 1),
        })
    output = BASE / "confirm74_train_only_q_model.json"
    output.write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps({"output": str(output),
                      "rescue": [row["rescue_count"] for row in model["bins"]],
                      "test_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
