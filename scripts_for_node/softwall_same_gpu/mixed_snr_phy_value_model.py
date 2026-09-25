#!/usr/bin/env python3.11
"""Fail-closed lookup for the train-only Confirm77 two-feature PHY model."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MixedSnrPhyValue:
    leaf_index: int
    q_rescue: float
    q_lower95: float
    q_upper95: float
    p_neural_success: float


class TrainOnlyMixedSnrPhyValueModel:
    def __init__(self, model_path: Path, project_root: Path) -> None:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if (model["schema"] != "softwall-confirm77-train-only-mixed-snr-phy-rescue-model-v1"
                or model["test_data_accessed"]
                or model["noise_reference"] != "pre_fading"
                or model["online_features"] !=
                    ["channel_estimate_power", "received_grid_power"]
                or model["true_snr_is_online_input"]):
            raise ValueError("not a qualified train-only mixed-SNR model")
        for path_key, sha_key in (("source", "source_sha256"),
                                  ("protocol", "protocol_sha256")):
            if hashlib.sha256((project_root / model[path_key]).read_bytes()).hexdigest() != model[sha_key]:
                raise ValueError(f"{path_key} changed after model export")
        self.x_support = tuple(float(x) for x in model["x_support"])
        self.y_support = tuple(float(y) for y in model["y_support"])
        self.x_edges = tuple(float(x) for x in model["x_upper_edges"])
        self.y_medians = tuple(float(y) for y in model["y_medians_by_x_bin"])
        self.leaves = tuple(model["leaves"])
        if (len(self.x_edges) + 1 != len(self.y_medians)
                or len(self.leaves) != 2 * len(self.y_medians)
                or tuple(sorted(self.x_edges)) != self.x_edges
                or self.x_support[0] >= self.x_support[1]
                or self.y_support[0] >= self.y_support[1]
                or any(row["index"] != i for i, row in enumerate(self.leaves))):
            raise ValueError("invalid mixed-SNR model bins")

    def lookup(self, channel_estimate_power: float,
               received_grid_power: float) -> MixedSnrPhyValue | None:
        x, y = channel_estimate_power, received_grid_power
        if (not math.isfinite(x) or not math.isfinite(y)
                or not self.x_support[0] <= x <= self.x_support[1]
                or not self.y_support[0] <= y <= self.y_support[1]):
            return None
        x_bin = bisect.bisect_right(self.x_edges, x)
        leaf_index = 2 * x_bin + int(y >= self.y_medians[x_bin])
        row = self.leaves[leaf_index]
        return MixedSnrPhyValue(
            leaf_index, float(row["q_rescue_smoothed"]),
            float(row["q_rescue_wilson95_lower"]),
            float(row["q_rescue_wilson95_upper"]),
            float(row["p_neural_success_smoothed"]),
        )
