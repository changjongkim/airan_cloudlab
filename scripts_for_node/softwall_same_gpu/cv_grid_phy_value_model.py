#!/usr/bin/env python3.11
"""Fail-closed loader for the Confirm102 train-only CV grid model."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CvGridPhyValue:
    leaf_index: int
    q_rescue: float
    q_lower95: float
    q_upper95: float
    p_neural_success: float


class TrainOnlyCvGridPhyValueModel:
    def __init__(self, model_path: Path, project_root: Path) -> None:
        model = json.loads(model_path.read_text())
        if (model["schema"] != "softwall-confirm102-train-only-cv-grid-phy-model-v1"
                or model["test_or_confirm102_records_accessed"]
                or model["features"] !=
                    ["channel_estimate_power", "received_grid_power"]):
            raise ValueError("not a train-only Confirm102 grid model")
        source = project_root / model["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != model["source_sha256"]:
            raise ValueError("training source changed after export")
        self.x_support = tuple(float(value) for value in model["x_support"])
        self.y_support = tuple(float(value) for value in model["y_support"])
        self.x_edges = tuple(float(value) for value in model["x_upper_edges"])
        self.y_edges = tuple(tuple(float(value) for value in row)
                             for row in model["y_upper_edges_by_x_bin"])
        self.x_bins = int(model["selected_x_bins"])
        self.y_bins = int(model["selected_y_bins"])
        self.leaves = tuple(model["leaves"])
        if (len(self.x_edges) + 1 != self.x_bins
                or len(self.y_edges) != self.x_bins
                or any(len(row) + 1 != self.y_bins for row in self.y_edges)
                or len(self.leaves) != self.x_bins * self.y_bins):
            raise ValueError("invalid grid shape")

    def lookup(self, channel_estimate_power: float,
               received_grid_power: float) -> CvGridPhyValue | None:
        x, y = channel_estimate_power, received_grid_power
        if (not math.isfinite(x) or not math.isfinite(y)
                or not self.x_support[0] <= x <= self.x_support[1]
                or not self.y_support[0] <= y <= self.y_support[1]):
            return None
        x_bin = bisect.bisect_right(self.x_edges, x)
        y_bin = bisect.bisect_right(self.y_edges[x_bin], y)
        index = x_bin * self.y_bins + y_bin
        row = self.leaves[index]
        return CvGridPhyValue(
            index, float(row["q_rescue_smoothed"]),
            float(row["q_rescue_wilson95_lower"]),
            float(row["q_rescue_wilson95_upper"]),
            float(row["p_neural_success_smoothed"]),
        )
