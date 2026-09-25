#!/usr/bin/env python3.11
"""Online lookup of train-only incremental NeuralRx rescue probability.

The model is valid only for the synthetic channel contract recorded in its
artifact. A caller must not silently treat its Wilson interval as a hard
wireless guarantee or reuse it on an unqualified DU/channel distribution.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhyValue:
    bin_index: int
    q_rescue: float
    q_lower95: float
    q_upper95: float
    p_neural_success: float


class TrainOnlyPhyValueModel:
    def __init__(self, model_path: Path, project_root: Path) -> None:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if (model["schema"] != "softwall-confirm74-train-only-phy-rescue-model-v1"
                or model["test_data_accessed"]
                or model["feature"] != "channel_estimate_power"
                or model["noise_reference"] != "pre_fading"):
            raise ValueError("not a qualified train-only PHY-value artifact")
        source = project_root / model["source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != model["source_sha256"]:
            raise ValueError("training trace changed after model export")
        self.support_min = float(model["support_min"])
        self.support_max = float(model["support_max"])
        self.edges = tuple(float(value) for value in model["bin_upper_edges"])
        self.bins = tuple(model["bins"])
        if (len(self.bins) != len(self.edges) + 1
                or tuple(sorted(self.edges)) != self.edges
                or self.support_min >= self.support_max):
            raise ValueError("invalid PHY-value model bins")

    def lookup(self, observed_channel_estimate_power: float) -> PhyValue | None:
        x = observed_channel_estimate_power
        if not math.isfinite(x) or x < self.support_min or x > self.support_max:
            return None
        index = bisect.bisect_right(self.edges, x)
        item = self.bins[index]
        return PhyValue(
            index,
            float(item["q_rescue_smoothed"]),
            float(item["q_rescue_wilson95_lower"]),
            float(item["q_rescue_wilson95_upper"]),
            float(item["p_neural_success_smoothed"]),
        )
