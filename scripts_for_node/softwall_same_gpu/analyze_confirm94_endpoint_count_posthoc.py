#!/usr/bin/env python3.11
"""Attribute Confirm94 joint-versus-staged gains to NRx cardinality posthoc."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "results/softwall_same_gpu"
    path = base / "confirm94_independent_phy_screen.json"
    screen = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in screen["records"] if row["status"] == "feasible"]
    groups = {}
    for count in range(3):
        selected = [row for row in rows if len(row["staged_selected"]) == count]
        gains = [row["joint_ai"] - row["staged_ai"] for row in selected]
        groups[str(count)] = {
            "cases": len(selected),
            "joint_ai_gain_cases": sum(value > 1e-9 for value in gains),
            "mean_joint_minus_staged_ai": statistics.mean(gains) if gains else None,
            "joint_uses_two_nrx": sum(len(row["joint_selected"]) == 2
                                      for row in selected),
        }
    report = {
        "schema": "softwall-confirm94-endpoint-count-posthoc-v1",
        "posthoc": True,
        "screen_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feasible": len(rows),
        "joint_uses_two_nrx": sum(len(row["joint_selected"]) == 2 for row in rows),
        "max_radio_uses_two_nrx": sum(len(row["max_radio_selected"]) == 2
                                       for row in rows),
        "joint_equals_max_radio_subset": sum(
            row["joint_selected"] == row["max_radio_selected"] for row in rows),
        "by_staged_nrx_count": groups,
        "interpretation": "The large joint-versus-minimal-NRx AI gain is mostly use of a spare second NeuralRx endpoint, not an advantage over the two-endpoint max-radio baseline. Posthoc diagnostic, not a preregistered performance claim.",
    }
    output = base / "confirm94_endpoint_count_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
