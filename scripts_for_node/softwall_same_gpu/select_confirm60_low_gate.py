#!/usr/bin/env python3.11
"""Select a low-feature hopeless-request threshold from training data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.train.read_bytes()
    data = json.loads(source)
    if data["seed"] != 20456001 or data["noise_reference"] != "pre_fading":
        raise ValueError("unexpected training trace")
    rows = data["results"][0]["records"]
    if len(rows) != 500:
        raise ValueError("incomplete training trace")
    ordered = sorted(rows, key=lambda row: row["observed_features"]["channel_estimate_power"])
    first_decodable = next(row for row in ordered if row["neural_correct"] or row["conventional_correct"])
    threshold = first_decodable["observed_features"]["channel_estimate_power"]
    skipped = [row for row in rows if row["observed_features"]["channel_estimate_power"] < threshold]
    result = {
        "schema": "softwall-confirm60-low-gate-selection-v1",
        "training_sha256": hashlib.sha256(source).hexdigest(),
        "selection": "minimum observable channel power among training TBs decoded by either branch; skip only strictly below",
        "threshold": threshold,
        "training_count": len(rows),
        "training_skipped": len(skipped),
        "training_neural_success_lost": sum(row["neural_correct"] for row in skipped),
        "training_conventional_success_lost": sum(row["conventional_correct"] for row in skipped),
        "warning": "Training zero loss is not a held-out or worst-case guarantee; low-SNR undecodable TBs may be a synthetic MCS/link-adaptation artifact."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
