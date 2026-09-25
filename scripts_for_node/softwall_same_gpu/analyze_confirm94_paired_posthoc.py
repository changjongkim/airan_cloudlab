#!/usr/bin/env python3.11
"""Posthoc paired outcome and uncertainty audit of frozen Confirm94 decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values: list[float], seed: int) -> dict:
    if not values:
        return {"mean": None, "bootstrap_95pct": None}
    rng = random.Random(seed)
    size = len(values)
    boot = sorted(
        statistics.mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(10000)
    )
    return {
        "mean": statistics.mean(values),
        "bootstrap_95pct": [boot[249], boot[9749]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "results/softwall_same_gpu"
    protocol_path = base / "confirm94_independent_phy_protocol.json"
    screen_path = base / "confirm94_independent_phy_screen.json"
    protocol, screen = read(protocol_path), read(screen_path)
    trace_path = base / "raw" / f"{protocol['trace_prefix']}_test.json"
    raw = read(trace_path)
    if screen["trace_sha256"] != hashlib.sha256(trace_path.read_bytes()).hexdigest():
        raise RuntimeError("frozen-screen input trace changed")
    if screen["source_hashes"] is not True or len(screen["records"]) != protocol["cases"]:
        raise RuntimeError("frozen-screen provenance mismatch")
    pool = [row for result in raw["results"] for row in result["records"]]
    rng = random.Random(protocol["sampling_seed"])
    rows = []
    for case, record in enumerate(screen["records"]):
        chosen = rng.sample(pool, protocol["cells"])
        radio_floor = rng.choice(protocol["radio_floor_choices"])
        if record["case"] != case:
            raise RuntimeError("case order changed")
        if record["status"] != "feasible":
            continue
        if record["radio_floor"] != radio_floor:
            raise RuntimeError("sampling or radio floor mismatch")
        def timely_ai(selection: list[str]) -> int:
            # Confirm93's one-event model permits all seven only when a
            # selected NRx succeeds; otherwise six are all-fail feasible.
            selected = set(selection)
            return 6 + int(any(
                chr(ord("a") + i) in selected and cell["neural_correct"]
                for i, cell in enumerate(chosen)
            ))
        joint = timely_ai(record["joint_selected"])
        staged = timely_ai(record["staged_selected"])
        max_radio = timely_ai(record["max_radio_selected"])
        rows.append({
            "case": case,
            "actual_ai_joint": joint,
            "actual_ai_staged": staged,
            "actual_ai_max_radio": max_radio,
            "actual_ai_joint_minus_staged": joint - staged,
            "actual_correct_joint_minus_staged":
                record["joint_actual_correct"] - record["staged_actual_correct"],
            "actual_correct_joint_minus_max_radio":
                record["joint_actual_correct"] - record["max_radio_actual_correct"],
            "expected_ai_joint_minus_staged": record["joint_ai"] - record["staged_ai"],
            "expected_radio_joint_minus_staged":
                record["joint_radio_gain"] - record["staged_radio_gain"],
            "expected_radio_joint_minus_max_radio":
                record["joint_radio_gain"] - record["max_radio_gain"],
        })
    if len(rows) != screen["feasible"]:
        raise RuntimeError("feasible case count changed")
    fields = (
        "actual_ai_joint_minus_staged", "actual_correct_joint_minus_staged",
        "actual_correct_joint_minus_max_radio", "expected_ai_joint_minus_staged",
        "expected_radio_joint_minus_staged", "expected_radio_joint_minus_max_radio",
    )
    report = {
        "schema": "softwall-confirm94-paired-outcomes-posthoc-v1",
        "posthoc": True,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "screen_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
        "trace_sha256": screen["trace_sha256"],
        "feasible_cases": len(rows),
        "paired": {
            field: {
                **interval([float(row[field]) for row in rows], 940000 + i),
                "sum": sum(row[field] for row in rows),
                "positive_cases": sum(row[field] > 0 for row in rows),
                "negative_cases": sum(row[field] < 0 for row in rows),
            }
            for i, field in enumerate(fields)
        },
        "actual_ai_totals": {
            key: sum(row[f"actual_ai_{key}"] for row in rows)
            for key in ("joint", "staged", "max_radio")
        },
        "rows": rows,
        "interpretation": "Posthoc descriptive analysis of a prospectively fixed CPU screen. Seven-unit timely-AI counts are conditional one-event model outcomes reconstructed from held-out neural CRC labels, not measured GPU AI completions. Case bootstrap conditions on the generated 1000-row PHY pool and does not prove production radio noninferiority or hard deadlines.",
    }
    output = base / "confirm94_paired_outcomes_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "feasible_cases": len(rows),
        "actual_ai_totals": report["actual_ai_totals"],
        "paired": report["paired"],
    }, indent=2))


if __name__ == "__main__":
    main()
