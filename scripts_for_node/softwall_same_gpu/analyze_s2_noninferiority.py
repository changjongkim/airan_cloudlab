#!/usr/bin/env python3
"""Evaluate the frozen paired noninferiority and timing gates for S2."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-pp", type=float, default=0.25)
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_(?P<policy>eager_dual|s2_reserved)"
        + r"_job(?P<job>[0-9]+)_ran[.]json$"
    )
    groups: dict[int, dict[str, dict]] = {}
    sources: dict[int, dict[str, str]] = {}
    for path in sorted(args.raw.glob(f"{args.campaign}_r*_ran.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        round_index = int(match["round"])
        policy = match["policy"]
        if policy in groups.setdefault(round_index, {}):
            raise SystemExit(f"duplicate r{round_index} {policy}")
        groups[round_index][policy] = json.loads(path.read_text(encoding="utf-8"))
        sources.setdefault(round_index, {})[policy] = path.name

    rows = []
    differences = []
    s2_wins = []
    eager_wins = []
    for round_index in sorted(groups):
        group = groups[round_index]
        if set(group) != {"eager_dual", "s2_reserved"}:
            continue
        eager = group["eager_dual"]
        s2 = group["s2_reserved"]
        fields = (
            "iterations", "period_ms", "deadline_ms", "snr_db",
            "payload_seed", "channel_seed_base",
        )
        for field in fields:
            if eager.get(field) != s2.get(field):
                raise SystemExit(f"unpaired {field} in round {round_index}")
        count = eager["iterations"]
        if len(eager["records"]) != count or len(s2["records"]) != count:
            raise SystemExit(f"record count mismatch in round {round_index}")
        round_differences = []
        round_s2_wins = 0
        round_eager_wins = 0
        seed_mismatches = 0
        nrx_decision_mismatches = 0
        for index, (eager_record, s2_record) in enumerate(
            zip(eager["records"], s2["records"])
        ):
            if (
                eager_record.get("channel_seed")
                != s2_record.get("channel_seed")
            ):
                seed_mismatches += 1
            eager_correct = int(bool(eager_record["correct"]))
            s2_correct = int(bool(s2_record["correct"]))
            difference = s2_correct - eager_correct
            round_differences.append(difference)
            differences.append(difference)
            if difference > 0:
                round_s2_wins += 1
                s2_wins.append((round_index, index))
            elif difference < 0:
                round_eager_wins += 1
                eager_wins.append((round_index, index))
            eager_nrx = eager_record.get("nrx_branch_correct")
            s2_nrx = s2_record.get("nrx_branch_correct")
            if eager_nrx is not None and s2_nrx is not None and eager_nrx != s2_nrx:
                nrx_decision_mismatches += 1
        duration_s = count * s2["period_ms"] / 1000.0
        rows.append({
            "round": round_index,
            "iterations": count,
            "eager_correct": eager["correct_releases"],
            "s2_correct": s2["correct_releases"],
            "difference_pp": statistics.mean(round_differences) * 100.0,
            "s2_wins": round_s2_wins,
            "eager_wins": round_eager_wins,
            "channel_seed_mismatches": seed_mismatches,
            "nrx_decision_mismatches": nrx_decision_mismatches,
            "s2_deadline_misses": s2["deadline_misses"],
            "s2_duplicate_commits": s2["duplicate_commits"],
            "s2_admission_rejections": s2["admission_rejections"],
            "s2_bound_violations": (
                s2["nrx_bound_violations"] + s2["conv_bound_violations"]
            ),
            "s2_ai_violations": (
                s2["background_budget_violations"]
                + s2["background_release_crossings"]
            ),
            "eager_ai_violations": (
                eager["background_budget_violations"]
                + eager["background_release_crossings"]
            ),
            "eager_background_per_s": eager["background_units"] / duration_s,
            "s2_background_per_s": s2["background_units"] / duration_s,
            "sources": sources[round_index],
        })
    if not rows:
        raise SystemExit("no complete paired rounds")

    count = len(differences)
    mean = statistics.mean(differences)
    variance = statistics.variance(differences) if count > 1 else 0.0
    standard_error = math.sqrt(variance / count)
    lower = (mean - 1.96 * standard_error) * 100.0
    upper = (mean + 1.96 * standard_error) * 100.0
    difference_pp = mean * 100.0
    s2_background = statistics.mean(row["s2_background_per_s"] for row in rows)
    eager_background = statistics.mean(
        row["eager_background_per_s"] for row in rows
    )
    valid_background_rows = [
        row for row in rows
        if row["s2_ai_violations"] == 0 and row["eager_ai_violations"] == 0
    ]
    valid_background = None
    if valid_background_rows:
        valid_s2 = statistics.mean(
            row["s2_background_per_s"] for row in valid_background_rows
        )
        valid_eager = statistics.mean(
            row["eager_background_per_s"] for row in valid_background_rows
        )
        valid_background = {
            "pairs": len(valid_background_rows),
            "s2": valid_s2,
            "eager": valid_eager,
            "relative_gain_percent": (valid_s2 / valid_eager - 1) * 100.0,
        }
    gates = {
        "utility_noninferior": lower > -args.margin_pp,
        "deadline_misses_zero": sum(row["s2_deadline_misses"] for row in rows) == 0,
        "duplicate_commits_zero": sum(row["s2_duplicate_commits"] for row in rows) == 0,
        "admission_rejections_zero": sum(row["s2_admission_rejections"] for row in rows) == 0,
        "bound_violations_zero": sum(row["s2_bound_violations"] for row in rows) == 0,
        "ai_violations_zero": sum(
            row["s2_ai_violations"] + row["eager_ai_violations"]
            for row in rows
        ) == 0,
        "channel_seeds_match": sum(row["channel_seed_mismatches"] for row in rows) == 0,
        "background_throughput_gain": s2_background > eager_background,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema": "softwall-s2-noninferiority-v1",
        "campaign": args.campaign,
        "paired_releases": count,
        "margin_percentage_points": args.margin_pp,
        "s2_minus_eager_percentage_points": difference_pp,
        "confidence_interval_95_percentage_points": [lower, upper],
        "s2_wins": len(s2_wins),
        "eager_wins": len(eager_wins),
        "s2_win_indices": s2_wins,
        "eager_win_indices": eager_wins,
        "mean_background_per_s": {
            "s2": s2_background,
            "eager": eager_background,
            "relative_gain_percent": (s2_background / eager_background - 1) * 100.0,
        },
        "valid_background_pairs": sum(
            row["s2_ai_violations"] == 0 and row["eager_ai_violations"] == 0
            for row in rows
        ),
        "valid_pair_background_per_s": valid_background,
        "gates": gates,
        "rounds": rows,
    }
    lines = [
        "# SoftWall S2 natural-channel noninferiority",
        "",
        "| round | releases | eager correct | S2 correct | S2−eager (pp) | S2/eager wins | seed mismatch | NRx decision mismatch | S2 misses | S2 bound violations | AI violations S/E | S2 bg/s | eager bg/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['iterations']} | {row['eager_correct']} | "
            f"{row['s2_correct']} | {row['difference_pp']:.3f} | "
            f"{row['s2_wins']}/{row['eager_wins']} | "
            f"{row['channel_seed_mismatches']} | {row['nrx_decision_mismatches']} | "
            f"{row['s2_deadline_misses']} | {row['s2_bound_violations']} | "
            f"{row['s2_ai_violations']}/{row['eager_ai_violations']} | "
            f"{row['s2_background_per_s']:.1f} | {row['eager_background_per_s']:.1f} |"
        )
    lines.extend([
        "",
        f"Paired releases: {count}; S2−eager: {difference_pp:.4f} percentage points; 95% CI [{lower:.4f}, {upper:.4f}].",
        f"Noninferiority margin: -{args.margin_pp:.3f} percentage points; pass: {gates['utility_noninferior']}.",
        f"Background: S2 {s2_background:.1f}/s, eager {eager_background:.1f}/s, gain {(s2_background / eager_background - 1) * 100:.2f}%.",
        (
            f"Valid-pair background ({valid_background['pairs']} rounds): "
            f"S2 {valid_background['s2']:.1f}/s, eager {valid_background['eager']:.1f}/s, "
            f"gain {valid_background['relative_gain_percent']:.2f}%."
            if valid_background is not None
            else "Valid-pair background: no rounds without AI violations."
        ),
        f"All frozen gates pass: {gates['all_pass']}.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
