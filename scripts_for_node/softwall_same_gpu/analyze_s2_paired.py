#!/usr/bin/env python3
"""Compare S2 policies release by release on shared channel-seed traces."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


POLICIES = ("conventional_only", "nrx_only", "eager_dual", "s2_reserved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_(?P<policy>"
        + "|".join(POLICIES)
        + r")_job(?P<job>[0-9]+)_ran[.]json$"
    )
    rounds: dict[int, dict[str, dict]] = {}
    sources: dict[int, dict[str, str]] = {}
    for path in sorted(args.raw.glob(f"{args.campaign}_r*_ran.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        round_index = int(match["round"])
        policy = match["policy"]
        if policy in rounds.setdefault(round_index, {}):
            raise SystemExit(f"duplicate round/policy result: r{round_index} {policy}")
        rounds[round_index][policy] = json.loads(path.read_text(encoding="utf-8"))
        sources.setdefault(round_index, {})[policy] = path.name

    rows = []
    for round_index in sorted(rounds):
        group = rounds[round_index]
        missing = set(POLICIES) - set(group)
        if missing:
            continue
        iterations = {value["iterations"] for value in group.values()}
        seeds = {value.get("channel_seed_base") for value in group.values()}
        if len(iterations) != 1 or len(seeds) != 1:
            raise SystemExit(f"unpaired configuration in round {round_index}")
        count = iterations.pop()
        records = {key: value["records"] for key, value in group.items()}
        if any(len(items) != count for items in records.values()):
            raise SystemExit(f"record count mismatch in round {round_index}")

        mismatch_indices = []
        s2_gains = []
        s2_losses = []
        independent_union = 0
        for index in range(count):
            conventional = bool(records["conventional_only"][index]["correct"])
            nrx = bool(records["nrx_only"][index]["correct"])
            eager = bool(records["eager_dual"][index]["correct"])
            s2 = bool(records["s2_reserved"][index]["correct"])
            independent_union += int(conventional or nrx)
            if eager != s2:
                mismatch_indices.append(index)
            if s2 and not nrx:
                s2_gains.append(index)
            if eager and not s2:
                s2_losses.append(index)
        duration_s = count * group["s2_reserved"]["period_ms"] / 1000.0
        rows.append({
            "round": round_index,
            "iterations": count,
            "channel_seed_base": next(iter(seeds)),
            "correct": {
                policy: int(group[policy]["correct_releases"])
                for policy in POLICIES
            },
            "independent_policy_union_correct": independent_union,
            "s2_eager_mismatch_count": len(mismatch_indices),
            "s2_eager_mismatch_indices": mismatch_indices,
            "s2_gains_over_nrx_only": len(s2_gains),
            "s2_gain_indices": s2_gains,
            "s2_losses_vs_eager": len(s2_losses),
            "s2_loss_indices": s2_losses,
            "s2_deadline_misses": group["s2_reserved"]["deadline_misses"],
            "s2_duplicate_commits": group["s2_reserved"]["duplicate_commits"],
            "s2_admission_rejections": group["s2_reserved"]["admission_rejections"],
            "s2_bound_violations": (
                group["s2_reserved"]["nrx_bound_violations"]
                + group["s2_reserved"]["conv_bound_violations"]
            ),
            "s2_background_per_s": group["s2_reserved"]["background_units"] / duration_s,
            "eager_background_per_s": group["eager_dual"]["background_units"] / duration_s,
            "s2_background_violations": (
                group["s2_reserved"]["background_budget_violations"]
                + group["s2_reserved"]["background_release_crossings"]
            ),
            "sources": sources[round_index],
        })
    if not rows:
        raise SystemExit("no complete paired rounds found")

    total = sum(row["iterations"] for row in rows)
    mismatches = sum(row["s2_eager_mismatch_count"] for row in rows)
    gains = sum(row["s2_gains_over_nrx_only"] for row in rows)
    losses = sum(row["s2_losses_vs_eager"] for row in rows)
    s2_bg = statistics.mean(row["s2_background_per_s"] for row in rows)
    eager_bg = statistics.mean(row["eager_background_per_s"] for row in rows)
    lines = [
        "# SoftWall S2 natural-channel paired analysis",
        "",
        "| round | releases | conventional | NRx-only | eager | S2 | independent union | S2/eager mismatches | S2 gains over NRx | S2 losses vs eager | S2 misses | S2 bound violations | S2 bg/s | eager bg/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        correct = row["correct"]
        lines.append(
            f"| {row['round']} | {row['iterations']} | "
            f"{correct['conventional_only']} | {correct['nrx_only']} | "
            f"{correct['eager_dual']} | {correct['s2_reserved']} | "
            f"{row['independent_policy_union_correct']} | "
            f"{row['s2_eager_mismatch_count']} | "
            f"{row['s2_gains_over_nrx_only']} | {row['s2_losses_vs_eager']} | "
            f"{row['s2_deadline_misses']} | {row['s2_bound_violations']} | "
            f"{row['s2_background_per_s']:.1f} | {row['eager_background_per_s']:.1f} |"
        )
    lines.extend([
        "",
        f"Complete paired rounds: {len(rows)}; releases: {total}.",
        f"S2/eager correctness mismatches: {mismatches}; S2 gains over NRx-only: {gains}; S2 losses versus eager: {losses}.",
        f"Mean background throughput: S2 {s2_bg:.1f}/s, eager {eager_bg:.1f}/s ({(s2_bg / eager_bg - 1) * 100:.2f}% delta).",
        "",
        "`independent union` combines outcomes from separate policy processes and is diagnostic at the decoder transition; eager is the actual both-branch execution result.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
