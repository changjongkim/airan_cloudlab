#!/usr/bin/env python3
"""Post-hoc, record-level audit of Confirm51 serial fallback blocking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PERIODS = (90, 45, 25, 12)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    details = []
    errors = []
    for period in PERIODS:
        eager = load(args.raw / (
            f"confirm51_r1_p{period}_eager_dual_job{args.job}"
            f"_r1_eager_dual_job{args.job}_ran.json"
        ))
        external = load(args.raw / (
            f"confirm51_r1_p{period}_external_job{args.job}"
            f"_cap80_job{args.job}_controller.json"
        ))
        er = eager["records"]
        xr = external["records"]
        if not (
            len(er) == len(xr) == 1000
            and eager["slurm_job_id"] == external["slurm_job_id"] == args.job
            and eager["host"] == external["host"]
            and eager["period_ms"] == external["period_ms"] == period
            and eager["deadline_ms"] == external["deadline_ms"] == 80
            and eager["payload_seed"] == external["payload_seed"]
            and eager["channel_seed_base"] == external["channel_seed_base"]
        ):
            errors.append(f"P{period}: matched-run input or provenance mismatch")
        if any(a["index"] != b["index"] or a["channel_seed"] != b["channel_seed"]
               for a, b in zip(er, xr)):
            errors.append(f"P{period}: per-record channel pairing mismatch")
        rejected = [i for i, item in enumerate(xr) if item["admission_rejected"]]
        prior_fallback = sum(i > 0 and xr[i - 1]["fallback"] for i in rejected)
        chain_lengths = []
        for i, item in enumerate(xr):
            if item["fallback"]:
                length = 0
                while i + 1 + length < len(xr) and xr[i + 1 + length]["admission_rejected"]:
                    length += 1
                chain_lengths.append(length)
        chain_counts = Counter(chain_lengths)
        chain_text = ", ".join(
            f"{length}×{count}" for length, count in sorted(chain_counts.items())
        )
        eager_nrx_correct = sum(bool(er[i]["nrx_branch_correct"][0]) for i in rejected)
        eager_correct = sum(bool(er[i]["correct"]) for i in rejected)
        external_correct = sum(bool(xr[i]["correct"]) for i in rejected)
        ai_ms = [item["execution_ms"] for item in external["background_records"]]
        external_nrx_ms = [
            item["front_gpu_ms"] + item["post_gpu_ms"]
            for item in xr if item["commit_kind"] == "nrx"
        ]
        if len(rejected) != external["admission_rejections"]:
            errors.append(f"P{period}: rejection summary mismatch")
        if len(ai_ms) != external["background_units"]:
            errors.append(f"P{period}: AI summary mismatch")
        rows.append(
            f"| {period} | {len(rejected)} | {prior_fallback} | "
            f"{eager_nrx_correct} | {eager_correct} | {external_correct} | "
            f"{external['correct_releases']} | {eager['correct_releases']} | "
            f"{external['background_units']} | {chain_text} |"
        )
        details.append(
            f"- P{period}: external NRx-committed front+post GPU p99 "
            f"{nearest_quantile(external_nrx_ms, .99):.3f} ms; "
            + (f"AI RPC n={len(ai_ms)}, p99={nearest_quantile(ai_ms, .99):.3f} ms, "
               f"max={max(ai_ms):.3f} ms."
               if ai_ms else "AI RPC n=0.")
        )
    lines = [
        "# Confirm51: serial fallback blocking mechanism audit",
        "",
        "Post-hoc, evidence grade C. The comparisons pair channel seed and release index, "
        "but the policies ran in separate time windows and receiver outcomes can vary.",
        "",
        "| period ms | external NRx rejects | rejects immediately after fallback | eager NRx correct on rejected input | eager TB correct there | external TB correct there | external TB correct total | eager TB correct total | external AI units | consecutive rejects per fallback: length×count |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        *details,
        "",
        "At P45, all 25 rejects immediately follow a fallback. The controller "
        "waits until release+53 ms for reserved recovery and then runs conventional, "
        "so the next P45 release can start more than 8 ms late even before "
        "conventional execution. However, the next release arrives at +45 ms: "
        "only 8 ms remains before the previous fallback begins, and after its "
        "25 ms reserved interval ends at +78 ms, only 20 ms remains before "
        "the next request's +98 ms NRx cutoff. Both windows are shorter than "
        "the declared 50 ms NRx bound. Thus removing the host wait alone "
        "cannot safely recover the rejected NRx opportunities under an "
        "exclusive-GPU execution contract. Overlap requires a separately "
        "qualified interference bound or physical cancellation. The rejected "
        "inputs' eager outcomes identify potential radio value, not the "
        "correctness that a new implementation would necessarily achieve.",
        "",
        "The NRx GPU service and AI RPC distributions are observed values, not "
        "WCETs or safe admission bounds. Reducing the 40 ms AI host budget "
        "from these maxima alone would be an unjustified contract change.",
        "",
        f"Input/summary errors: {len(errors)}.",
    ]
    if errors:
        lines.extend(["", "Errors:", *(f"- {error}" for error in errors)])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"confirm51 blocking audit errors={len(errors)}")


if __name__ == "__main__":
    main()
