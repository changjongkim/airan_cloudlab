#!/usr/bin/env python3
"""Link adaptation with a recovery path: which MCS gives the most goodput at a given Es/No.

usage: analyze_la.py CHANNEL [OUT.json]
inputs: results/backstop_slot/raw/la_CHANNEL_m<MCS>[_suffix].json from la.sh / la2.sh

Per MCS and Es/No point, both receivers decoded the same slots (20 LDPC iterations each):
  first-transmission block error rate of the conventional receiver, and of the conventional
  receiver followed by the neural receiver for the TBs it fails.
Goodput per uplink slot of a full-buffer user with retransmissions (each takes a slot and
succeeds with probability RETX_OK):   TB size / (1 + BLER / RETX_OK).
``TIMED`` scales the recoveries to what the timed system keeps of the offline recoveries.
Two link-adaptation policies:
  most goodput    the MCS with the most goodput (an ideal rate controller)
  10% target      the highest MCS whose first-transmission error rate is within 10% (the usual
                  outer-loop target); with the recovery path the error rate after recovery counts
Also the Es/No at which each MCS crosses the 10% target, with and without the recovery path
(log-linear interpolation between the measured points).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RAW = ROOT / "results/backstop_slot/raw"
RETX_OK = 0.95
TIMED = 0.86      # timed recoveries / offline recoveries (mixed channels: 38% of failures against 44%)
TARGET = 0.10


def crossing(points: list[tuple[float, float]]) -> float | None:
    """Es/No at which the error rate falls to TARGET (points: (Es/No, error rate), ascending Es/No)."""
    floor = 1e-3
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 > TARGET >= y1:
            a, b = math.log(max(y0, floor)), math.log(max(y1, floor))
            return x0 + (x1 - x0) * (a - math.log(TARGET)) / (a - b)
    return None


def main() -> None:
    channel = sys.argv[1]
    counts: dict[tuple[float, int], dict] = {}
    info: dict[int, dict] = {}
    for path in sorted(RAW.glob(f"la_{channel}_m*.json")):
        mcs = int(re.search(r"_m(\d+)", path.stem).group(1))
        probe = json.loads(path.read_text())
        meta = json.loads((Path(probe["dataset"]) / f"{probe['profile']}_meta.json").read_text())
        source = meta["source"]
        info[mcs] = {"tb_bits": meta["tb_bytes"] * 8, "code_rate": source["target_coderate"]}
        shift = 10 * math.log10(source["num_bits_per_symbol"] * source["target_coderate"])
        for ebno, r in probe["by_ebno_db"].items():
            esno = round(2 * (float(ebno) + shift)) / 2           # points are on a 0.5 dB grid
            c = counts.setdefault((esno, mcs), {"tbs": 0, "conventional": 0, "neural_only": 0})
            for key in c:
                c[key] += r[key]
    rows: dict[float, dict[int, dict]] = {}
    good = lambda bits, bler: bits / (1 + bler / RETX_OK)
    for (esno, mcs), c in sorted(counts.items()):
        conv = 1 - c["conventional"] / c["tbs"]
        both = 1 - (c["conventional"] + c["neural_only"]) / c["tbs"]
        timed = conv - TIMED * (conv - both)
        bits = info[mcs]["tb_bits"]
        rows.setdefault(esno, {})[mcs] = {
            **info[mcs], "tbs": c["tbs"], "bler_conventional": conv, "bler_with_recovery": both,
            "bler_with_recovery_timed": timed, "goodput_conventional": good(bits, conv),
            "goodput_with_recovery": good(bits, both), "goodput_with_recovery_timed": good(bits, timed)}
    out = {"channel": channel, "retx_ok": RETX_OK, "timed_share": TIMED, "target": TARGET, "points": {}, "thresholds": {}}
    for esno in sorted(rows):
        by = rows[esno]
        print(f"\n## {channel}, Es/No {esno} dB")
        print("| MCS | code rate | TB bits | TBs | BLER conventional | BLER with recovery (timed) | goodput conventional (bits/slot) | goodput with recovery (timed) |")
        print("|---|---|---|---|---|---|---|---|")
        for mcs in sorted(by):
            r = by[mcs]
            print(f"| {mcs} | {r['code_rate']:.3f} | {r['tb_bits']} | {r['tbs']} | {100 * r['bler_conventional']:.1f}% | "
                  f"{100 * r['bler_with_recovery']:.1f}% ({100 * r['bler_with_recovery_timed']:.1f}%) | "
                  f"{r['goodput_conventional']:.0f} | {r['goodput_with_recovery']:.0f} ({r['goodput_with_recovery_timed']:.0f}) |")
        best = lambda key: max(by, key=lambda m: by[m][key])
        target = lambda key: max([m for m in by if by[m][key] <= TARGET] or [min(by)])
        m0, m1, m2 = best("goodput_conventional"), best("goodput_with_recovery"), best("goodput_with_recovery_timed")
        t0, t1 = target("bler_conventional"), target("bler_with_recovery_timed")
        g0 = by[m0]["goodput_conventional"]
        point = {
            "best_mcs_conventional": m0, "best_mcs_with_recovery": m1, "best_mcs_with_recovery_timed": m2,
            "gain_best_mcs_pct": 100 * (by[m1]["goodput_with_recovery"] / g0 - 1),
            "gain_best_mcs_timed_pct": 100 * (by[m2]["goodput_with_recovery_timed"] / g0 - 1),
            "gain_same_mcs_timed_pct": 100 * (by[m0]["goodput_with_recovery_timed"] / g0 - 1),
            "target10_mcs_conventional": t0, "target10_mcs_with_recovery_timed": t1,
            "target10_goodput_conventional": by[t0]["goodput_conventional"],
            "target10_goodput_same_mcs_timed": by[t0]["goodput_with_recovery_timed"],
            "target10_goodput_with_recovery_timed": by[t1]["goodput_with_recovery_timed"],
            "gain_target10_same_mcs_timed_pct": 100 * (by[t0]["goodput_with_recovery_timed"] / by[t0]["goodput_conventional"] - 1),
            "gain_target10_timed_pct": 100 * (by[t1]["goodput_with_recovery_timed"] / by[t0]["goodput_conventional"] - 1),
            "by_mcs": by}
        out["points"][str(esno)] = point
        print(f"most goodput: conventional MCS {m0}, with recovery MCS {m2}; goodput gain {point['gain_best_mcs_timed_pct']:+.1f}% "
              f"(at the conventional MCS {point['gain_same_mcs_timed_pct']:+.1f}%)")
        print(f"10% target: conventional MCS {t0}, with recovery MCS {t1}; goodput gain {point['gain_target10_timed_pct']:+.1f}% "
              f"(at the conventional MCS {point['gain_target10_same_mcs_timed_pct']:+.1f}%)")
    print("\n## Es/No at which the error rate crosses the 10% target")
    print("| MCS | conventional (dB) | with recovery, timed (dB) | shift (dB) |")
    print("|---|---|---|---|")
    for mcs in sorted(info):
        series = lambda key: [(e, rows[e][mcs][key]) for e in sorted(rows) if mcs in rows[e]]
        a, b = crossing(series("bler_conventional")), crossing(series("bler_with_recovery_timed"))
        out["thresholds"][str(mcs)] = {"conventional": a, "with_recovery_timed": b}
        f = lambda v: "–" if v is None else f"{v:.2f}"
        print(f"| {mcs} | {f(a)} | {f(b)} | {'–' if a is None or b is None else f'{a - b:.2f}'} |")
    # Raising the target: the MCS is the highest whose conventional error rate is within the target.
    print("\n## Goodput by the error-rate target of the conventional receiver (mean over the Es/No points, bits per slot)")
    print("| target | conventional receiver only | with the recovery path (same MCS) | gain |")
    print("|---|---|---|---|")
    out["by_target"] = {}
    for goal in (0.05, 0.10, 0.20, 0.30, 0.40):
        conv_sum = rec_sum = 0.0
        for esno in sorted(rows):
            by = rows[esno]
            mcs = max([m for m in by if by[m]["bler_conventional"] <= goal] or [min(by)])
            conv_sum += by[mcs]["goodput_conventional"]
            rec_sum += by[mcs]["goodput_with_recovery_timed"]
        n = len(rows)
        out["by_target"][str(goal)] = {"goodput_conventional": conv_sum / n, "goodput_with_recovery_timed": rec_sum / n}
        print(f"| {100 * goal:.0f}% | {conv_sum / n:.0f} | {rec_sum / n:.0f} | {100 * (rec_sum / conv_sum - 1):+.1f}% |")
    best_conv = max(out["by_target"].values(), key=lambda v: v["goodput_conventional"])["goodput_conventional"]
    best_rec = max(out["by_target"].values(), key=lambda v: v["goodput_with_recovery_timed"])["goodput_with_recovery_timed"]
    print(f"best target of each: conventional {best_conv:.0f}, with recovery {best_rec:.0f} ({100 * (best_rec / best_conv - 1):+.1f}%)")
    points = out["points"].values()
    mean = lambda key: sum(p[key] for p in points) / len(out["points"])
    summary = {
        "points": len(out["points"]),
        "target10_mean_gain_same_mcs_pct": 100 * (mean("target10_goodput_same_mcs_timed") / mean("target10_goodput_conventional") - 1),
        "target10_mean_gain_pct": 100 * (mean("target10_goodput_with_recovery_timed") / mean("target10_goodput_conventional") - 1),
        "target10_points_with_higher_mcs": sum(p["target10_mcs_with_recovery_timed"] > p["target10_mcs_conventional"] for p in points),
        "best_mean_gain_pct": 100 * (sum(p["by_mcs"][p["best_mcs_with_recovery_timed"]]["goodput_with_recovery_timed"] for p in points)
                                    / sum(p["by_mcs"][p["best_mcs_conventional"]]["goodput_conventional"] for p in points) - 1),
        "best_points_with_higher_mcs": sum(p["best_mcs_with_recovery_timed"] > p["best_mcs_conventional"] for p in points),
    }
    out["summary"] = summary
    print(f"\nmean over {summary['points']} Es/No points, 10% target: goodput {summary['target10_mean_gain_same_mcs_pct']:+.1f}% at the "
          f"conventional MCS, {summary['target10_mean_gain_pct']:+.1f}% with the MCS raised "
          f"(raised at {summary['target10_points_with_higher_mcs']} points)")
    print(f"most goodput: {summary['best_mean_gain_pct']:+.1f}% (MCS raised at {summary['best_points_with_higher_mcs']} points)")
    target_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"results/backstop_slot/link_adaptation_{channel}.json"
    target_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
