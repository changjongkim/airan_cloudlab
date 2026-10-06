#!/usr/bin/env python3
"""AI served at the same goodput loss: the settings of the rule against the caps of the low-priority baseline.

usage: analyze_frontier.py OUT.json "LABEL=LA_CLOSED.json" ...     (LA_CLOSED.json: output of analyze_la_closed.py)

Per condition there are two families of operating points (AI served, goodput lost against the recovery path
without AI):
  rule      wm (no AI next to a neural receiver) and wr3 / wr2 / wr1 (AI next to it while 3 / 2 / 1 neural
            receivers of the server are free)
  caps      low-priority AI with a cap of 20-100% (p20 ... p100) and the fixed 10% share (s10)
For a loss tolerance X, a family serves the most AI that its settings reach with a mean loss of at most X.  A
server can alternate between two settings (and between a setting and no AI), so the points between two measured
settings count: the AI at X is read from the straight lines between the measured points and the point without
AI (0 tokens/s, no loss), taking the upper envelope.  A loss below zero (run-to-run variation) counts as zero.
Printed per condition: every point, then per tolerance the AI of each family, the measured setting of each family
with the most AI inside the tolerance, and the ratio of the two families.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RULE = ("wm", "wr3", "wr2", "wr1")
CAPS = ("s10", "p20", "p30", "p40", "p50", "p60", "p70", "p100")
TOLERANCES = (0.25, 0.5, 1.0, 2.0)
LABEL = {"wm": "no AI next to NeuralRx", "wr3": "reserve 3", "wr2": "reserve 2", "wr1": "reserve 1", "s10": "fixed 10%",
         "p100": "no cap"}


def name(code: str) -> str:
    return LABEL.get(code, f"cap {code[1:]}%")


def best(points: dict, codes, tolerance: float):
    inside = [(points[c]["ai"], c) for c in codes if c in points and points[c]["loss"] <= tolerance + 1e-9]
    return max(inside)[1] if inside else None


def reach(points: dict, codes, tolerance: float) -> float:
    """Most AI (tokens/s) with a loss of at most `tolerance`, on the lines between the measured settings and no AI."""
    pts = [(0.0, 0.0)] + [(points[c]["ai"], max(0.0, points[c]["loss"])) for c in codes if c in points]
    top = max(ai for ai, loss in pts if loss <= tolerance + 1e-9)
    for i, (ai_a, loss_a) in enumerate(pts):
        for ai_b, loss_b in pts[i + 1:]:
            (lo_ai, lo), (hi_ai, hi) = sorted([(ai_a, loss_a), (ai_b, loss_b)], key=lambda q: q[1])
            if lo <= tolerance < hi:
                top = max(top, lo_ai + (tolerance - lo) / (hi - lo) * (hi_ai - lo_ai))
    return top


def main() -> None:
    out_path = Path(sys.argv[1])
    out = {}
    for item in sys.argv[2:]:
        label, path = item.rsplit("=", 1)
        rows = json.loads(Path(path).read_text())["policies"]
        base = rows["n"]["goodput_by_seed"]
        points = {}
        for code in RULE + CAPS:
            r = rows.get(code)
            if r is None:
                continue
            by_seed = [100.0 * (1.0 - g / b) for g, b in zip(r["goodput_by_seed"], base)]
            points[code] = {"ai": r["ai_slo"], "loss": -r["vs_recovery_no_ai_pct"], "loss_by_seed": by_seed,
                            "seeds": len(r["seeds"]), "lost_candidates_pct": r["lost_candidates_pct"],
                            "l1_late_pct": r["l1_late_pct"]}
        print(f"## {label}  (candidates without a NeuralRx, no AI: {rows['n']['lost_candidates_pct']:.1f}%; "
              f"slots with a candidate: {rows['n']['nrx_demand_pct']:.1f}%)")
        print("| setting | seeds | AI served | goodput lost (seeds) | candidates without a NeuralRx | L1 late |")
        print("|---|---|---|---|---|---|")
        for code in RULE + CAPS:
            if code in points:
                p = points[code]
                family = "rule" if code in RULE else "caps"
                print(f"| {family}: {name(code)} | {p['seeds']} | {p['ai'] / 1e3:.1f}k | {p['loss']:+.2f}% "
                      f"({min(p['loss_by_seed']):+.2f} to {max(p['loss_by_seed']):+.2f}) | {p['lost_candidates_pct']:.1f}% | "
                      f"{p['l1_late_pct']:.3f}% |")
        print("\n| goodput loss at most | rule: AI served | caps: AI served | rule / caps | rule: best measured setting | "
              "caps: best measured setting |")
        print("|---|---|---|---|---|---|")
        table = {}
        for tolerance in TOLERANCES:
            ours, theirs = best(points, RULE, tolerance), best(points, CAPS, tolerance)
            ai_rule, ai_caps = reach(points, RULE, tolerance), reach(points, CAPS, tolerance)
            cell = lambda c: "none" if c is None else f"{name(c)}, {points[c]['ai'] / 1e3:.1f}k ({points[c]['loss']:+.2f}%)"
            ratio = ai_rule / ai_caps if ai_caps > 0 else None
            table[str(tolerance)] = {"rule": ours, "caps": theirs, "rule_ai": ai_rule, "caps_ai": ai_caps, "ratio": ratio}
            print(f"| {tolerance:.2f}% | {ai_rule / 1e3:.1f}k | {ai_caps / 1e3:.1f}k | {'–' if ratio is None else f'{ratio:.2f}x'} | "
                  f"{cell(ours)} | {cell(theirs)} |")
        # caps that some setting of the rule dominates (at least the AI, at most the loss), and the reverse
        dominated = [c for c in CAPS if c in points and any(points[r]["ai"] >= points[c]["ai"] and points[r]["loss"] <= points[c]["loss"]
                                                           for r in RULE if r in points)]
        reverse = [r for r in RULE if r in points and any(points[c]["ai"] >= points[r]["ai"] and points[c]["loss"] <= points[r]["loss"]
                                                          for c in CAPS if c in points)]
        print(f"\ncaps with a setting of the rule that serves at least the AI at no more loss: {', '.join(name(c) for c in dominated) or 'none'}")
        print(f"settings of the rule with a cap that serves at least the AI at no more loss: {', '.join(name(c) for c in reverse) or 'none'}\n")
        out[label] = {"points": points, "tolerances": table, "caps_dominated": dominated, "rule_dominated": reverse}
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
