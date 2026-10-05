#!/usr/bin/env python3
"""Korean tables of the link adaptation result (analyze_la.py output).

usage: table_la.py LINK_ADAPTATION.json
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    d = json.load(open(sys.argv[1]))
    points = d["points"]
    print("| Es/No (dB) | goodput 최대 MCS: 기존 → 복구 경로 | goodput 이득 | 10% 목표 MCS: 기존 → 복구 경로 | 이득: 같은 MCS | 이득: MCS를 올림 |")
    print("|---|---|---|---|---|---|")
    for esno in sorted(points, key=float):
        p = points[esno]
        raised = p["target10_mcs_with_recovery_timed"] > p["target10_mcs_conventional"]
        print(f"| {float(esno):.1f} | {p['best_mcs_conventional']} → {p['best_mcs_with_recovery_timed']} | {p['gain_best_mcs_timed_pct']:+.1f}% | "
              f"{p['target10_mcs_conventional']} → {'**' if raised else ''}{p['target10_mcs_with_recovery_timed']}{'**' if raised else ''} | "
              f"{p['gain_target10_same_mcs_timed_pct']:+.1f}% | {p['gain_target10_timed_pct']:+.1f}% |")
    s = d["summary"]
    print(f"\n평균({s['points']}개 Es/No): goodput 최대 규칙 {s['best_mean_gain_pct']:+.1f}% (MCS가 오른 점 {s['best_points_with_higher_mcs']}개); "
          f"10% 목표 규칙 같은 MCS {s['target10_mean_gain_same_mcs_pct']:+.1f}%, MCS를 올림 {s['target10_mean_gain_pct']:+.1f}% "
          f"(오른 점 {s['target10_points_with_higher_mcs']}개)")
    print("\n| MCS | 부호율 | 오류율 10%가 되는 Es/No: 기존 (dB) | 복구 경로 (dB) | 차이 (dB) |")
    print("|---|---|---|---|---|")
    rates = {}
    for p in points.values():
        for m, r in p["by_mcs"].items():
            rates[int(m)] = r["code_rate"]
    for m in sorted(rates):
        t = d["thresholds"].get(str(m)) or {}
        a, b = t.get("conventional"), t.get("with_recovery_timed")
        f = lambda v: "–" if v is None else f"{v:.2f}"
        print(f"| {m} | {rates[m]:.3f} | {f(a)} | {f(b)} | {'–' if a is None or b is None else f'{a - b:.2f}'} |")
    print("\n| Es/No (dB) | " + " | ".join(f"MCS {m}" for m in sorted(rates)) + " |")
    print("|---|" + "---|" * len(rates))
    for esno in sorted(points, key=float):
        by = points[esno]["by_mcs"]
        cells = []
        for m in sorted(rates):
            r = by.get(str(m))
            cells.append("–" if r is None else f"{100 * r['bler_conventional']:.1f} → {100 * r['bler_with_recovery_timed']:.1f}%")
        print(f"| {float(esno):.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
