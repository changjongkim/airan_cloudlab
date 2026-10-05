#!/usr/bin/env python3
"""README tables of the closed-loop link adaptation runs (output of analyze_la_closed.py).

usage: table_la_closed.py full LA_CLOSED.json            one table: every policy of one condition
       table_la_closed.py brief "LABEL=LA_CLOSED.json" ...  one row per condition
       table_la_closed.py tex "LABEL=LA_CLOSED.json" ...    rows of the table of the paper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {"x": "복구 경로 없음", "xp100": "복구 경로 없음 + 낮은 우선순위 AI(한도 없음)", "n": "복구 경로, AI 없음",
         "wm": "**Antiphase**", "wr3": "Antiphase의 다른 설정: lane 여유 3", "wr1": "Antiphase의 다른 설정: lane 여유 1",
         "s10": "고정 10%", "p30": "낮은 우선순위 + 30%", "p70": "낮은 우선순위 + 70%", "p100": "낮은 우선순위, 한도 없음"}
ORDER = ("x", "xp100", "n", "wm", "wr3", "wr1", "s10", "p30", "p70", "p100")


def k(v: float) -> str:
    return f"{v / 1e3:.1f}k"


def full(path: str) -> None:
    rows = json.loads(Path(path).read_text())["policies"]
    print("| 방식 | 시드 | 평균 MCS | 재전송이 필요한 TB | goodput (bits/사용자 슬롯) | 복구 경로 없음 대비 | AI 없는 복구 경로 대비 | "
          "후보가 난 슬롯 | NRx를 못 받은 후보 | 마감 뒤에 끝난 NRx | AI 처리량 | L1 놓침 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for code in ORDER:
        r = rows.get(code)
        if r is None:
            continue
        nrx = code not in ("x", "xp100")
        print(f"| {NAMES[code]} | {len(r['seeds'])} | {r['mean_mcs']:.2f} | {r['retx_pct']:.1f}% | {k(r['goodput_bits'])} | "
              + ("–" if not nrx else f"{r['vs_no_recovery_pct']:+.1f}%") + " | "
              + ("–" if code == "n" else f"{r['vs_recovery_no_ai_pct']:+.1f}%") + " | "
              + f"{r['nrx_demand_pct']:.1f}% | " + (f"{r['lost_candidates_pct']:.1f}% | {r['late_candidates_pct']:.1f}%" if nrx else "– | –")
              + f" | {k(r['ai_slo']) if r['ai_slo'] else '–'} | {r['l1_late_pct']:.3f}% |")


def brief(items: list[str]) -> None:
    print("| 조건 | 시드 | 복구 경로의 이득 (평균 MCS) | AI 없음: 후보가 난 슬롯 / NRx를 못 받은 후보 | **Antiphase** | 고정 10% | "
          "낮은 우선순위 + 30% | 낮은 우선순위 + 70% | 낮은 우선순위, 한도 없음 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item in items:
        label, path = item.rsplit("=", 1)
        rows = json.loads(Path(path).read_text())["policies"]
        n, x = rows["n"], rows["x"]
        cells = []
        for code in ("wm", "s10", "p30", "p70", "p100"):
            r = rows.get(code)
            cells.append("–" if r is None else f"{k(r['ai_slo'])} / {r['vs_recovery_no_ai_pct']:+.1f}% / {r['mean_mcs'] - n['mean_mcs']:+.2f}")
        seeds = sorted({len(rows[c]["seeds"]) for c in ("n", "wm", "s10", "p30", "p70", "p100") if c in rows})
        print(f"| {label} | {seeds[0]}" + (f"–{seeds[-1]}" if len(seeds) > 1 else "") + f" | {n['vs_no_recovery_pct']:+.1f}% "
              f"({x['mean_mcs']:.2f} → {n['mean_mcs']:.2f}) | {n['nrx_demand_pct']:.1f}% / {n['lost_candidates_pct']:.1f}% | "
              + " | ".join(cells) + " |")


def tex(items: list[str]) -> None:
    for item in items:
        label, path = item.rsplit("=", 1)
        rows = json.loads(Path(path).read_text())["policies"]
        n, x = rows["n"], rows["x"]
        cells = []
        for code in ("wm", "s10", "p30", "p70", "p100"):
            r = rows[code]
            v = r["vs_recovery_no_ai_pct"]
            sign = "$+$" if v >= 0.05 else ("$-$" if v <= -0.05 else "")
            cells.append(f"{r['ai_slo'] / 1e3:.1f}k / {sign}{abs(v):.1f}\\%")
        print(f"{label} & $+${n['vs_no_recovery_pct']:.1f}\\% & {x['mean_mcs']:.1f} / {n['mean_mcs']:.1f} & "
              f"{n['nrx_demand_pct']:.0f}\\% & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    if sys.argv[1] == "tex":
        tex(sys.argv[2:])
        raise SystemExit
    import io
    from contextlib import redirect_stdout
    text = io.StringIO()
    with redirect_stdout(text):
        full(sys.argv[2]) if sys.argv[1] == "full" else brief(sys.argv[2:])
    lines = text.getvalue().splitlines()
    for i, line in enumerate(lines):                     # minus signs in the numbers (not in the separator row)
        print(line if i == 1 else line.replace(" -", " −").replace("(-", "(−"))
