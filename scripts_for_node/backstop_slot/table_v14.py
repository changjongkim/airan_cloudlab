#!/usr/bin/env python3
"""Compact Korean tables from sweep JSON files (analyze_sweep.py output).

usage: table_v14.py SWEEP.json [POLICY ...]
Prints one row per policy: AI served (seed range), recoveries kept (seed range), L1 late, seeds.
"""

from __future__ import annotations

import json
import os
import sys

OUR = os.environ.get("OUR_V14", "wm")
NAMES = {
    "n": "AI 없음", "wm": "**Antiphase**", "wn": "큰 단위도 기존 수신기 옆 허용", "ws": "가장 작은 단위만 기존 수신기 옆 허용", "wc": "큰 단위도 허용 + 70% 한도",
    "wr3": "큰 단위도 허용 + lane 여유 3", "wr2": "큰 단위도 허용 + lane 여유 2", "wr1": "큰 단위도 허용 + lane 여유 1",
    "vf": "v13 규칙 (조각 중단 없음)", "p100": "낮은 우선순위, 한도 없음",
    "yyr": "추정기로 비율 선택", "yyp": "낮은 우선순위 + 추정기",
}


def name(code: str) -> str:
    if code in NAMES:
        return NAMES[code]
    if code[0] == "s":
        return f"고정 {code[1:]}%"
    if code[0] == "p":
        return f"낮은 우선순위 + {code[1:]}%"
    shares = code[1:].split("l")[0].replace("x", "/")
    return (f"부하 따라 {shares}%" if code[0] == "d" else f"낮은 우선순위 + 부하 따라 {shares}%")


def main() -> None:
    rows = {r["policy"]: r for r in json.load(open(sys.argv[1]))}
    order = sys.argv[2:] or [c for c in rows if c != "n"]
    ours = rows.get(OUR)
    print("| 방식 | AI 처리량 (시드 범위) | 복구 유지율 (시드 범위) | L1 놓침 | 시드 | Antiphase의 AI 처리량 배수 |")
    print("|---|---|---|---|---|---|")
    for code in order:
        r = rows.get(code)
        if r is None:
            continue
        if code == "n":
            print(f"| AI 없음 | – | 100% | {r['l1_late_pct']:.3f}% | {len(r['seeds'])} | – |")
            continue
        ratio = "–" if code == OUR or not ours else f"{ours['ai_slo'] / r['ai_slo']:.2f}배"
        bold = "**" if code == OUR else ""
        print(f"| {name(code)} | {bold}{r['ai_slo'] / 1e3:.1f}k{bold} ({r['ai_slo_min'] / 1e3:.1f}–{r['ai_slo_max'] / 1e3:.1f}k) | "
              f"{bold}{r['recovered_pct']:.1f}%{bold} ({r['recovered_pct_min']:.1f}–{r['recovered_pct_max']:.1f}%) | "
              f"{r['l1_late_pct']:.3f}% | {len(r['seeds'])} | {ratio} |")


if __name__ == "__main__":
    main()
