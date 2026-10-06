#!/usr/bin/env python3
"""Tables of the scheduling metrics from the output of analyze_sched.py and analyze_warmup.py.

usage: table_sched.py md  SCHED.json            steady-state table for the README (Korean names)
       table_sched.py tex SCHED.json            the same rows for the paper
       table_sched.py when md|tex WARMUP.json   late TBs by the time since the start of the run
       table_sched.py ranges SCHED.json ...     what the scheme adds and how many times more each baseline adds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["n", "OURS", "s10", "yyr", "p30", "yyp", "e", "p70", "p100"]
KO = {"n": "AI 없음", "OURS": "**Antiphase**", "s10": "고정 10% (Static)", "yyr": "추정기 (Estimator)",
      "p30": "낮은 우선순위 + 30% (Priority)", "yyp": "낮은 우선순위 + 추정기", "e": "낮은 우선순위 + 부하 따라 비율",
      "p70": "낮은 우선순위 + 70%", "p100": "낮은 우선순위, 한도 없음"}
EN = {"n": "No AI", "OURS": r"\sys", "s10": r"\textit{Static}", "yyr": r"\textit{Estimator}", "p30": r"\textit{Priority}",
      "yyp": r"\textit{Estimator+Priority}", "e": r"\textit{Follow+Priority}", "p70": r"\textit{Priority-70}",
      "p100": r"\textit{Priority-max}"}


def rows(path: str) -> list[tuple[str, dict]]:
    by = {r["policy"]: r for r in json.loads(Path(path).read_text())}
    out = []
    for key in ORDER:
        if key == "OURS":
            r = by.get("wd") or by.get("wm")
        elif key == "e":
            r = next((by[c] for c in by if c[0] == "e"), None)
        else:
            r = by.get(key)
        if r is not None:
            out.append((key, r))
    return out


def per_million(r: dict) -> str:
    return f"{r['l1_late_tbs']} / {r['l1_tbs'] / 1e6:.2f}M"


def main() -> None:
    mode = sys.argv[1]
    if mode == "when":
        style, data = sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
        spans = ["first 5 s", "5-10 s", "10-20 s", "after 20 s"]
        for key in ORDER:
            code = next((c for c in data if (key == "OURS" and c in ("wd", "wm")) or (key == "e" and c[0] == "e") or c == key), None)
            if code is None:
                continue
            cells = []
            for span in spans:
                v = data[code]["spans"][span]
                cells.append("–" if not v["tbs"] else (f"{v['late']}" if span != "after 20 s" else f"{v['late']} / {v['tbs'] / 1e6:.2f}M"))
            if style == "md":
                print(f"| {KO[key]} | {data[code]['seeds']} | " + " | ".join(cells) + " |")
            else:
                print(f"{EN[key]} & " + " & ".join(cells) + r" \\")
        return
    if mode == "ranges":
        for path in sys.argv[2:]:
            rs = dict(rows(path))
            none, ours = rs["n"], rs["OURS"]
            print(Path(path).name)
            for label, key in (("L1 p99.9 added", "l1_p999_ms"), ("L1 p99.99 added", "l1_p9999_ms"), ("L1 p50 added", "l1_p50_ms"),
                               ("NeuralRx run p50 added", "nrx_run_p50_ms"), ("NeuralRx run p99 added", "nrx_run_p99_ms")):
                mine = ours[key] - none[key]
                others = "  ".join(f"{k} {r[key] - none[key]:+.2f} ({(r[key] - none[key]) / mine:.1f}x)" for k, r in rs.items() if k not in ("n", "OURS"))
                print(f"  {label}: ours {mine:+.3f}   {others}")
            print("  AI p99: " + "  ".join(f"{k} {r['ai_p99_ms']:.0f}" for k, r in rs.items() if r.get("ai_p99_ms")))
            print("  late per million: " + "  ".join(f"{k} {1e4 * r['l1_late_pooled_pct']:.1f}" for k, r in rs.items()))
        return
    f = lambda r, k, s: format(r[k], s) if r.get(k) is not None else "–"
    for key, r in rows(sys.argv[2]):
        ai = f"{r['ai_slo'] / 1e3:.1f}" if key != "n" else "0.0"
        if mode == "md":
            print(f"| {KO[key]} | {ai} | {f(r, 'recovered_pct', '.1f')}% | {r['l1_p50_ms']:.2f} / {r['l1_p999_ms']:.2f} / {r['l1_p9999_ms']:.2f} | "
                  f"{per_million(r)} | {f(r, 'nrx_run_p50_ms', '.2f')} / {f(r, 'nrx_run_p99_ms', '.2f')} | "
                  f"{f(r, 'ai_p50_ms', '.0f')} / {f(r, 'ai_p99_ms', '.0f')} | {f(r, 'ai_rejected_pct', '.1f')}{'%' if key != 'n' else ''} | {r['gpu_idle_pct']:.1f}% |")
        else:
            print(f"{EN[key]} & {ai} & {f(r, 'recovered_pct', '.1f')} & {r['l1_p50_ms']:.2f} / {r['l1_p999_ms']:.2f} & {r['l1_late_tbs']} & "
                  f"{f(r, 'nrx_run_p50_ms', '.2f')} / {f(r, 'nrx_run_p99_ms', '.2f')} & "
                  f"{(f(r, 'ai_p50_ms', '.0f') + ' / ' + f(r, 'ai_p99_ms', '.0f')) if key != 'n' else '--'} & "
                  f"{f(r, 'ai_rejected_pct', '.1f') if key != 'n' else '--'} & {r['gpu_idle_pct']:.1f} " + r"\\")


if __name__ == "__main__":
    main()
