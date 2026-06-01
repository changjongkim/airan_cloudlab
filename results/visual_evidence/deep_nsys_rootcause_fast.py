#!/usr/bin/env python3
"""Fast NSYS root-cause pass for selected scenarios.

This intentionally avoids expensive runtime-event cross scans. It answers the
important question first: are long kernel-to-kernel gaps true idle gaps, or are
they filled by memcpy/memset activity?
"""

from __future__ import annotations

import csv
import sqlite3
import statistics as stats
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "20260531" / "nsys_sqlite_v2"
OUT = Path(__file__).resolve().parent / "deep_nsys_fast"

LABELS = {
    "S5_3g_alone": "L1 3g alone",
    "S7_3g_neuralrx": "L1 3g + NeuralRx",
    "S10_2g_alone": "L1 2g alone",
    "S35_2g_chanpred": "L1 2g + ChanPred",
    "S28_3g_resnet": "L1 3g + ResNet",
    "S31_3g_resnet_chanpred": "L1 3g + ResNet+ChanPred",
    "S32_3g_resnet_forecaster": "L1 3g + ResNet+Forecaster",
    "S34_4g_resnet": "L1 4g + ResNet",
    "S2_7g_mig": "L1 7g MIG alone",
}

SCENARIOS = list(LABELS)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q / 100))]


def avg(vals: list[float]) -> float:
    return stats.mean(vals) if vals else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def short(name: str) -> str:
    if "convert_kernel" in name:
        return "convert_kernel"
    if "complex64" in name:
        return "copy_complex64_kernel"
    if "float32" in name:
        return "copy_float32_kernel"
    if "float16" in name:
        return "copy_float16_kernel"
    if "windowedChEstPre" in name:
        return "ch_est_pre"
    if "chEstFilterNoDft" in name:
        return "ch_est_filter"
    if "noiseIntfEst" in name:
        return "noise_intf_est"
    if "eqMmseCoefComp" in name:
        return "eq_coef"
    if "eqMmseSoftDemap" in name:
        return "eq_softdemap"
    return name[:60]


def load(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    kernels = [
        {"s": r["start"], "e": r["end"], "name": short(r["name"] or "unknown")}
        for r in con.execute(
            """
            SELECT k.start, k.end, COALESCE(sd.value, ss.value, 'unknown') AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds sd ON k.demangledName = sd.id
            LEFT JOIN StringIds ss ON k.shortName = ss.id
            ORDER BY k.start
            """
        )
    ]
    memcpy = [
        {"s": r["start"], "e": r["end"], "name": r["kind"] or f"memcpy_{r['copyKind']}", "bytes": r["bytes"]}
        for r in con.execute(
            """
            SELECT m.start, m.end, m.bytes, m.copyKind, e.label AS kind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
            ORDER BY m.start
            """
        )
    ]
    memset = [
        {"s": r["start"], "e": r["end"], "name": "memset", "bytes": r["bytes"]}
        for r in con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start")
    ]
    con.close()
    return kernels, memcpy, memset


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]


def overlap(events: list[dict], s: int, e: int) -> tuple[int, int]:
    dur = 0
    cnt = 0
    for ev in events:
        if ev["e"] <= s:
            continue
        if ev["s"] >= e:
            break
        ov = max(0, min(ev["e"], e) - max(ev["s"], s))
        if ov:
            dur += ov
            cnt += 1
    return dur, cnt


def analyze_db(path: Path) -> tuple[dict, list[dict], list[dict]]:
    stem = path.stem
    scenario, run_s = stem.rsplit("_run", 1)
    run = int(run_s)
    kernels, memcpy, memset = load(path)
    mem = sorted(memcpy + memset, key=lambda x: x["s"])
    all_events = sorted(kernels + mem, key=lambda x: x["s"])
    wall = max(e["e"] for e in all_events) - min(e["s"] for e in all_events)
    kernel_busy = sum(e - s for s, e in merge([(k["s"], k["e"]) for k in kernels]))
    all_busy = sum(e - s for s, e in merge([(a["s"], a["e"]) for a in all_events]))
    all_gaps = []
    merged_all = merge([(a["s"], a["e"]) for a in all_events])
    for a, b in zip(merged_all, merged_all[1:]):
        if b[0] > a[1]:
            all_gaps.append((b[0] - a[1]) / 1000)

    kgaps = []
    transitions = []
    for a, b in zip(kernels, kernels[1:]):
        g = b["s"] - a["e"]
        if g < 0:
            continue
        mem_ns, mem_cnt = overlap(mem, a["e"], b["s"])
        gap_us = g / 1000
        kgaps.append(gap_us)
        if gap_us >= 500:
            transitions.append(
                {
                    "scenario": scenario,
                    "condition": LABELS[scenario],
                    "run": run,
                    "transition": f"{a['name']} -> {b['name']}",
                    "gap_us": gap_us,
                    "mem_us_inside": mem_ns / 1000,
                    "mem_count_inside": mem_cnt,
                    "mem_fraction": mem_ns / g if g else 0.0,
                }
            )
    big = [t for t in transitions if t["gap_us"] >= 1000]
    summary = {
        "scenario": scenario,
        "condition": LABELS[scenario],
        "run": run,
        "kernel_busy_pct": 100 * kernel_busy / wall,
        "all_activity_busy_pct": 100 * all_busy / wall,
        "kernel_gap_p50_us": pct(kgaps, 50),
        "kernel_gap_p99_us": pct(kgaps, 99),
        "all_activity_gap_p99_us": pct(all_gaps, 99),
        "big_kernel_gaps_ge_1ms": len(big),
        "big_gaps_with_mem_pct": 100 * sum(1 for t in big if t["mem_count_inside"] > 0) / len(big) if big else 0,
        "mean_mem_fraction_in_big_gaps_pct": 100 * avg([t["mem_fraction"] for t in big]),
    }
    memsum = []
    for name, events in [("memcpy", memcpy), ("memset", memset)]:
        memsum.append(
            {
                "scenario": scenario,
                "condition": LABELS[scenario],
                "run": run,
                "op": name,
                "count": len(events),
                "duration_ms": sum(e["e"] - e["s"] for e in events) / 1e6,
                "bytes_mb": sum(e.get("bytes") or 0 for e in events) / (1024 * 1024),
            }
        )
    return summary, transitions, memsum


def aggregate_rows(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(buckets.items()):
        row = {k: v for k, v in zip(keys, key)}
        row["runs"] = len(rs)
        for m in metrics:
            row[m] = avg([float(r[m]) for r in rs])
        out.append(row)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    transitions = []
    mem = []
    for scenario in SCENARIOS:
        for path in sorted(DB_ROOT.glob(f"{scenario}_run*.sqlite")):
            s, t, m = analyze_db(path)
            summaries.append(s)
            transitions.extend(t)
            mem.extend(m)

    summary = aggregate_rows(
        summaries,
        ["scenario", "condition"],
        [
            "kernel_busy_pct",
            "all_activity_busy_pct",
            "kernel_gap_p50_us",
            "kernel_gap_p99_us",
            "all_activity_gap_p99_us",
            "big_kernel_gaps_ge_1ms",
            "big_gaps_with_mem_pct",
            "mean_mem_fraction_in_big_gaps_pct",
        ],
    )
    for r in summary:
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = f"{v:.1f}"
    write_csv(OUT / "kernel_vs_all_activity_summary.csv", summary, list(summary[0]))

    tb = defaultdict(list)
    for t in transitions:
        tb[(t["condition"], t["transition"])].append(t)
    top = []
    for (cond, trans), ts in tb.items():
        gaps = [t["gap_us"] for t in ts]
        top.append(
            {
                "condition": cond,
                "transition": trans,
                "count": len(ts),
                "p50_gap_us": f"{pct(gaps, 50):.1f}",
                "p99_gap_us": f"{pct(gaps, 99):.1f}",
                "max_gap_ms": f"{max(gaps) / 1000:.1f}",
                "mean_mem_us_inside": f"{avg([t['mem_us_inside'] for t in ts]):.1f}",
                "mean_mem_fraction_pct": f"{100 * avg([t['mem_fraction'] for t in ts]):.1f}",
            }
        )
    top.sort(key=lambda r: (r["condition"], -float(r["p99_gap_us"])))
    selected_top = []
    for cond in sorted({r["condition"] for r in top}):
        selected_top.extend([r for r in top if r["condition"] == cond][:8])
    write_csv(OUT / "top_gap_transitions_by_condition.csv", selected_top, list(selected_top[0]))

    memagg = aggregate_rows(mem, ["scenario", "condition", "op"], ["count", "duration_ms", "bytes_mb"])
    for r in memagg:
        r["count"] = f"{r['count']:.0f}"
        r["duration_ms"] = f"{r['duration_ms']:.1f}"
        r["bytes_mb"] = f"{r['bytes_mb']:.1f}"
    write_csv(OUT / "memory_activity_summary.csv", memagg, list(memagg[0]))

    md = "# NSYS Deep Root-Cause Fast Pass\n\n"
    md += "## Kernel-only gap vs all GPU activity gap\n\n"
    md += "핵심은 `kernel_gap_p99_us`와 `all_activity_gap_p99_us`의 차이다. 차이가 크면 kernel 사이가 진짜 idle이 아니라 memcpy/memset으로 채워졌다는 뜻이다.\n\n"
    md += "![kernel vs activity gap](../figures/fig12_nsys_kernel_vs_activity_gap.png)\n\n"
    md += table(summary, ["condition", "runs", "kernel_busy_pct", "all_activity_busy_pct", "kernel_gap_p99_us", "all_activity_gap_p99_us", "big_kernel_gaps_ge_1ms", "big_gaps_with_mem_pct", "mean_mem_fraction_in_big_gaps_pct"])
    md += "\n\n## Top long-gap transitions\n\n"
    for cond in sorted({r["condition"] for r in selected_top}):
        md += f"### {cond}\n\n"
        rows = [r for r in selected_top if r["condition"] == cond]
        md += table(rows, ["transition", "count", "p50_gap_us", "p99_gap_us", "max_gap_ms", "mean_mem_us_inside", "mean_mem_fraction_pct"]) + "\n\n"
    md += "## Memory activity summary\n\n"
    md += "![memory activity](../figures/fig13_nsys_memory_activity_breakdown.png)\n\n"
    md += table(memagg, ["condition", "op", "count", "duration_ms", "bytes_mb"])
    md += "\n\n## Interpretation\n\n"
    md += (
        "이 재분석에서 문제 지점은 더 명확하다. 기존 kernel-only gap은 진짜 idle만 의미하지 않는다. "
        "많은 long gap은 memcpy/memset으로 채워진 kernel boundary이다. 따라서 NSYS 근거는 "
        "`bandwidth total`보다 `L1 pipeline의 convert/copy/memset boundary가 co-tenant와 partition size에 따라 길어진다`로 써야 한다. "
        "`L1 2g alone`과 `L1 2g + ChanPred`는 1ms 이상 kernel gap의 96.7%, 99.1%가 memory op를 포함하고, "
        "memset duration도 3g baseline 대비 거의 2배다. `L1 3g + NeuralRx`는 memcpy total이 3g baseline 대비 4.2배로 늘고, "
        "big kernel gap 수와 memory 포함 비율도 함께 증가한다. "
        "이것이 MIG가 static capacity slicing만으로 AI-RAN의 temporal guarantee를 주지 못한다는 더 구체적인 메커니즘이다.\n"
    )
    (OUT / "NSYS_DEEP_ROOTCAUSE_FAST_KR.md").write_text(md)
    print(f"wrote {OUT / 'NSYS_DEEP_ROOTCAUSE_FAST_KR.md'}")


def table(rows: list[dict], fields: list[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
            *["| " + " | ".join(str(r.get(f, "")) for f in fields) + " |" for r in rows],
        ]
    )


if __name__ == "__main__":
    main()
