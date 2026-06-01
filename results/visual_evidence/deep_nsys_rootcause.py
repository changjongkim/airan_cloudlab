#!/usr/bin/env python3
"""Deep NSYS SQLite root-cause analysis for cuPHY L1 + AI co-tenancy.

This script goes below the previously generated summary CSVs. In particular it
checks whether "kernel-to-kernel gaps" are true GPU idle gaps or whether they
are filled by memcpy/memset activity. That distinction is critical for the
AI-RAN/MIG argument.
"""

from __future__ import annotations

import bisect
import csv
import sqlite3
import statistics as stats
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = ROOT / "20260531" / "nsys_sqlite_v2"
OUT = Path(__file__).resolve().parent / "deep_nsys"


LABELS = {
    "S2_7g_mig": "L1 7g MIG alone",
    "S5_3g_alone": "L1 3g alone",
    "S6_3g_qwen": "L1 3g + Qwen",
    "S7_3g_neuralrx": "L1 3g + NeuralRx",
    "S9_3g_3AI_1g": "L1 3g + 3 AI on 1g slices",
    "S10_2g_alone": "L1 2g alone",
    "S12_2g_2AI": "L1 2g + 2 AI",
    "S13_3g_sat_compute": "L1 3g + GEMM/sat-compute",
    "S14_3g_sat_hbm": "L1 3g + HBM saturation",
    "S15_4g_sat_compute": "L1 4g + GEMM/sat-compute",
    "S17_2g_sat_compute": "L1 2g + GEMM/sat-compute",
    "S18_4g_neuralrx": "L1 4g + NeuralRx",
    "S21_4g_2sat": "L1 4g + 2 synthetic stressors",
    "S22_2g_neuralrx": "L1 2g + NeuralRx",
    "S24_3g_2sat": "L1 3g + 2 synthetic stressors",
    "S26_4g_3sat": "L1 4g + 3 synthetic stressors",
    "S27_3g_chanpred": "L1 3g + ChanPred",
    "S28_3g_resnet": "L1 3g + ResNet",
    "S29_3g_forecaster": "L1 3g + Forecaster",
    "S30_3g_xapp": "L1 3g + XApp",
    "S31_3g_resnet_chanpred": "L1 3g + ResNet+ChanPred",
    "S32_3g_resnet_forecaster": "L1 3g + ResNet+Forecaster",
    "S33_4g_chanpred": "L1 4g + ChanPred",
    "S34_4g_resnet": "L1 4g + ResNet",
    "S35_2g_chanpred": "L1 2g + ChanPred",
    "S36_4g_forecaster": "L1 4g + Forecaster",
}

SELECTED = [
    "S5_3g_alone",
    "S7_3g_neuralrx",
    "S10_2g_alone",
    "S35_2g_chanpred",
    "S28_3g_resnet",
    "S31_3g_resnet_chanpred",
    "S32_3g_resnet_forecaster",
]


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int(len(vals) * q / 100.0))
    return vals[idx]


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return stats.mean(vals) if vals else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def short_name(name: str) -> str:
    if "convert_kernel" in name:
        return "convert_kernel"
    if "complex64" in name:
        return "copy_complex64"
    if "float32" in name:
        return "copy_float32"
    if "float16" in name:
        return "copy_float16"
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
    if len(name) > 48:
        return name[:45] + "..."
    return name


def scenario_from_path(path: Path) -> tuple[str, int]:
    stem = path.stem
    scenario, run = stem.rsplit("_run", 1)
    return scenario, int(run)


def load_db(path: Path) -> dict:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    kernels = [
        {
            "start": r["start"],
            "end": r["end"],
            "type": "kernel",
            "name": short_name(r["name"] or "unknown"),
            "stream": r["streamId"],
            "corr": r["correlationId"],
        }
        for r in con.execute(
            """
            SELECT k.start, k.end, k.streamId, k.correlationId,
                   COALESCE(sd.value, ss.value, 'unknown') AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds sd ON k.demangledName = sd.id
            LEFT JOIN StringIds ss ON k.shortName = ss.id
            ORDER BY k.start
            """
        )
    ]
    memcpy = [
        {
            "start": r["start"],
            "end": r["end"],
            "type": "memcpy",
            "name": r["copy_name"] or f"memcpy_{r['copyKind']}",
            "stream": r["streamId"],
            "bytes": r["bytes"],
            "corr": r["correlationId"],
        }
        for r in con.execute(
            """
            SELECT m.start, m.end, m.streamId, m.correlationId, m.bytes, m.copyKind, e.label AS copy_name
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
            ORDER BY m.start
            """
        )
    ]
    memset = [
        {
            "start": r["start"],
            "end": r["end"],
            "type": "memset",
            "name": "memset",
            "stream": r["streamId"],
            "bytes": r["bytes"],
            "corr": r["correlationId"],
        }
        for r in con.execute("SELECT start, end, streamId, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start")
    ]
    runtime = [
        {
            "start": r["start"],
            "end": r["end"],
            "name": r["name"] or "runtime",
            "corr": r["correlationId"],
        }
        for r in con.execute(
            """
            SELECT r.start, r.end, r.correlationId, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            LEFT JOIN StringIds s ON r.nameId = s.id
            ORDER BY r.start
            """
        )
    ]
    con.close()
    return {"kernels": kernels, "memcpy": memcpy, "memset": memset, "runtime": runtime}


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for s, e in sorted(intervals):
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def interval_overlap_total(intervals: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    total = 0
    count = 0
    for s, e in intervals:
        if e <= start:
            continue
        if s >= end:
            break
        ov = max(0, min(e, end) - max(s, start))
        if ov:
            total += ov
            count += 1
    return total, count


def analyze_run(path: Path) -> tuple[dict, list[dict], list[dict]]:
    scenario, run = scenario_from_path(path)
    d = load_db(path)
    kernels = d["kernels"]
    mem_events = sorted(d["memcpy"] + d["memset"], key=lambda x: x["start"])
    all_events = sorted(kernels + mem_events, key=lambda x: x["start"])
    mem_intervals = [(e["start"], e["end"]) for e in mem_events]
    kernel_intervals = [(e["start"], e["end"]) for e in kernels]
    all_intervals = [(e["start"], e["end"]) for e in all_events]
    merged_kernel = merge_intervals(kernel_intervals)
    merged_all = merge_intervals(all_intervals)

    wall_start = min(s for s, _ in all_intervals)
    wall_end = max(e for _, e in all_intervals)
    wall = wall_end - wall_start
    kernel_busy = sum(e - s for s, e in merged_kernel)
    all_busy = sum(e - s for s, e in merged_all)
    all_idle = max(0, wall - all_busy)

    kernel_gaps = []
    transitions = []
    for a, b in zip(kernels, kernels[1:]):
        gap = b["start"] - a["end"]
        if gap < 0:
            continue
        mem_ns, mem_count = interval_overlap_total(mem_intervals, a["end"], b["start"])
        runtime_inside = 0
        for r in d["runtime"]:
            if r["end"] <= a["end"]:
                continue
            if r["start"] >= b["start"]:
                break
            runtime_inside += max(0, min(r["end"], b["start"]) - max(r["start"], a["end"]))
        kernel_gaps.append(gap)
        transitions.append(
            {
                "scenario": scenario,
                "run": run,
                "from": a["name"],
                "to": b["name"],
                "gap_us": gap / 1000.0,
                "mem_us_inside": mem_ns / 1000.0,
                "mem_count_inside": mem_count,
                "runtime_us_inside": runtime_inside / 1000.0,
                "mem_fraction": mem_ns / gap if gap else 0.0,
            }
        )

    all_gaps = [max(0, b[0] - a[1]) for a, b in zip(merged_all, merged_all[1:]) if b[0] > a[1]]
    big = [t for t in transitions if t["gap_us"] >= 1000.0]
    mem_filled = [t for t in big if t["mem_count_inside"] > 0]
    summary = {
        "scenario": scenario,
        "condition": LABELS.get(scenario, scenario),
        "run": run,
        "wall_ms": wall / 1e6,
        "kernel_busy_ms": kernel_busy / 1e6,
        "all_gpu_busy_ms": all_busy / 1e6,
        "all_gpu_idle_ms": all_idle / 1e6,
        "kernel_busy_pct": 100 * kernel_busy / wall if wall else 0,
        "all_gpu_busy_pct": 100 * all_busy / wall if wall else 0,
        "kernel_gap_p50_us": pct([g / 1000.0 for g in kernel_gaps], 50),
        "kernel_gap_p99_us": pct([g / 1000.0 for g in kernel_gaps], 99),
        "all_activity_gap_p99_us": pct([g / 1000.0 for g in all_gaps], 99),
        "big_kernel_gaps_ge_1ms": len(big),
        "big_kernel_gaps_with_mem_pct": 100 * len(mem_filled) / len(big) if big else 0,
        "mean_mem_fraction_inside_big_gaps": 100 * mean([t["mem_fraction"] for t in big]),
    }
    return summary, transitions, mem_events


def aggregate() -> tuple[list[dict], list[dict], list[dict]]:
    summaries = []
    transitions = []
    mem_rows = []
    for path in sorted(DB_ROOT.glob("*.sqlite")):
        summary, trans, mem = analyze_run(path)
        summaries.append(summary)
        transitions.extend(trans)
        scen, run = scenario_from_path(path)
        by_type = defaultdict(lambda: {"count": 0, "dur_ns": 0, "bytes": 0})
        for e in mem:
            key = e["name"] if e["type"] == "memcpy" else "memset"
            by_type[key]["count"] += 1
            by_type[key]["dur_ns"] += e["end"] - e["start"]
            by_type[key]["bytes"] += e.get("bytes", 0) or 0
        for k, v in by_type.items():
            mem_rows.append(
                {
                    "scenario": scen,
                    "condition": LABELS.get(scen, scen),
                    "run": run,
                    "op": k,
                    "count": v["count"],
                    "duration_ms": v["dur_ns"] / 1e6,
                    "bytes_mb": v["bytes"] / (1024 * 1024),
                }
            )
    return summaries, transitions, mem_rows


def summarize_by_scenario(summaries: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in summaries:
        buckets[r["scenario"]].append(r)
    rows = []
    for scenario, rs in sorted(buckets.items()):
        rows.append(
            {
                "scenario": scenario,
                "condition": LABELS.get(scenario, scenario),
                "runs": len(rs),
                "wall_ms": f"{mean([r['wall_ms'] for r in rs]):.1f}",
                "kernel_busy_pct": f"{mean([r['kernel_busy_pct'] for r in rs]):.1f}",
                "all_gpu_busy_pct": f"{mean([r['all_gpu_busy_pct'] for r in rs]):.1f}",
                "kernel_gap_p99_us": f"{mean([r['kernel_gap_p99_us'] for r in rs]):.0f}",
                "all_activity_gap_p99_us": f"{mean([r['all_activity_gap_p99_us'] for r in rs]):.1f}",
                "big_kernel_gaps_ge_1ms": f"{mean([r['big_kernel_gaps_ge_1ms'] for r in rs]):.0f}",
                "big_kernel_gaps_with_mem_pct": f"{mean([r['big_kernel_gaps_with_mem_pct'] for r in rs]):.1f}",
                "mean_mem_fraction_inside_big_gaps": f"{mean([r['mean_mem_fraction_inside_big_gaps'] for r in rs]):.1f}",
            }
        )
    return rows


def top_transition_rows(transitions: list[dict], scenario: str, min_gap_us: float = 500.0) -> list[dict]:
    buckets = defaultdict(list)
    for t in transitions:
        if t["scenario"] != scenario or t["gap_us"] < min_gap_us:
            continue
        buckets[(t["from"], t["to"])].append(t)
    rows = []
    for (a, b), ts in buckets.items():
        rows.append(
            {
                "condition": LABELS.get(scenario, scenario),
                "transition": f"{a} -> {b}",
                "count": len(ts),
                "p50_gap_us": f"{pct([t['gap_us'] for t in ts], 50):.1f}",
                "p99_gap_us": f"{pct([t['gap_us'] for t in ts], 99):.1f}",
                "max_gap_ms": f"{max(t['gap_us'] for t in ts) / 1000.0:.1f}",
                "mean_mem_us_inside": f"{mean([t['mem_us_inside'] for t in ts]):.1f}",
                "mean_mem_fraction": f"{100 * mean([t['mem_fraction'] for t in ts]):.1f}%",
            }
        )
    rows.sort(key=lambda r: float(r["p99_gap_us"]), reverse=True)
    return rows[:12]


def mem_summary(mem_rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for r in mem_rows:
        if r["op"] in {"Device-to-Device", "memset"}:
            buckets[(r["scenario"], r["op"])].append(r)
    rows = []
    for (scenario, op), rs in sorted(buckets.items()):
        rows.append(
            {
                "scenario": scenario,
                "condition": LABELS.get(scenario, scenario),
                "op": op,
                "count": f"{mean([int(r['count']) for r in rs]):.0f}",
                "duration_ms": f"{mean([r['duration_ms'] for r in rs]):.1f}",
                "bytes_mb": f"{mean([r['bytes_mb'] for r in rs]):.1f}",
            }
        )
    return rows


def md_table(rows: list[dict], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(r.get(f, "")) for f in fields) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def build_report(summary_rows: list[dict], transitions: list[dict], mem_rows: list[dict]) -> None:
    selected_summary = [r for r in summary_rows if r["scenario"] in SELECTED]
    transition_blocks = []
    for scenario in SELECTED:
        transition_blocks.append(f"### {LABELS.get(scenario, scenario)}\n\n" + md_table(top_transition_rows(transitions, scenario), [
            "transition",
            "count",
            "p50_gap_us",
            "p99_gap_us",
            "max_gap_ms",
            "mean_mem_us_inside",
            "mean_mem_fraction",
        ]))
    selected_mem = [r for r in mem_summary(mem_rows) if r["scenario"] in SELECTED]
    md = f"""# NSYS SQLite Deep Root-Cause 분석

생성일: 2026-06-01  
소스: `cloudlab_results/results/20260531/nsys_sqlite_v2/*.sqlite`

## 핵심 발견

기존 분석의 가장 큰 문제는 `kernel-to-kernel gap`을 거의 그대로 scheduling idle처럼 읽었다는 점이다. SQLite 원본을 다시 보면, 긴 kernel gap 상당수는 완전히 빈 시간이 아니라 **kernel 사이에 memcpy/memset activity가 끼어 있는 구간**이다. 따라서 문제 지점은 단순한 GPU idle이 아니라, L1 pipeline의 반복적인 `convert -> copy/memset -> 다음 cuPHY kernel` boundary에서 생기는 temporal gap이다.

이 관점에서는 주장이 더 선명해진다.

- 2g L1은 kernel-only p99 gap이 크게 증가한다. 이것은 small partition이 L1 pipeline의 scheduling headroom을 줄인다는 증거다.
- 3g L1 + NeuralRx는 all-activity busy time과 memory activity가 증가한다. 이것은 NeuralRx가 L1 주변의 copy/memset/runtime boundary를 더 무겁게 만든다는 증거다.
- kernel-only gap p99가 커도 all-activity gap p99는 매우 작을 수 있다. 이 경우는 "GPU가 비어 있었다"가 아니라 "kernel 사이에 memory operation이 끼어 있었다"로 해석해야 한다.

## 1. Kernel-only gap vs all-activity gap

아래 표에서 `kernel_gap_p99_us`는 kernel만 보고 다음 kernel까지의 gap을 계산한 값이다. `all_activity_gap_p99_us`는 kernel, memcpy, memset interval을 모두 합쳐 GPU activity timeline을 만든 뒤 남는 gap이다.

핵심은 두 값의 차이다. kernel-only gap이 수백~천 us인데 all-activity gap은 수 us 수준이면, 그 시간은 진짜 idle이 아니라 memory/copy/set activity가 채우고 있다는 뜻이다.

{md_table(selected_summary, [
    "condition",
    "runs",
    "kernel_busy_pct",
    "all_gpu_busy_pct",
    "kernel_gap_p99_us",
    "all_activity_gap_p99_us",
    "big_kernel_gaps_ge_1ms",
    "big_kernel_gaps_with_mem_pct",
    "mean_mem_fraction_inside_big_gaps",
])}

## 2. 긴 gap은 어느 transition에서 생기는가

아래 표들은 각 조건에서 500us 이상 gap이 생기는 transition을 p99 gap 기준으로 정렬한 것이다. 반복적으로 보이는 문제 transition은 `convert_kernel -> ch_est_pre`, `convert_kernel -> noise_intf_est`, `convert_kernel -> eq_coef`, `copy_float32 -> convert_kernel`, `copy_complex64 -> convert_kernel` 계열이다.

이것은 L1의 특정 PHY compute kernel 하나가 오래 걸린다는 뜻이 아니다. 오히려 `convert/copy` 이후 다음 PHY stage로 넘어가는 boundary에서 tail이 생긴다. AI-RAN 관점에서는 이 boundary가 frame pipeline의 fragile point다.

{chr(10).join(transition_blocks)}

## 3. Memory operation 자체가 workload별로 다르다

아래 표는 selected condition에서 Device-to-Device memcpy와 memset을 요약한 것이다. NeuralRx, ResNet, ChanPred 조합은 copy/set duration과 bytes pattern이 다르다. 그래서 generic D2D/H2D saturation으로는 실제 PHY-AI의 temporal pattern을 완전히 재현하기 어렵다.

{md_table(selected_mem, ["condition", "op", "count", "duration_ms", "bytes_mb"])}

## 4. 수정된 해석

NSYS에서 진짜로 말할 수 있는 것은 다음이다.

1. **문제 지점은 raw HBM bandwidth 하나가 아니다.** Kernel-only gap과 all-activity gap이 크게 다르기 때문에, 긴 gap은 대부분 GPU가 완전히 노는 시간이 아니라 copy/memset/runtime activity가 L1 kernel 사이에 끼어든 결과다.

2. **L1 pipeline의 취약 지점은 convert/copy boundary다.** `convert_kernel` 이후 channel estimation, noise/interference estimation, equalization으로 넘어가는 transition에서 p99 gap이 커진다. 이 boundary가 AI co-tenant와 partition size 변화에 민감하다.

3. **2g L1은 scheduling headroom이 부족하다.** 2g 조건은 L1 alone에서도 kernel gap p99가 3g보다 크다. AI workload가 없어도 작은 partition 자체가 위험한 출발점이다.

4. **NeuralRx는 단순 compute stress가 아니다.** NeuralRx 조건은 memory activity와 runtime/API boundary를 더 무겁게 만든다. 그래서 generic D2D/H2D/GEMM stress가 안전하다는 6/1 F 결과와 NeuralRx/coloc failure는 모순이 아니다.

5. **논문 문장은 이렇게 좁혀야 한다.** "MIG는 bandwidth isolation이 완전히 안 된다"가 아니라, "MIG는 static capacity isolation을 제공하지만, AI-RAN L1 pipeline의 convert/copy/runtime boundary에서 필요한 temporal scheduling guarantee를 제공하지 못한다"가 정확하다.
"""
    (OUT / "NSYS_DEEP_ROOTCAUSE_KR.md").write_text(md)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, transitions, mem_rows = aggregate()
    summary_rows = summarize_by_scenario(summaries)
    write_csv(
        OUT / "kernel_vs_all_activity_gap_summary.csv",
        summary_rows,
        [
            "scenario",
            "condition",
            "runs",
            "wall_ms",
            "kernel_busy_pct",
            "all_gpu_busy_pct",
            "kernel_gap_p99_us",
            "all_activity_gap_p99_us",
            "big_kernel_gaps_ge_1ms",
            "big_kernel_gaps_with_mem_pct",
            "mean_mem_fraction_inside_big_gaps",
        ],
    )
    transition_rows = []
    for s in sorted({t["scenario"] for t in transitions}):
        transition_rows.extend(top_transition_rows(transitions, s, min_gap_us=500.0))
    write_csv(
        OUT / "top_long_gap_transitions.csv",
        transition_rows,
        [
            "condition",
            "transition",
            "count",
            "p50_gap_us",
            "p99_gap_us",
            "max_gap_ms",
            "mean_mem_us_inside",
            "mean_mem_fraction",
        ],
    )
    write_csv(OUT / "memory_activity_by_condition.csv", mem_summary(mem_rows), ["scenario", "condition", "op", "count", "duration_ms", "bytes_mb"])
    build_report(summary_rows, transitions, mem_rows)
    print(f"wrote {OUT / 'NSYS_DEEP_ROOTCAUSE_KR.md'}")


if __name__ == "__main__":
    main()
