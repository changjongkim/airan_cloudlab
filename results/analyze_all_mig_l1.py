#!/usr/bin/env python3
"""Aggregate 20260531 + 20260601 MIG/L1 experiment artifacts.

The goal is not to replace the hand-written analysis, but to make omissions
visible: every experiment directory, JSON L1 capture, and known summary CSV is
indexed into a small set of master tables.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "all_deep_dive"
DAY_DIRS = [ROOT / "20260531", ROOT / "20260601"]


def pct(v: float, b: float) -> float:
    return (v / b - 1.0) * 100.0 if b else 0.0


def mean(vals: list[float]) -> float:
    return stats.mean(vals) if vals else 0.0


def sd(vals: list[float]) -> float:
    return stats.pstdev(vals) if len(vals) > 1 else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def day_of(path: Path) -> str:
    for d in DAY_DIRS:
        try:
            path.relative_to(d)
            return d.name
        except ValueError:
            pass
    return ""


def top_group(path: Path) -> str:
    day = ROOT / day_of(path)
    try:
        rel = path.relative_to(day)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def inventory() -> list[dict]:
    rows = []
    for day in DAY_DIRS:
        groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for f in day.rglob("*"):
            if not f.is_file():
                continue
            g = top_group(f)
            groups[g]["files"] += 1
            suffix = f.suffix.lower().lstrip(".") or "no_ext"
            groups[g][suffix] += 1
            if "traceback" in f.read_text(errors="ignore").lower() if f.suffix.lower() in {".log", ".txt", ".md"} else False:
                groups[g]["traceback_files"] += 1
        for g, counts in sorted(groups.items()):
            row = {"day": day.name, "group": g}
            row.update(counts)
            rows.append(row)
    fields = [
        "day",
        "group",
        "files",
        "json",
        "log",
        "csv",
        "md",
        "txt",
        "png",
        "rep",
        "sqlite",
        "py",
        "traceback_files",
    ]
    write_csv(OUT / "artifact_inventory.csv", rows, fields)
    return rows


def load_l1_jsons() -> list[dict]:
    rows = []
    for day in DAY_DIRS:
        for f in sorted(day.rglob("*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            if "mean_ms" not in data or "p99_ms" not in data:
                continue
            label = data.get("label", f.stem)
            raw = data.get("raw_ms") or []
            rows.append(
                {
                    "day": day.name,
                    "group": top_group(f),
                    "parent": f.parent.name,
                    "file": str(f.relative_to(ROOT)),
                    "label": label,
                    "condition": condition_from_label(day.name, top_group(f), f.parent.name, str(label)),
                    "iterations": data.get("iterations", ""),
                    "num_cells": data.get("num_cells", ""),
                    "mean_ms": float(data.get("mean_ms", 0)),
                    "p50_ms": float(data.get("p50_ms", 0)),
                    "p95_ms": float(data.get("p95_ms", 0)),
                    "p99_ms": float(data.get("p99_ms", 0)),
                    "min_ms": float(data.get("min_ms", 0)),
                    "max_ms": float(data.get("max_ms", 0)),
                    "raw_n": len(raw),
                }
            )
    fields = [
        "day",
        "group",
        "parent",
        "file",
        "label",
        "condition",
        "iterations",
        "num_cells",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "raw_n",
    ]
    write_csv(OUT / "l1_json_runs.csv", rows, fields)
    return rows


def condition_from_label(day: str, group: str, parent: str, label: str) -> str:
    """Return the experiment condition, stripping run suffixes where labels carry them."""
    if day == "20260601" and group in {"F_saturation", "G_coloc", "H_dual"}:
        return re.sub(r"_run\d+$", "", label)
    return parent


def summarize_l1(rows: list[dict], log_rows: list[dict] | None = None) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["day"], r["group"], r["condition"])].append(r)
    json_keys = set(buckets.keys())
    if log_rows:
        for r in log_rows:
            key = (r["day"], r["group"], r["condition"])
            # Prefer JSON when both exist. Use logs for log-only baselines and P4/P5/P7.
            if key not in json_keys:
                buckets[key].append(r)
    out = []
    for (day, group, condition), rs in sorted(buckets.items()):
        means = [r["mean_ms"] for r in rs]
        p99s = [r["p99_ms"] for r in rs]
        maxs = [r.get("max_ms", r["p99_ms"]) for r in rs]
        out.append(
            {
                "day": day,
                "group": group,
                "condition": condition,
                "n": len(rs),
                "mean_ms": f"{mean(means):.3f}",
                "mean_sd": f"{sd(means):.3f}",
                "p99_ms": f"{mean(p99s):.3f}",
                "p99_sd": f"{sd(p99s):.3f}",
                "max_ms": f"{mean(maxs):.3f}",
                "max_sd": f"{sd(maxs):.3f}",
            }
        )
    fields = ["day", "group", "condition", "n", "mean_ms", "mean_sd", "p99_ms", "p99_sd", "max_ms", "max_sd"]
    write_csv(OUT / "l1_condition_summary.csv", out, fields)
    return out


def parse_real_l1_logs() -> list[dict]:
    pat = re.compile(r"\[realL1\]\s+([^\n:]+):\s+mean=([0-9.]+)ms\s+p95=([0-9.]+)ms\s+p99=([0-9.]+)ms")
    rows = []
    for day in DAY_DIRS:
        for f in sorted(day.rglob("*.log")):
            text = f.read_text(errors="ignore")
            for m in pat.finditer(text):
                label = m.group(1)
                rows.append(
                    {
                        "day": day.name,
                        "group": top_group(f),
                        "parent": f.parent.name,
                        "file": str(f.relative_to(ROOT)),
                        "label": label,
                        "condition": condition_from_label(day.name, top_group(f), f.parent.name, label),
                        "mean_ms": float(m.group(2)),
                        "p95_ms": float(m.group(3)),
                        "p99_ms": float(m.group(4)),
                    }
                )
    fields = ["day", "group", "parent", "file", "label", "condition", "mean_ms", "p95_ms", "p99_ms"]
    write_csv(OUT / "l1_log_runs.csv", rows, fields)
    return rows


def load_summary_csvs() -> dict[str, list[dict]]:
    wanted = {
        "F_summary": ROOT / "20260601" / "analysis_F" / "F_summary.csv",
        "G_summary": ROOT / "20260601" / "analysis_G" / "G_summary.csv",
        "Dv2_summary": ROOT / "20260531" / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv",
        "A_paper_table": ROOT / "20260531" / "nsys_deep_A_analysis" / "paper_table.csv",
    }
    loaded: dict[str, list[dict]] = {}
    for name, path in wanted.items():
        if not path.exists():
            loaded[name] = []
            continue
        with path.open(newline="") as f:
            loaded[name] = list(csv.DictReader(f))
    return loaded


def build_markdown(
    inv: list[dict],
    l1_summary: list[dict],
    summaries: dict[str, list[dict]],
) -> None:
    inv_totals = defaultdict(int)
    for r in inv:
        inv_totals[r["day"]] += int(r.get("files") or 0)

    def cond(name: str, day: str | None = None) -> list[dict]:
        return [r for r in l1_summary if r["condition"] == name and (day is None or r["day"] == day)]

    f_rows = summaries.get("F_summary", [])
    g_rows = summaries.get("G_summary", [])
    dv2_rows = summaries.get("Dv2_summary", [])
    a_rows = summaries.get("A_paper_table", [])

    f_baseline = next((r for r in f_rows if r["condition"] == "F_0_alone"), None)
    f_inflated = [
        r
        for r in f_rows
        if r.get("condition") != "F_0_alone" and float(r.get("p99_delta_pct", 0) or 0) > 0
    ]
    g_baselines = {r["condition"]: r for r in g_rows if r["condition"].startswith("G_0")}

    lines: list[str] = []
    lines.append("# MIG L1 Deep Dive Across All 20260531-20260601 Experiments")
    lines.append("")
    lines.append("Generated from local artifacts by `cloudlab_results/results/analyze_all_mig_l1.py`.")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(f"- 20260531 files indexed: {inv_totals['20260531']}")
    lines.append(f"- 20260601 files indexed: {inv_totals['20260601']}")
    lines.append("- Master tables generated in `cloudlab_results/results/all_deep_dive/`:")
    lines.append("  - `artifact_inventory.csv`")
    lines.append("  - `l1_json_runs.csv`")
    lines.append("  - `l1_log_runs.csv`")
    lines.append("  - `l1_condition_summary.csv`")
    lines.append("")
    lines.append("## 2. Revised Verdict")
    lines.append("")
    lines.append(
        "The earlier 20260531-only interpretation over-weighted small-N cross-partition tail events. "
        "The full dataset changes the conclusion: MIG cross-partition saturation is mostly isolated, "
        "while same-partition co-location of cuPHY L1 with NeuralRx is catastrophically bad."
    )
    lines.append("")
    lines.append("| Question | Full-data answer | Evidence |")
    lines.append("|---|---|---|")
    lines.append(
        "| Does cross-partition bandwidth saturation break L1? | Mostly no | 20260601 F: 40 saturation conditions, 0 positive p99 inflation cases |"
    )
    lines.append(
        "| Does small MIG partition hurt L1 alone? | Yes | 20260531/20260601 baselines: 2g has clearly worse mean/p99 than 3g/4g |"
    )
    lines.append(
        "| Does same-partition PHY-AI coloc break L1? | Yes, massively | 20260601 G: p99 +373% to +537% |"
    )
    lines.append(
        "| Is the problem continuous raw HBM GB/s contention? | Not supported | F D2D/H2D/GEMM/stack/kitchen conditions do not inflate p99 |"
    )
    lines.append(
        "| What remains the real-time risk? | Temporal sharing inside one partition, plus static partition headroom | G/H bimodal coloc and 2g baseline penalty |"
    )
    lines.append("")
    lines.append("## 3. 20260531: Initial Broad Sweep")
    lines.append("")
    lines.append("### 3.1 Tier1 baselines and phase experiments")
    lines.append("")
    selected_31_names = [
        "n20_baseline_fullGPU_v2",
        "n20_baseline_7g_single",
        "n20_baseline_4g_alone",
        "n20_baseline_3g_alone",
        "n20_baseline_2g_alone",
        "n20_phase1_qwen_small",
        "n20_phase4_neuralrx",
        "n20_phase4_chanpred",
        "n20_phase4_xapp",
    ]
    selected_31 = []
    for name in selected_31_names:
        selected_31.extend(cond(name, "20260531"))
    lines.append("| Condition | N | Mean ms | p99 ms |")
    lines.append("|---|---:|---:|---:|")
    for r in selected_31:
        lines.append(f"| {r['condition']} | {r['n']} | {r['mean_ms']} | {r['p99_ms']} |")
    lines.append("")
    lines.append(
        "Interpretation: these early runs established that small partitions are risky and that some "
        "AI co-tenant configurations correlate with large L1 tail latency. However, later n=10/n=5 "
        "experiments show that not all of this should be attributed to cross-partition bandwidth contention."
    )
    lines.append("")
    lines.append("### 3.2 Dv2 n=10 phase decomposition")
    lines.append("")
    lines.append("| Scenario | N | p99 mean us | p99 CI | p999 mean us | max mean us |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in dv2_rows:
        lines.append(
            f"| {r['scenario']} | {r['n']} | {float(r['p99_mean']):.1f} | "
            f"{float(r['p99_ci_lo']):.1f}-{float(r['p99_ci_hi']):.1f} | "
            f"{float(r['p999_mean']):.1f} | {float(r['max_mean']):.1f} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: Dv2 weakens the simple bandwidth-contention story. H2D/D2D/compute/launch/chanpred "
        "all overlap or sit near the alone baseline in p99. Rare max outliers exist, but the central p99 result "
        "does not support a monotonic cross-partition saturation effect."
    )
    lines.append("")
    lines.append("### 3.3 Stage A wall-clock alignment")
    lines.append("")
    lines.append("| Scenario | L1 kernels | p99 gap us | p999 gap us | max gap ms | top1 idle share |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in a_rows:
        lines.append(
            f"| {r['scenario']} | {r['n_l1_kernels']} | {r['p99_gap_us']} | "
            f"{r['p999_gap_us']} | {r['max_gap_ms']} | {r['top1_pct_idle_share']} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: Stage A remains useful as a burst-localization study, but it should not by itself be used "
        "as final proof of cross-partition bandwidth failure because later controlled F/Dv2 sweeps do not reproduce "
        "positive p99 inflation under saturation."
    )
    lines.append("")
    lines.append("## 4. 20260601 F: Cross-Partition Saturation Matrix")
    lines.append("")
    if f_baseline:
        lines.append(
            f"Baseline F_0_alone: N={f_baseline['n']}, mean={float(f_baseline['mean_ms']):.2f} ms, "
            f"p99={float(f_baseline['p99_ms']):.2f} ms."
        )
    lines.append("")
    block_stats: dict[str, list[float]] = defaultdict(list)
    worst_by_block: dict[str, dict] = {}
    for r in f_rows:
        if r["condition"] == "F_0_alone":
            continue
        block = r["block"]
        delta = float(r["p99_delta_pct"])
        block_stats[block].append(delta)
        if block not in worst_by_block or delta > float(worst_by_block[block]["p99_delta_pct"]):
            worst_by_block[block] = r
    lines.append("| Block | Conditions | Mean p99 delta | Worst p99 delta | Worst condition |")
    lines.append("|---|---:|---:|---:|---|")
    for block in sorted(block_stats):
        vals = block_stats[block]
        worst = worst_by_block[block]
        lines.append(
            f"| {block} | {len(vals)} | {mean(vals):+.1f}% | "
            f"{float(worst['p99_delta_pct']):+.1f}% | {worst['condition']} |"
        )
    lines.append("")
    lines.append(f"Positive p99 inflation count: {len(f_inflated)} / {max(len(f_rows)-1, 0)}.")
    lines.append("")
    lines.append(
        "Interpretation: this is the strongest evidence against the naive fixed-HBM-bandwidth contention hypothesis. "
        "Even aggressive D2D, H2D, GEMM, workload stacking, and kitchen-sink stressors in other MIG partitions did not "
        "increase L1 p99 above baseline."
    )
    lines.append("")
    lines.append("## 5. 20260601 G: Same-Partition NeuralRx Co-Location")
    lines.append("")
    lines.append("| Condition | N | Mean ms | p99 ms | Relevant baseline | p99 delta |")
    lines.append("|---|---:|---:|---:|---|---:|")
    base_map = {
        "G_1a_3g_coloc": "G_0a_3g_alone",
        "G_1b_4g_coloc": "G_0b_4g_alone",
        "G_1c_2g_coloc": "G_0c_2g_alone",
        "G_5_4gColoc_chanpred": "G_0b_4g_alone",
        "G_6_2gColoc_chanpred_3g": "G_0c_2g_alone",
    }
    for r in g_rows:
        cond = r["condition"]
        if cond.startswith("G_0"):
            continue
        base_name = base_map.get(cond, "G_1a_3g_coloc")
        base = next((x for x in g_rows if x["condition"] == base_name), None)
        delta = pct(float(r["p99_ms"]), float(base["p99_ms"])) if base else 0.0
        lines.append(
            f"| {cond} | {r['n']} | {float(r['mean_ms']):.2f} | {float(r['p99_ms']):.2f} | "
            f"{base_name} | {delta:+.1f}% |"
        )
    lines.append("")
    lines.append(
        "Interpretation: G is the decisive dataset. Same-partition L1+NeuralRx co-location causes catastrophic p99 "
        "inflation. External AI type matters much less once coloc is active; the coloc condition itself dominates."
    )
    lines.append("")
    lines.append("## 6. 20260601 H: Dual-Concurrent Sanity Captures")
    lines.append("")
    h_rows = [r for r in l1_summary if r["day"] == "20260601" and r["group"] == "H_dual"]
    lines.append("| Condition | N | Mean ms | p99 ms | Max ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in h_rows:
        lines.append(f"| {r['condition']} | {r['n']} | {r['mean_ms']} | {r['p99_ms']} | {r['max_ms']} |")
    lines.append("")
    lines.append(
        "Interpretation: H agrees with F/G: cross-partition saturation captures remain near baseline, while G coloc cases "
        "show bimodal behavior where median-like behavior can look normal but p99/max become catastrophic."
    )
    lines.append("")
    lines.append("## 7. What Should Be Changed In The Paper Story")
    lines.append("")
    lines.append("1. Do not claim that cross-partition MIG bandwidth isolation generally fails. The full F/Dv2 data do not support that.")
    lines.append("2. Keep the partition-fragmentation claim: small L1 partitions have worse standalone headroom.")
    lines.append("3. Shift the main failure mode to same-partition temporal sharing: L1 + NeuralRx coloc creates massive bimodal tails.")
    lines.append("4. Treat early 5/31 cross-partition p99 spikes as exploratory observations requiring statistical qualification.")
    lines.append("5. Reframe bandwidth carefully: raw cross-partition HBM bandwidth is not the main culprit; in-partition SM/memory time-sharing and burst occupancy are.")
    lines.append("")
    lines.append("## 8. Open Analysis Gaps")
    lines.append("")
    lines.append("- Export H dual nsys traces to SQLite and align NeuralRx kernels against L1 catastrophic frames.")
    lines.append("- Re-run or finish I_ncu/J_mps if needed; current I files exist but need metric-level parsing and J has no obvious summaries.")
    lines.append("- Build figures for the revised final story: F negative matrix, G coloc p99 explosion, H bimodal raw distribution, partition baseline headroom.")
    lines.append("- Audit old `MIG_L1_ISOLATION_SYNTHESIS.md` language so it does not overclaim cross-partition bandwidth failure.")
    lines.append("")
    (OUT / "MIG_L1_ALL_EXPERIMENTS_DEEP_DIVE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    json_rows = load_l1_jsons()
    log_rows = parse_real_l1_logs()
    l1_summary = summarize_l1(json_rows, log_rows)
    summaries = load_summary_csvs()
    build_markdown(inv, l1_summary, summaries)
    print(f"Wrote {OUT}")
    print(f"Indexed JSON L1 runs: {len(json_rows)}")
    print(f"Indexed log L1 runs: {len(log_rows)}")


if __name__ == "__main__":
    main()
