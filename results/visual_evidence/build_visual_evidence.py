#!/usr/bin/env python3
"""Build visual evidence pack for the MIG/AI-RAN argument.

The script intentionally avoids pandas so the notebook can run in the current
CloudLab result environment with only Python stdlib + matplotlib.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics as stats
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DATA = OUT / "data"
SUMMARY = ROOT / "all_deep_dive" / "l1_condition_summary.csv"
F_SUMMARY = ROOT / "20260601" / "analysis_F" / "F_summary.csv"
G_SUMMARY = ROOT / "20260601" / "analysis_G" / "G_summary.csv"
L1_LOG_RUNS = ROOT / "all_deep_dive" / "l1_log_runs.csv"
DV2_SUMMARY = ROOT / "20260531" / "nsys_deep_Dv2_analysis" / "Dv2_summary.csv"
NSYS_A_TABLE = ROOT / "20260531" / "nsys_deep_A_analysis" / "paper_table.csv"
MEMORY_OPS = ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "memory_ops_analysis.csv"
NSYS_FAST = OUT / "deep_nsys_fast"
NSYS_KERNEL_ACTIVITY = NSYS_FAST / "kernel_vs_all_activity_summary.csv"
NSYS_MEMORY_ACTIVITY = NSYS_FAST / "memory_activity_summary.csv"
NSYS_TRANSITIONS = NSYS_FAST / "top_gap_transitions_by_condition.csv"


COLORS = {
    "blue": "#2F5F8F",
    "teal": "#2A9D8F",
    "gold": "#D6A21E",
    "red": "#C44536",
    "purple": "#7B5EA7",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def fnum(value: str | float | int) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return stats.mean(vals) if vals else float("nan")


def pct(new: float, base: float) -> float:
    return (new / base - 1.0) * 100.0 if base else float("nan")


def savefig(name: str) -> str:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(OUT))


def l1_summary() -> list[dict[str, str]]:
    return read_csv(SUMMARY)


def find_condition(rows: list[dict[str, str]], condition: str, day: str | None = None) -> dict[str, str]:
    for r in rows:
        if r["condition"] == condition and (day is None or r["day"] == day):
            return r
    raise KeyError(condition)


def plot_partition_baseline(rows: list[dict[str, str]]) -> str:
    items = [
        ("Full GPU", "n20_baseline_fullGPU_v2"),
        ("7g MIG", "n20_baseline_7g_single"),
        ("4g MIG", "n20_baseline_4g_alone"),
        ("3g MIG", "n20_baseline_3g_alone"),
        ("2g MIG", "n20_baseline_2g_alone"),
    ]
    out = []
    base = fnum(find_condition(rows, "n20_baseline_fullGPU_v2")["mean_ms"])
    for label, cond in items:
        r = find_condition(rows, cond)
        out.append(
            {
                "label": label,
                "condition": cond,
                "mean_ms": fnum(r["mean_ms"]),
                "p99_ms": fnum(r["p99_ms"]),
                "mean_delta_pct_vs_full": pct(fnum(r["mean_ms"]), base),
            }
        )
    write_csv(DATA / "partition_baseline.csv", out, ["label", "condition", "mean_ms", "p99_ms", "mean_delta_pct_vs_full"])

    labels = [r["label"] for r in out]
    means = [r["mean_ms"] for r in out]
    p99s = [r["p99_ms"] for r in out]
    x = range(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - 0.18 for i in x], means, width=0.36, color=COLORS["blue"], label="mean")
    plt.bar([i + 0.18 for i in x], p99s, width=0.36, color=COLORS["gold"], label="p99")
    plt.axhline(base, color=COLORS["gray"], linestyle="--", linewidth=1, label="Full GPU mean")
    for i, r in enumerate(out):
        if r["label"] == "2g MIG":
            plt.text(i, r["mean_ms"] + 2.0, f"+{r['mean_delta_pct_vs_full']:.1f}% mean", ha="center", color=COLORS["red"], weight="bold")
    plt.xticks(list(x), labels)
    plt.ylabel("L1 frame latency (ms)")
    plt.title("Partition fragmentation reduces L1 headroom")
    plt.legend(ncol=3, frameon=False)
    return savefig("fig01_partition_baseline.png")


def plot_phase4_neuralrx(rows: list[dict[str, str]]) -> str:
    baseline = fnum(find_condition(rows, "n20_baseline_3g_alone")["p99_ms"])
    items = [
        ("3g alone", "n20_baseline_3g_alone"),
        ("Qwen-small", "n20_phase1_qwen_small"),
        ("ChanPred", "n20_phase4_chanpred"),
        ("XApp", "n20_phase4_xapp"),
        ("NeuralRx", "n20_phase4_neuralrx"),
    ]
    out = []
    for label, cond in items:
        r = find_condition(rows, cond)
        out.append({"label": label, "condition": cond, "p99_ms": fnum(r["p99_ms"]), "p99_delta_pct_vs_3g_alone": pct(fnum(r["p99_ms"]), baseline)})
    write_csv(DATA / "phase4_phy_ai_p99.csv", out, ["label", "condition", "p99_ms", "p99_delta_pct_vs_3g_alone"])

    labels = [r["label"] for r in out]
    vals = [r["p99_ms"] for r in out]
    colors = [COLORS["gray"], COLORS["teal"], COLORS["teal"], COLORS["teal"], COLORS["red"]]
    plt.figure(figsize=(8, 4.4))
    plt.bar(labels, vals, color=colors)
    plt.axhline(baseline, color=COLORS["gray"], linestyle="--", linewidth=1)
    for i, r in enumerate(out):
        if r["label"] != "3g alone":
            plt.text(i, r["p99_ms"] + 4, f"+{r['p99_delta_pct_vs_3g_alone']:.0f}%", ha="center", weight="bold" if r["label"] == "NeuralRx" else "normal")
    plt.ylabel("L1 p99 latency (ms)")
    plt.title("PHY-AI co-tenant risk: NeuralRx is not a generic AI")
    return savefig("fig02_phase4_neuralrx_risk.png")


def plot_f_saturation() -> str:
    rows = read_csv(F_SUMMARY)
    non_base = [r for r in rows if r["condition"] != "F_0_alone"]
    blocks: dict[str, list[float]] = {}
    for r in non_base:
        blocks.setdefault(r["block"], []).append(fnum(r["p99_delta_pct"]))
    out = []
    for block, vals in sorted(blocks.items()):
        out.append({"block": block, "n": len(vals), "mean_p99_delta_pct": mean(vals), "max_p99_delta_pct": max(vals)})
    write_csv(DATA / "f_saturation_block_summary.csv", out, ["block", "n", "mean_p99_delta_pct", "max_p99_delta_pct"])

    labels = [r["block"].replace("_", "\n") for r in out]
    mean_vals = [r["mean_p99_delta_pct"] for r in out]
    max_vals = [r["max_p99_delta_pct"] for r in out]
    x = range(len(labels))
    plt.figure(figsize=(9.5, 4.6))
    plt.bar([i - 0.18 for i in x], mean_vals, width=0.36, color=COLORS["blue"], label="block mean")
    plt.bar([i + 0.18 for i in x], max_vals, width=0.36, color=COLORS["teal"], label="block max")
    plt.axhline(0, color=COLORS["red"], linewidth=1.2)
    plt.xticks(list(x), labels)
    plt.ylabel("L1 p99 delta vs F_0_alone (%)")
    plt.title("Generic cross-partition saturation did not inflate L1 p99")
    plt.legend(frameon=False)
    plt.text(len(labels) - 1.2, 1.5, "0 / 39 positive p99 deltas", color=COLORS["red"], weight="bold", ha="right")
    return savefig("fig03_f_saturation_negative_result.png")


def plot_g_coloc() -> str:
    rows = read_csv(G_SUMMARY)
    by_cond = {r["condition"]: r for r in rows}
    pairs = [
        ("3g", "G_0a_3g_alone", "G_1a_3g_coloc"),
        ("4g", "G_0b_4g_alone", "G_1b_4g_coloc"),
        ("2g", "G_0c_2g_alone", "G_1c_2g_coloc"),
    ]
    out = []
    for part, alone, coloc in pairs:
        a = fnum(by_cond[alone]["p99_ms"])
        c = fnum(by_cond[coloc]["p99_ms"])
        out.append({"partition": part, "alone_p99_ms": a, "coloc_p99_ms": c, "p99_delta_pct": pct(c, a)})
    write_csv(DATA / "g_coloc_l1_p99.csv", out, ["partition", "alone_p99_ms", "coloc_p99_ms", "p99_delta_pct"])

    labels = [r["partition"] for r in out]
    x = range(len(labels))
    plt.figure(figsize=(7.5, 4.6))
    plt.bar([i - 0.18 for i in x], [r["alone_p99_ms"] for r in out], width=0.36, color=COLORS["gray"], label="L1 alone")
    plt.bar([i + 0.18 for i in x], [r["coloc_p99_ms"] for r in out], width=0.36, color=COLORS["red"], label="L1 + NeuralRx coloc")
    for i, r in enumerate(out):
        plt.text(i + 0.18, r["coloc_p99_ms"] + 12, f"+{r['p99_delta_pct']:.0f}%", ha="center", weight="bold", color=COLORS["red"])
    plt.xticks(list(x), labels)
    plt.ylabel("L1 p99 latency (ms)")
    plt.title("Same-partition PHY-AI co-location is catastrophic")
    plt.legend(frameon=False)
    return savefig("fig04_g_coloc_explosion.png")


def plot_h_dual(rows: list[dict[str, str]]) -> str:
    order = [
        ("Baseline", "H_baseline_3g_alone_l1"),
        ("D2D", "H_F1_D2D_1024MB_str8_l1"),
        ("GEMM", "H_F3_GEMM_4096_l1"),
        ("Stack4\nChanPred", "H_F5_stack4_chanpred_l1"),
        ("Kitchen", "H_F_kitchen_l1"),
        ("3g coloc\n+ ext", "H_G1_3gColoc_chanpred_l1"),
        ("2g coloc\n+ ext", "H_G2_2gColoc_chanpred_3g_l1"),
    ]
    out = []
    base = fnum(find_condition(rows, "H_baseline_3g_alone_l1", "20260601")["p99_ms"])
    for label, cond in order:
        r = find_condition(rows, cond, "20260601")
        out.append({"label": label, "condition": cond, "p99_ms": fnum(r["p99_ms"]), "p99_delta_pct": pct(fnum(r["p99_ms"]), base)})
    write_csv(DATA / "h_dual_p99.csv", out, ["label", "condition", "p99_ms", "p99_delta_pct"])

    colors = [COLORS["gray"], COLORS["teal"], COLORS["teal"], COLORS["teal"], COLORS["teal"], COLORS["red"], COLORS["red"]]
    plt.figure(figsize=(9, 4.6))
    plt.bar([r["label"] for r in out], [r["p99_ms"] for r in out], color=colors)
    plt.axhline(base, color=COLORS["gray"], linestyle="--", linewidth=1)
    plt.ylabel("L1 p99 latency (ms)")
    plt.title("H sanity check: external stress safe, coloc unsafe")
    plt.text(3.1, base + 10, "external stress stays near baseline", color=COLORS["teal"], ha="center")
    plt.text(5.5, 330, "coloc blows up", color=COLORS["red"], ha="center", weight="bold")
    return savefig("fig05_h_dual_sanity.png")


def parse_ai_throughput() -> list[dict]:
    roots = [ROOT / "20260531" / "ai_full_matrix", ROOT / "20260531" / "ai_supplement"]
    rows = []
    patterns = [
        ("tflops", re.compile(r"tflops=([0-9.]+)")),
        ("gbps", re.compile(r"bw=([0-9.]+)GB/s")),
        ("img_s", re.compile(r"\(([0-9.]+) img/s\)")),
        ("batch_s", re.compile(r"([0-9.]+) batch/s")),
        ("it_s", re.compile(r"([0-9.]+) it/s")),
        ("inf_s", re.compile(r"\(([0-9.]+) inf/s")),
        ("pred_s", re.compile(r"\(([0-9.]+) pred/s")),
    ]
    for root in roots:
        if not root.exists():
            continue
        for log in root.glob("*/alone/run_*.log"):
            text = log.read_text(errors="ignore")
            parent = log.parents[1].name
            m = re.search(r"(.+)_([1-4]g)(?:_bs([0-9]+))?$", parent)
            if not m:
                continue
            workload, part, bs = m.group(1), m.group(2), m.group(3) or ""
            if "Traceback" in text or "CUDA out of memory" in text:
                rows.append({"workload": workload, "partition": part, "batch_size": bs, "metric": "fit", "value": 0.0, "status": "failed"})
                continue
            for metric, pat in patterns:
                mm = pat.search(text)
                if mm:
                    rows.append({"workload": workload, "partition": part, "batch_size": bs, "metric": metric, "value": fnum(mm.group(1)), "status": "ok"})
                    break
    write_csv(DATA / "ai_throughput_parsed.csv", rows, ["workload", "partition", "batch_size", "metric", "value", "status"])
    return rows


def plot_ai_partition_scaling(rows: list[dict]) -> str:
    def val(workload: str, part: str, metric: str, bs: str = "") -> float:
        vals = [
            fnum(r["value"])
            for r in rows
            if r["workload"] == workload and r["partition"] == part and r["metric"] == metric and (not bs or r["batch_size"] == bs)
        ]
        return mean(vals)

    series = [
        ("sat_compute TFLOPS", "sat_compute", "tflops", "", "TFLOPS"),
        ("sat_hbm GB/s", "sat_hbm", "gbps", "", "GB/s"),
        ("ResNet bs64 img/s", "resnet", "img_s", "64", "img/s"),
        ("Forecaster bs64 batch/s", "forecaster", "batch_s", "64", "batch/s"),
    ]
    parts = ["1g", "2g", "3g", "4g"]
    out = []
    plt.figure(figsize=(9, 5))
    for idx, (label, workload, metric, bs, unit) in enumerate(series):
        vals = [val(workload, p, metric, bs) for p in parts]
        base = vals[0] if vals[0] and not math.isnan(vals[0]) else 1.0
        normalized = [v / base if not math.isnan(v) else float("nan") for v in vals]
        for p, raw, norm in zip(parts, vals, normalized):
            out.append({"series": label, "workload": workload, "partition": p, "metric": metric, "unit": unit, "raw_value": raw, "normalized_vs_1g": norm})
        plt.plot(parts, normalized, marker="o", linewidth=2, label=label)
    write_csv(DATA / "ai_partition_scaling.csv", out, ["series", "workload", "partition", "metric", "unit", "raw_value", "normalized_vs_1g"])
    plt.ylabel("Normalized throughput vs 1g")
    plt.xlabel("MIG partition")
    plt.title("AI workloads also depend on partition size")
    plt.legend(frameon=False)
    return savefig("fig06_ai_partition_scaling.png")


def parse_ai_latency() -> list[dict]:
    rows = []
    for root_name in ["ai_per_op_latency", "ai_per_op_latency_b"]:
        root = ROOT / "20260531" / root_name
        if not root.exists():
            continue
        for log in root.glob("*/*/run_*.log"):
            text = log.read_text(errors="ignore")
            js = re.search(r"-json\]\s+({.*})", text)
            if not js:
                continue
            try:
                data = json.loads(js.group(1))
            except Exception:
                continue
            group = log.parents[1].name
            mm = re.search(r"(.+)_([1-4]g)$", group)
            if not mm:
                continue
            rows.append(
                {
                    "workload": mm.group(1),
                    "partition": mm.group(2),
                    "mode": log.parent.name,
                    "mean_ms": data.get("mean_ms", ""),
                    "p99_ms": data.get("p99_ms", ""),
                    "n_calls": data.get("n_calls", ""),
                }
            )
    write_csv(DATA / "ai_per_op_latency_parsed.csv", rows, ["workload", "partition", "mode", "mean_ms", "p99_ms", "n_calls"])
    return rows


def plot_ai_per_op_p99(rows: list[dict]) -> str:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        buckets.setdefault((r["workload"], r["partition"], r["mode"]), []).append(fnum(r["p99_ms"]))
    out = []
    for workload in sorted({r["workload"] for r in rows}):
        for part in ["1g", "2g", "3g", "4g"]:
            a = mean(buckets.get((workload, part, "alone"), []))
            w = mean(buckets.get((workload, part, "with_l1"), []))
            if math.isnan(a) or math.isnan(w):
                continue
            out.append({"workload": workload, "partition": part, "alone_p99_ms": a, "with_l1_p99_ms": w, "p99_delta_pct": pct(w, a)})
    write_csv(DATA / "ai_per_op_p99_delta.csv", out, ["workload", "partition", "alone_p99_ms", "with_l1_p99_ms", "p99_delta_pct"])

    selected = [r for r in out if r["workload"] in {"chanpred", "neuralrx", "qwen", "resnet"}]
    labels = [f"{r['workload']}-{r['partition']}" for r in selected]
    vals = [r["p99_delta_pct"] for r in selected]
    colors = [COLORS["red"] if v > 10 else COLORS["gold"] if v > 3 else COLORS["teal"] for v in vals]
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, vals, color=colors)
    plt.axhline(0, color=COLORS["gray"], linewidth=1)
    plt.axhline(10, color=COLORS["red"], linestyle="--", linewidth=1)
    plt.ylabel("AI per-op p99 delta with L1 background (%)")
    plt.title("AI side is not free: per-operation p99 can inflate")
    plt.xticks(rotation=35, ha="right")
    return savefig("fig07_ai_per_op_p99_delta.png")


def plot_tradeoff_summary() -> str:
    rows = [
        {"case": "L1 small slice", "l1_risk": 40.2, "ai_cost": 0, "label": "2g L1\n+40% mean"},
        {"case": "Separate NeuralRx", "l1_risk": 376.0, "ai_cost": 0, "label": "NeuralRx separate\n+376% L1 p99"},
        {"case": "Same-partition NeuralRx", "l1_risk": 536.7, "ai_cost": 33.0, "label": "coloc\n+537% L1 p99"},
        {"case": "AI small slice", "l1_risk": 0, "ai_cost": 100.0, "label": "Qwen 1g fail\nAI fit risk"},
        {"case": "AI p99 inflation", "l1_risk": 0, "ai_cost": 27.0, "label": "ChanPred\n+27% AI p99"},
    ]
    write_csv(DATA / "tradeoff_summary.csv", rows, ["case", "l1_risk", "ai_cost", "label"])
    plt.figure(figsize=(7, 5.4))
    for r in rows:
        color = COLORS["red"] if r["l1_risk"] > 100 else COLORS["gold"] if r["ai_cost"] > 50 else COLORS["teal"]
        plt.scatter(r["ai_cost"], r["l1_risk"], s=160, color=color, alpha=0.9)
        plt.text(r["ai_cost"] + 3, r["l1_risk"] + 8, r["label"], fontsize=9)
    plt.xlabel("AI service cost / risk (% or fit failure)")
    plt.ylabel("L1 latency risk (%)")
    plt.title("Evidence synthesis: MIG creates a two-sided AI-RAN tradeoff")
    plt.xlim(-5, 125)
    plt.ylim(-20, 590)
    return savefig("fig08_tradeoff_summary.png")


def plot_static_partition_sweep(rows: list[dict[str, str]]) -> str:
    items = [
        ("L1 3g\nQwen-small on 2g + 2g", "n20_phase2_M1_3way_balanced"),
        ("L1 2g\nAI on 3g + 2g", "n20_phase2_M2_3way_L1small"),
        ("L1 4g\nAI on 2g + 1g", "n20_phase2_M3_3way_asym"),
        ("L1 4g\nthree AI workloads on 1g each", "n20_phase2_M4_4way_1L1_3AI"),
        ("L1 2g\nQwen-small on 3g", "n20_phase3_D1_L1_starved"),
        ("L1 ~3g\nAI on 4g", "n20_phase3_D2_L1_boosted"),
    ]
    base = fnum(find_condition(rows, "n20_baseline_3g_alone")["p99_ms"])
    out = []
    for label, cond in items:
        r = find_condition(rows, cond)
        out.append(
            {
                "label": label.replace("\n", " / "),
                "condition": cond,
                "mean_ms": fnum(r["mean_ms"]),
                "p99_ms": fnum(r["p99_ms"]),
                "p99_delta_pct_vs_3g_alone": pct(fnum(r["p99_ms"]), base),
            }
        )
    write_csv(DATA / "static_partition_sweep.csv", out, ["label", "condition", "mean_ms", "p99_ms", "p99_delta_pct_vs_3g_alone"])

    labels = [r["label"].replace(" / ", "\n") for r in out]
    vals = [r["p99_ms"] for r in out]
    colors = [COLORS["red"] if r["label"].startswith("L1 2g") else COLORS["gold"] for r in out]
    plt.figure(figsize=(10.8, 5.4))
    plt.bar(labels, vals, color=colors)
    plt.axhline(base, color=COLORS["gray"], linestyle="--", linewidth=1, label="3g L1 alone baseline")
    for i, r in enumerate(out):
        plt.text(i, r["p99_ms"] + 1.8, f"+{r['p99_delta_pct_vs_3g_alone']:.0f}%", ha="center", fontsize=9)
    plt.ylabel("L1 p99 latency (ms)")
    plt.title("Static partition plans expose the L1-vs-AI capacity tradeoff")
    plt.xticks(rotation=25, ha="right")
    plt.legend(frameon=False)
    return savefig("fig09_static_partition_sweep.png")


def parse_ai_throughput_v2() -> list[dict]:
    root = ROOT / "20260531" / "ai_throughput_v2"
    rows = []
    pats = [
        ("it_s", re.compile(r"([0-9.]+) it/s")),
        ("pred_s", re.compile(r"\(([0-9.]+) pred/s")),
        ("inf_s", re.compile(r"\(([0-9.]+) inf/s")),
    ]
    for log in root.glob("*/*run_*.log"):
        text = log.read_text(errors="ignore")
        group = log.parents[0].name
        if group.endswith("_with_l1"):
            workload = group[: -len("_with_l1")]
            mode = "with_l1"
        elif group.endswith("_alone"):
            workload = group[: -len("_alone")]
            mode = "alone"
        else:
            continue
        for metric, pat in pats:
            m = pat.search(text)
            if m:
                rows.append({"workload": workload, "mode": mode, "metric": metric, "value": fnum(m.group(1))})
                break
    write_csv(DATA / "ai_throughput_v2_parsed.csv", rows, ["workload", "mode", "metric", "value"])
    return rows


def plot_ai_throughput_vs_p99(ai_latency_rows: list[dict]) -> str:
    t_rows = parse_ai_throughput_v2()
    t_buckets: dict[tuple[str, str], list[float]] = {}
    for r in t_rows:
        t_buckets.setdefault((r["workload"], r["mode"]), []).append(fnum(r["value"]))
    throughput_out = []
    for workload in sorted({r["workload"] for r in t_rows}):
        a = mean(t_buckets.get((workload, "alone"), []))
        w = mean(t_buckets.get((workload, "with_l1"), []))
        if not math.isnan(a) and not math.isnan(w):
            throughput_out.append({"workload": workload, "throughput_delta_pct": pct(w, a)})

    p99_rows = read_csv(DATA / "ai_per_op_p99_delta.csv")
    p99_by_workload: dict[str, list[float]] = {}
    for r in p99_rows:
        p99_by_workload.setdefault(r["workload"], []).append(fnum(r["p99_delta_pct"]))
    out = []
    for r in throughput_out:
        workload = "qwen" if r["workload"] == "qwen_small" else r["workload"]
        p99_max = max(p99_by_workload.get(workload, [float("nan")]))
        out.append({"workload": r["workload"], "throughput_delta_pct": r["throughput_delta_pct"], "max_per_op_p99_delta_pct": p99_max})
    write_csv(DATA / "ai_throughput_vs_p99.csv", out, ["workload", "throughput_delta_pct", "max_per_op_p99_delta_pct"])

    labels = [r["workload"].replace("_", "\n") for r in out]
    x = range(len(labels))
    plt.figure(figsize=(8.4, 4.8))
    plt.bar([i - 0.18 for i in x], [r["throughput_delta_pct"] for r in out], width=0.36, color=COLORS["teal"], label="mean throughput delta")
    plt.bar([i + 0.18 for i in x], [r["max_per_op_p99_delta_pct"] for r in out], width=0.36, color=COLORS["red"], label="max per-op p99 delta")
    plt.axhline(0, color=COLORS["gray"], linewidth=1)
    plt.ylabel("Delta with L1 background (%)")
    plt.title("Mean throughput can look stable while AI tail latency moves")
    plt.xticks(list(x), labels)
    plt.legend(frameon=False)
    return savefig("fig10_ai_throughput_vs_p99.png")


def plot_coloc_external_dominance() -> str:
    rows = read_csv(G_SUMMARY)
    label_map = {
        "G_1a_3g_coloc": "L1 3g + NeuralRx\nsame partition",
        "G_2_3gColoc_chanpred": "+ external ChanPred",
        "G_2_3gColoc_forecaster": "+ external Forecaster",
        "G_2_3gColoc_qwen_small": "+ external Qwen-small",
        "G_2_3gColoc_resnet": "+ external ResNet",
        "G_2_3gColoc_sat_compute": "+ external GEMM/sat-compute",
        "G_2_3gColoc_sat_hbm": "+ external HBM saturation",
        "G_2_3gColoc_xapp": "+ external XApp",
        "G_3_3gColoc_het_chanpred_resnet": "+ external ChanPred+ResNet",
        "G_4_3gColoc_homo_2chanpred": "+ external 2x ChanPred",
    }
    wanted = list(label_map)
    out = []
    for r in rows:
        if r["condition"] in label_map:
            out.append({"label": label_map[r["condition"]].replace("\n", " / "), "condition": r["condition"], "p99_ms": fnum(r["p99_ms"])})
    out.sort(key=lambda r: wanted.index(r["condition"]))
    write_csv(DATA / "g_coloc_external_dominance.csv", out, ["label", "condition", "p99_ms"])

    plt.figure(figsize=(11.4, 5.0))
    colors = [COLORS["red"] if i == 0 else COLORS["gold"] for i in range(len(out))]
    plt.bar([r["label"].replace(" / ", "\n") for r in out], [r["p99_ms"] for r in out], color=colors)
    plt.ylabel("L1 p99 latency (ms)")
    plt.title("Once L1 and NeuralRx are colocated, external AI type is secondary")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(220, 390)
    return savefig("fig11_coloc_external_dominance.png")


def plot_nsys_gap_summary() -> str:
    rows = read_csv(NSYS_A_TABLE)
    label_map = {
        "A1_S35_2gL1_chanpred3g": "L1 2g + ChanPred on 3g\nworst small-L1 case",
        "A2_S34_4gL1_resnet2g": "L1 4g + ResNet on 2g\nlarger L1 case",
        "A3_M5c_3gL1": "L1 3g + ResNet+ChanPred\nheterogeneous AI",
        "A4_M8a_3gL1": "L1 3g + ResNet+Forecaster\nheterogeneous AI",
    }
    out = []
    for r in rows:
        out.append(
            {
                "scenario": label_map.get(r["scenario"], r["scenario"]),
                "p99_gap_us": fnum(r["p99_gap_us"]),
                "p999_gap_us": fnum(r["p999_gap_us"]),
                "max_gap_ms": fnum(r["max_gap_ms"]),
                "top1_pct_idle_share": fnum(r["top1_pct_idle_share"].strip("%")),
            }
        )
    write_csv(DATA / "nsys_gap_summary.csv", out, ["scenario", "p99_gap_us", "p999_gap_us", "max_gap_ms", "top1_pct_idle_share"])
    labels = [r["scenario"] for r in out]
    x = range(len(labels))
    plt.figure(figsize=(9, 4.8))
    plt.bar([i - 0.18 for i in x], [r["p99_gap_us"] for r in out], width=0.36, color=COLORS["blue"], label="p99 gap")
    plt.bar([i + 0.18 for i in x], [r["p999_gap_us"] for r in out], width=0.36, color=COLORS["gold"], label="p999 gap")
    plt.ylabel("Inter-kernel gap (us)")
    plt.title("NSYS: long-tail gaps depend on placement and workload mix")
    plt.xticks(list(x), labels)
    plt.legend(frameon=False)
    return savefig("fig12_nsys_gap_summary.png")


def plot_memory_ops_pressure() -> str:
    rows = read_csv(MEMORY_OPS)
    wanted = ["S5_3g_alone", "S6_3g_qwen", "S7_3g_neuralrx", "S27_3g_chanpred", "S28_3g_resnet", "S29_3g_forecaster", "S35_2g_chanpred"]
    label_map = {
        "S5_3g_alone": "L1 3g alone",
        "S6_3g_qwen": "L1 3g + Qwen",
        "S7_3g_neuralrx": "L1 3g + NeuralRx",
        "S27_3g_chanpred": "L1 3g + ChanPred",
        "S28_3g_resnet": "L1 3g + ResNet",
        "S29_3g_forecaster": "L1 3g + Forecaster",
        "S35_2g_chanpred": "L1 2g + ChanPred",
    }
    out = []
    for r in rows:
        if r["scenario"] in wanted:
            out.append({"scenario": label_map[r["scenario"]], "raw_scenario": r["scenario"], "memcpy_total_ms": fnum(r["memcpy_total_us"]) / 1000.0, "memcpy_p99_us": fnum(r["memcpy_p99_us"])})
    write_csv(DATA / "memory_ops_pressure.csv", out, ["scenario", "raw_scenario", "memcpy_total_ms", "memcpy_p99_us"])

    base_total = next(r["memcpy_total_ms"] for r in out if r["raw_scenario"] == "S5_3g_alone")
    base_p99 = next(r["memcpy_p99_us"] for r in out if r["raw_scenario"] == "S5_3g_alone")
    labels = [r["scenario"].replace(" + ", "\n+ ") for r in out]
    x = range(len(labels))
    plt.figure(figsize=(10, 4.8))
    plt.bar([i - 0.18 for i in x], [r["memcpy_total_ms"] / base_total for r in out], width=0.36, color=COLORS["purple"], label="memcpy total / S5")
    plt.bar([i + 0.18 for i in x], [r["memcpy_p99_us"] / base_p99 for r in out], width=0.36, color=COLORS["gold"], label="memcpy p99 / S5")
    plt.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    plt.ylabel("Normalized copy pressure vs S5 3g alone")
    plt.title("NSYS memory ops: copy pressure is workload-specific")
    plt.xticks(list(x), labels)
    plt.legend(frameon=False)
    return savefig("fig13_memory_ops_pressure.png")


def plot_nsys_kernel_vs_activity_gap() -> str:
    rows = read_csv(NSYS_KERNEL_ACTIVITY)
    order = [
        "S2_7g_mig",
        "S5_3g_alone",
        "S7_3g_neuralrx",
        "S10_2g_alone",
        "S35_2g_chanpred",
        "S28_3g_resnet",
        "S31_3g_resnet_chanpred",
        "S32_3g_resnet_forecaster",
        "S34_4g_resnet",
    ]
    by_scenario = {r["scenario"]: r for r in rows}
    out = [by_scenario[s] for s in order if s in by_scenario]
    write_csv(
        DATA / "nsys_kernel_vs_all_activity_summary.csv",
        out,
        [
            "scenario",
            "condition",
            "runs",
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

    labels = [r["condition"].replace(" + ", "\n+ ").replace(" alone", "\nalone").replace(" MIG", "\nMIG") for r in out]
    x = list(range(len(out)))
    kernel_gap = [fnum(r["kernel_gap_p99_us"]) for r in out]
    activity_gap = [fnum(r["all_activity_gap_p99_us"]) for r in out]
    mem_share = [fnum(r["big_gaps_with_mem_pct"]) for r in out]

    fig, ax1 = plt.subplots(figsize=(11.6, 5.2))
    ax1.bar([i - 0.18 for i in x], kernel_gap, width=0.36, color=COLORS["red"], label="kernel-only p99 gap")
    ax1.bar([i + 0.18 for i in x], activity_gap, width=0.36, color=COLORS["blue"], label="all GPU activity p99 gap")
    ax1.set_ylabel("Gap p99 (us)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right")
    ax1.set_title("NSYS SQLite: many long kernel gaps are filled by memcpy/memset")
    ax1.legend(loc="upper left", frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(x, mem_share, color=COLORS["gold"], marker="o", linewidth=2.0, label=">=1ms gaps with memory ops")
    ax2.set_ylabel(">=1ms kernel gaps containing memcpy/memset (%)")
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper right", frameon=False)
    return savefig("fig12_nsys_kernel_vs_activity_gap.png")


def plot_nsys_memory_activity_breakdown() -> str:
    rows = read_csv(NSYS_MEMORY_ACTIVITY)
    order = [
        "L1 7g MIG alone",
        "L1 3g alone",
        "L1 3g + NeuralRx",
        "L1 2g alone",
        "L1 2g + ChanPred",
        "L1 3g + ResNet",
        "L1 3g + ResNet+ChanPred",
        "L1 3g + ResNet+Forecaster",
        "L1 4g + ResNet",
    ]
    by_condition: dict[str, dict[str, float]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], {})[r["op"]] = fnum(r["duration_ms"])
    out = []
    for condition in order:
        vals = by_condition.get(condition, {})
        out.append(
            {
                "condition": condition,
                "memcpy_ms": f"{vals.get('memcpy', float('nan')):.1f}",
                "memset_ms": f"{vals.get('memset', float('nan')):.1f}",
            }
        )
    write_csv(DATA / "nsys_memory_activity_breakdown.csv", out, ["condition", "memcpy_ms", "memset_ms"])

    labels = [r["condition"].replace(" + ", "\n+ ").replace(" alone", "\nalone").replace(" MIG", "\nMIG") for r in out]
    memcpy = [fnum(r["memcpy_ms"]) for r in out]
    memset = [fnum(r["memset_ms"]) for r in out]
    x = list(range(len(out)))
    plt.figure(figsize=(11.2, 5.2))
    plt.bar(x, memset, color=COLORS["teal"], label="memset duration")
    plt.bar(x, memcpy, bottom=memset, color=COLORS["purple"], label="memcpy duration")
    plt.ylabel("Total profiled memory activity duration (ms)")
    plt.title("NSYS memory ops: 2g increases memset time; NeuralRx/ResNet increase memcpy time")
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.legend(frameon=False)
    return savefig("fig13_nsys_memory_activity_breakdown.png")


def plot_dv2_sanity() -> str:
    rows = read_csv(DV2_SUMMARY)
    out = [{"scenario": r["scenario"], "p99_mean_us": fnum(r["p99_mean"]), "p99_ci_lo": fnum(r["p99_ci_lo"]), "p99_ci_hi": fnum(r["p99_ci_hi"])} for r in rows]
    write_csv(DATA / "dv2_sanity.csv", out, ["scenario", "p99_mean_us", "p99_ci_lo", "p99_ci_hi"])
    labels = [r["scenario"].replace("Dv2_", "").replace("_", "\n") for r in out]
    vals = [r["p99_mean_us"] for r in out]
    yerr = [[v - r["p99_ci_lo"] for v, r in zip(vals, out)], [r["p99_ci_hi"] - v for v, r in zip(vals, out)]]
    plt.figure(figsize=(8.2, 4.6))
    plt.bar(labels, vals, yerr=yerr, color=COLORS["teal"], capsize=4)
    plt.ylabel("p99 inter-kernel gap (us)")
    plt.title("Dv2 replication: generic cross-partition stress stays near baseline")
    plt.xticks(rotation=20, ha="right")
    return savefig("fig14_dv2_sanity.png")


NSYS_LABELS = {
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


def md_table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(key, "")) for key, _ in fields) + " |")
    return "\n".join([header, sep, *body])


def nsys_profile_tables() -> dict[str, str]:
    longtail = {r["scenario"]: r for r in read_csv(ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "longtail_contribution.csv")}
    runtime = {r["scenario"]: r for r in read_csv(ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "runtime_api_analysis.csv")}
    memory = {r["scenario"]: r for r in read_csv(ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "memory_ops_analysis.csv")}
    gap_shape = {r["scenario"]: r for r in read_csv(ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "gap_distribution_shape.csv")}
    kernel_rows = read_csv(ROOT / "20260531" / "nsys_sqlite_v2_analysis" / "all_kernel_summary.csv")

    order = [
        "S2_7g_mig",
        "S5_3g_alone",
        "S6_3g_qwen",
        "S7_3g_neuralrx",
        "S9_3g_3AI_1g",
        "S10_2g_alone",
        "S12_2g_2AI",
        "S13_3g_sat_compute",
        "S14_3g_sat_hbm",
        "S17_2g_sat_compute",
        "S18_4g_neuralrx",
        "S21_4g_2sat",
        "S22_2g_neuralrx",
        "S24_3g_2sat",
        "S26_4g_3sat",
        "S27_3g_chanpred",
        "S28_3g_resnet",
        "S29_3g_forecaster",
        "S30_3g_xapp",
        "S31_3g_resnet_chanpred",
        "S32_3g_resnet_forecaster",
        "S33_4g_chanpred",
        "S34_4g_resnet",
        "S35_2g_chanpred",
        "S36_4g_forecaster",
    ]
    base_memcpy = fnum(memory["S5_3g_alone"]["memcpy_total_us"])
    profile = []
    for s in order:
        if s not in longtail:
            continue
        profile.append(
            {
                "condition": NSYS_LABELS.get(s, s),
                "p99_gap_us": f"{fnum(longtail[s]['p99_threshold_us']):.0f}",
                "p999_gap_us": f"{fnum(longtail[s]['p999_threshold_us']):.0f}",
                "top1_gap_share": f"{100 * fnum(longtail[s]['top1pct_fraction']):.1f}%",
                "runtime_p99_us": f"{fnum(runtime[s]['p99_api_us']):.0f}",
                "memcpy_total_vs_3g": f"{fnum(memory[s]['memcpy_total_us']) / base_memcpy:.1f}x",
            }
        )
    write_csv(
        DATA / "nsys_profile_matrix.csv",
        profile,
        ["condition", "p99_gap_us", "p999_gap_us", "top1_gap_share", "runtime_p99_us", "memcpy_total_vs_3g"],
    )

    selected = ["S5_3g_alone", "S7_3g_neuralrx", "S10_2g_alone", "S22_2g_neuralrx", "S27_3g_chanpred", "S35_2g_chanpred", "S31_3g_resnet_chanpred", "S32_3g_resnet_forecaster"]
    detail = []
    for s in selected:
        detail.append(
            {
                "condition": NSYS_LABELS.get(s, s),
                "p50_gap_us": f"{fnum(gap_shape[s]['p50_us']):.1f}",
                "p90_gap_us": f"{fnum(gap_shape[s]['p90_us']):.0f}",
                "p99_gap_us": f"{fnum(longtail[s]['p99_threshold_us']):.0f}",
                "p999_gap_us": f"{fnum(longtail[s]['p999_threshold_us']):.0f}",
                "max_gap_us": f"{fnum(gap_shape[s]['max_us']):.0f}",
                "top0_1_share": f"{100 * fnum(longtail[s]['top0_1pct_fraction']):.1f}%",
            }
        )
    write_csv(DATA / "nsys_gap_detail_selected.csv", detail, ["condition", "p50_gap_us", "p90_gap_us", "p99_gap_us", "p999_gap_us", "max_gap_us", "top0_1_share"])

    kernels = []
    selected_kernels = {"convert_kernel", "cupy_copy__complex64_complex64", "cupy_copy__float32_float32"}
    for r in kernel_rows:
        if r["scenario"] not in selected or r["kernel_name"] not in selected_kernels:
            continue
        kernels.append(
            {
                "condition": NSYS_LABELS.get(r["scenario"], r["scenario"]),
                "kernel_or_copy": r["kernel_name"].replace("cupy_copy__", "copy "),
                "count": r["count"],
                "median_us": f"{fnum(r['median_dur_us']):.1f}",
                "p99_post_gap_us": f"{fnum(r['p99_post_gap_us']):.0f}",
                "max_post_gap_ms": f"{fnum(r['max_post_gap_us']) / 1000.0:.1f}",
            }
        )
    write_csv(DATA / "nsys_kernel_gap_selected.csv", kernels, ["condition", "kernel_or_copy", "count", "median_us", "p99_post_gap_us", "max_post_gap_ms"])

    runtime_rows = []
    for s in selected:
        runtime_rows.append(
            {
                "condition": NSYS_LABELS.get(s, s),
                "api_calls": runtime[s]["total_api_calls"],
                "runtime_total_ms": f"{fnum(runtime[s]['total_api_time_us']) / 1000.0:.0f}",
                "runtime_p99_us": f"{fnum(runtime[s]['p99_api_us']):.0f}",
                "runtime_max_ms": f"{fnum(runtime[s]['max_api_us']) / 1000.0:.1f}",
                "top_api": runtime[s]["top_api_name"],
            }
        )
    write_csv(DATA / "nsys_runtime_selected.csv", runtime_rows, ["condition", "api_calls", "runtime_total_ms", "runtime_p99_us", "runtime_max_ms", "top_api"])

    dv2_rows = []
    for r in read_csv(DV2_SUMMARY):
        dv2_rows.append(
            {
                "condition": r["scenario"].replace("Dv2_", "").replace("_", " "),
                "n": r["n"],
                "p99_mean_us": f"{fnum(r['p99_mean']):.0f}",
                "p99_ci_us": f"{fnum(r['p99_ci_lo']):.0f}-{fnum(r['p99_ci_hi']):.0f}",
                "p999_mean_us": f"{fnum(r['p999_mean']):.0f}",
                "max_mean_us": f"{fnum(r['max_mean']):.0f}",
            }
        )
    write_csv(DATA / "dv2_sanity_table.csv", dv2_rows, ["condition", "n", "p99_mean_us", "p99_ci_us", "p999_mean_us", "max_mean_us"])

    rootcause_rows = []
    rootcause_order = [
        "L1 7g MIG alone",
        "L1 3g alone",
        "L1 3g + NeuralRx",
        "L1 2g alone",
        "L1 2g + ChanPred",
        "L1 3g + ResNet",
        "L1 3g + ResNet+ChanPred",
        "L1 3g + ResNet+Forecaster",
        "L1 4g + ResNet",
    ]
    rootcause_by_condition = {r["condition"]: r for r in read_csv(NSYS_KERNEL_ACTIVITY)}
    for condition in rootcause_order:
        if condition not in rootcause_by_condition:
            continue
        r = rootcause_by_condition[condition]
        rootcause_rows.append(
            {
                "condition": condition,
                "kernel_p99_us": f"{fnum(r['kernel_gap_p99_us']):.0f}",
                "all_activity_p99_us": f"{fnum(r['all_activity_gap_p99_us']):.0f}",
                "big_gaps_per_run": f"{fnum(r['big_kernel_gaps_ge_1ms']):.1f}",
                "big_gaps_with_mem": f"{fnum(r['big_gaps_with_mem_pct']):.1f}%",
                "mem_fraction": f"{fnum(r['mean_mem_fraction_in_big_gaps_pct']):.1f}%",
            }
        )

    memory_by_condition: dict[str, dict[str, float]] = {}
    for r in read_csv(NSYS_MEMORY_ACTIVITY):
        memory_by_condition.setdefault(r["condition"], {})[r["op"]] = fnum(r["duration_ms"])
    base_memcpy_ms = memory_by_condition["L1 3g alone"]["memcpy"]
    base_memset_ms = memory_by_condition["L1 3g alone"]["memset"]
    memory_deep_rows = []
    for condition in rootcause_order:
        vals = memory_by_condition.get(condition, {})
        if not vals:
            continue
        memcpy_ms = vals.get("memcpy", float("nan"))
        memset_ms = vals.get("memset", float("nan"))
        memory_deep_rows.append(
            {
                "condition": condition,
                "memcpy_ms": f"{memcpy_ms:.1f}",
                "memcpy_vs_3g": f"{memcpy_ms / base_memcpy_ms:.1f}x",
                "memset_ms": f"{memset_ms:.1f}",
                "memset_vs_3g": f"{memset_ms / base_memset_ms:.1f}x",
            }
        )

    transition_rows = []
    transition_keep = {
        ("L1 3g + NeuralRx", "copy_complex64_kernel -> convert_kernel"),
        ("L1 3g + NeuralRx", "convert_kernel -> noise_intf_est"),
        ("L1 3g + NeuralRx", "convert_kernel -> eq_coef"),
        ("L1 2g alone", "convert_kernel -> noise_intf_est"),
        ("L1 2g alone", "copy_float32_kernel -> convert_kernel"),
        ("L1 2g alone", "convert_kernel -> eq_coef"),
        ("L1 3g + ResNet+ChanPred", "convert_kernel -> ch_est_pre"),
        ("L1 3g + ResNet+Forecaster", "convert_kernel -> ch_est_pre"),
        ("L1 4g + ResNet", "convert_kernel -> ch_est_pre"),
    }
    for r in read_csv(NSYS_TRANSITIONS):
        if (r["condition"], r["transition"]) not in transition_keep:
            continue
        transition_rows.append(
            {
                "condition": r["condition"],
                "transition": r["transition"],
                "count": r["count"],
                "p50_gap_us": f"{fnum(r['p50_gap_us']):.0f}",
                "p99_gap_us": f"{fnum(r['p99_gap_us']):.0f}",
                "max_gap_ms": r["max_gap_ms"],
                "mem_fraction": f"{fnum(r['mean_mem_fraction_pct']):.1f}%",
            }
        )
    write_csv(
        DATA / "nsys_selected_rootcause_transitions.csv",
        transition_rows,
        ["condition", "transition", "count", "p50_gap_us", "p99_gap_us", "max_gap_ms", "mem_fraction"],
    )

    return {
        "rootcause": md_table(rootcause_rows, [
            ("condition", "Condition"),
            ("kernel_p99_us", "kernel-only p99 gap us"),
            ("all_activity_p99_us", "all-activity p99 gap us"),
            ("big_gaps_per_run", ">=1ms kernel gaps / run"),
            ("big_gaps_with_mem", "with memcpy/memset"),
            ("mem_fraction", "memory fraction inside big gaps"),
        ]),
        "memory_deep": md_table(memory_deep_rows, [
            ("condition", "Condition"),
            ("memcpy_ms", "memcpy total ms"),
            ("memcpy_vs_3g", "memcpy vs 3g alone"),
            ("memset_ms", "memset total ms"),
            ("memset_vs_3g", "memset vs 3g alone"),
        ]),
        "transitions_deep": md_table(transition_rows, [
            ("condition", "Condition"),
            ("transition", "Boundary transition"),
            ("count", "count"),
            ("p50_gap_us", "p50 gap us"),
            ("p99_gap_us", "p99 gap us"),
            ("max_gap_ms", "max gap ms"),
            ("mem_fraction", "mean mem fraction"),
        ]),
        "profile": md_table(profile, [
            ("condition", "NSYS condition"),
            ("p99_gap_us", "p99 gap us"),
            ("p999_gap_us", "p999 gap us"),
            ("top1_gap_share", "top 1% gap share"),
            ("runtime_p99_us", "runtime p99 us"),
            ("memcpy_total_vs_3g", "memcpy total vs 3g alone"),
        ]),
        "gap_detail": md_table(detail, [
            ("condition", "Condition"),
            ("p50_gap_us", "p50 gap us"),
            ("p90_gap_us", "p90 gap us"),
            ("p99_gap_us", "p99 gap us"),
            ("p999_gap_us", "p999 gap us"),
            ("max_gap_us", "max gap us"),
            ("top0_1_share", "top 0.1% gap share"),
        ]),
        "kernel": md_table(kernels, [
            ("condition", "Condition"),
            ("kernel_or_copy", "Kernel/copy"),
            ("count", "count"),
            ("median_us", "median duration us"),
            ("p99_post_gap_us", "p99 post-gap us"),
            ("max_post_gap_ms", "max post-gap ms"),
        ]),
        "runtime": md_table(runtime_rows, [
            ("condition", "Condition"),
            ("api_calls", "API calls"),
            ("runtime_total_ms", "runtime total ms"),
            ("runtime_p99_us", "runtime p99 us"),
            ("runtime_max_ms", "runtime max ms"),
            ("top_api", "top API"),
        ]),
        "dv2": md_table(dv2_rows, [
            ("condition", "Dv2 condition"),
            ("n", "n"),
            ("p99_mean_us", "p99 mean us"),
            ("p99_ci_us", "p99 CI us"),
            ("p999_mean_us", "p999 mean us"),
            ("max_mean_us", "max mean us"),
        ]),
    }


def build_markdown(figs: dict[str, str]) -> None:
    tables = nsys_profile_tables()
    md = f"""# MIG는 AI-RAN에 충분한 격리 추상화가 아니다: 시각 증거

생성일: 2026-06-01  
소스 범위: `cloudlab_results/results/20260531`, `cloudlab_results/results/20260601`

이 문서는 정리된 evidence를 그림 중심으로 다시 구성한 한국어 버전이다. 핵심 결론은 단순히 "L1만 격리하면 되는가"가 아니라, **L1 tail latency와 AI workload service quality를 동시에 만족해야 하는 AI-RAN에서 MIG의 정적 파티셔닝이 잘못된 제어 추상화**라는 점이다.

## 1. 작은 MIG slice는 L1 headroom을 줄인다

![Partition baseline]({figs['partition']})

5/31 baseline에서 2g L1은 Full GPU v2 대비 mean latency가 약 +40.2% 증가한다. 7g/4g/3g는 비교적 안정적이지만, slice가 작아질수록 L1이 사용할 수 있는 SM/cache/memory-system headroom이 줄어든다. 이 결과는 "MIG가 capacity는 나눠주지만 real-time headroom까지 보장하지는 않는다"는 첫 번째 증거다.

## 2. NeuralRx는 generic AI가 아니라 PHY-AI co-tenant다

![Phase4 NeuralRx]({figs['phase4']})

5/31 Phase4에서 3g L1 기준 NeuralRx co-tenant는 L1 p99를 약 +376%까지 키웠다. Qwen-small, ChanPred, XApp도 p99를 올리지만 NeuralRx는 훨씬 크다. 따라서 주장은 "모든 외부 AI가 항상 위험하다"가 아니라, **AI-RAN에서 실제로 붙이고 싶은 PHY-AI의 temporal behavior가 L1에 매우 위험할 수 있다**는 쪽이 더 정확하다.

## 3. 하지만 generic cross-partition saturation이 주범은 아니다

![F saturation]({figs['f']})

6/1 F saturation은 D2D/H2D/GEMM/ResNet/ChanPred/Forecaster/stack/kitchen stress를 걸었지만, non-baseline 39개 조건에서 positive L1 p99 inflation이 0개였다. 이 negative result가 중요하다. 우리의 주장은 "MIG의 cross-partition isolation이 전부 깨졌다"가 아니다. 오히려 generic saturation은 잘 막히기 때문에, AI-RAN failure가 더 선명해진다.

## 4. 같은 partition에 L1과 NeuralRx를 넣으면 p99가 폭발한다

![G coloc]({figs['g']})

6/1 G에서 same-partition L1+NeuralRx coloc은 3g p99 +372.6%, 4g p99 +536.7%, 2g p99 +504.5% 수준의 catastrophic tail을 만든다. 특히 4g에서도 해결되지 않는다. 즉 "큰 partition을 주면 coloc이 안전해진다"는 단순한 해법이 아니다.

## 5. H dual sanity check: 외부 stress는 안전, coloc은 위험

![H dual]({figs['h']})

6/1 H에서는 같은 3g L1 기준으로 외부 D2D/GEMM/stack/kitchen stress는 p99가 baseline 근처에 머문다. 반면 coloc 조건은 p99가 350ms 이상으로 튄다. 이것은 원인을 더 좁혀준다. 문제는 sustained cross-partition HBM bandwidth 하나가 아니라, **same-partition temporal sharing, runtime scheduling, copy/kernel phase alignment**가 L1 deadline과 충돌하는 구조다.

## 6. AI workload도 partition에 자유롭지 않다

![AI partition scaling]({figs['ai_scaling']})

AI workload는 leftover slice에 그냥 고정 배치할 수 없다. Qwen-small은 1g에서 실패했고, Qwen-7B 계열은 4g에서만 의미 있게 실행된다. ResNet, sat_compute, sat_hbm, Forecaster는 partition size에 따라 throughput이 크게 달라진다. 따라서 L1을 보호하려고 큰 slice를 주면 AI capacity가 줄고, AI throughput을 보존하려면 L1 headroom이 줄어든다.

## 7. AI 평균 throughput이 안정적이어도 AI p99는 흔들린다

![AI per-op p99]({figs['ai_p99']})

5/31 AI per-op latency에서 ChanPred는 2g 기준 p99 +27.0%, 3g +23.7%, 4g +19.6%까지 증가한다. NeuralRx 3g도 +12.9%, Qwen도 partition별로 +3.2~+6.7% 수준의 p99 증가가 보인다. 즉 throughput만 보면 양쪽이 안전해 보일 수 있지만, real-time 관점에서는 AI side도 tail-latency tradeoff를 가진다.

## 8. 최종 tradeoff

![Tradeoff]({figs['tradeoff']})

MIG에서 선택지는 모두 비용을 가진다.

- L1을 작은 slice에 두면 L1 headroom이 줄어든다.
- L1을 보호하려고 큰 slice를 주면 AI가 fit/throughput/p99 비용을 낸다.
- NeuralRx를 separate partition에 둬도 5/31에서는 L1 p99 risk가 컸다.
- L1과 NeuralRx를 같은 partition에 두면 6/1 G/H에서 p99가 catastrophic하게 폭발한다.
- 여러 AI workload를 나눠 배치하면 multi-AI phase behavior가 non-monotonic해진다.

이 마지막 그림은 단일 실험의 raw plot이 아니라 evidence synthesis plot이다. L1 축은 관측된 latency inflation을 사용했고, AI 축은 fit failure, AI p99 inflation, coloc NeuralRx throughput 감소 정황을 하나의 risk axis에 모은 것이다. 따라서 논문 본문에서는 앞선 개별 figure들을 primary evidence로 쓰고, 이 그림은 전체 tradeoff를 설명하는 summary figure로 쓰는 것이 안전하다.

## 9. 정적 partition plan은 L1과 AI 사이의 tradeoff를 드러낸다

![Static partition sweep]({figs['static_partition']})

5/31 Phase2/Phase3는 여러 MIG partition plan을 직접 바꿔가며 L1 p99를 본 sweep이다. 그림의 라벨은 코드명 대신 실제 배치 의미로 풀어썼다. 예를 들어 `L1 3g / Qwen-small on 2g+2g`는 L1이 3g slice를 쓰고, Qwen-small 계열 AI가 두 개의 2g slice에 배치된 조건이다. `L1 2g / Qwen-small on 3g`는 AI 쪽에 더 큰 slice를 주는 대신 L1이 2g로 줄어든 조건이다.

여기서 가장 중요한 점은 L1이 2g로 작아지는 순간 p99가 82~84ms대로 올라간다는 것이다. 반대로 L1을 4g 또는 약 3g로 키워도 p99가 완전히 baseline으로 돌아오지는 않는다. 즉, L1에 더 큰 slice를 주면 headroom은 좋아지지만 AI가 쓸 수 있는 GPU budget이 줄고, AI에 더 큰 slice를 주면 L1이 작아져 real-time tail이 악화된다. 이것이 MIG의 정적 slicing이 만드는 가장 기본적인 운영 tradeoff다.

이 그림은 내부 실험 코드명 대신 실제 partition과 workload 의미로 바꿔 표시했다. 빨간 막대는 L1이 2g로 starved된 조건이고, 노란 막대는 L1이 3g/4g를 받았지만 여전히 AI co-tenant와 함께 실행된 조건이다. 핵심은 특정 AI 하나가 아니라, **partition plan 자체가 L1 safety와 AI capacity를 동시에 결정한다**는 점이다.

## 10. AI throughput과 AI p99는 같은 이야기를 하지 않는다

![AI throughput vs p99]({figs['ai_tput_vs_p99']})

5/31 `ai_throughput_v2`에서는 L1 background가 있어도 AI mean throughput은 거의 변하지 않는다. ChanPred, NeuralRx, Qwen-small, XApp 모두 평균 처리량 변화만 보면 0~2% 수준으로 안전해 보인다. 하지만 같은 계열의 `ai_per_op_latency`를 보면 ChanPred는 최대 +27%, NeuralRx는 +12.9%, XApp은 +12.3%, Qwen은 +6.7%까지 per-operation p99가 증가한다.

이 그림의 목적은 AI side도 단순 throughput으로 평가하면 안 된다는 점을 보여주는 것이다. AI-RAN에서 AI inference가 scheduling loop 안에 들어오거나 near-real-time decision에 쓰이면, 평균 throughput이 아니라 per-op tail latency가 중요해진다. 그러면 MIG의 문제는 L1만의 문제가 아니다. L1을 보호하기 위한 partitioning이 AI의 fit/throughput/tail latency와 충돌하고, AI를 보호하기 위한 partitioning이 L1 headroom과 충돌한다.

## 11. Coloc이 시작되면 외부 AI 종류는 2차 문제가 된다

![Coloc external dominance]({figs['coloc_external']})

6/1 G 실험은 L1과 NeuralRx를 같은 MIG partition에 coloc한 뒤, 바깥 partition에 다른 AI workload를 추가로 올린 조건들을 비교한다. 결과는 매우 선명하다. 외부에 ChanPred, Forecaster, Qwen-small, ResNet, HBM saturation, XApp, 또는 복수 AI를 올려도 L1 p99는 대체로 356~371ms 근처에 머문다. 즉, 이 구간에서는 외부 AI 종류가 핵심 원인이 아니다. 이미 같은 partition 안에서 L1과 NeuralRx가 temporal resources를 공유하는 순간, tail failure가 지배적이 된다.

이 그림은 "어떤 AI가 외부에 있으면 위험한가?"라는 질문보다 "L1과 in-line PHY-AI를 같은 partition에 넣어도 되는가?"라는 질문이 더 중요하다는 점을 보여준다. 답은 데이터상 명확히 아니다. MIG는 partition 사이 capacity isolation은 줄 수 있지만, 같은 MIG device 안에서 L1과 PHY-AI가 runtime, kernel launch, copy, SM/memory path를 시간적으로 나눠 쓰는 문제는 해결하지 못한다.

## 12. NSYS SQLite 재분석: kernel-only gap은 idle이 아니다

![NSYS gap]({figs['nsys_gap']})

처음 NSYS 결과를 kernel-to-kernel gap만으로 보면 "L1 kernel 사이에 긴 idle이 있다"처럼 보인다. 그런데 SQLite에서 `CUPTI_ACTIVITY_KIND_KERNEL`, `MEMCPY`, `MEMSET` interval을 다시 합쳐보면 해석이 달라진다. 많은 long gap은 진짜로 GPU가 논 것이 아니라, 다음 L1 kernel로 넘어가기 전 boundary가 memcpy/memset 활동으로 채워진 구간이다.

따라서 여기서의 문제 지점은 단순 idle gap이 아니다. 더 정확히는 **L1 pipeline의 convert/copy/memset boundary가 partition size와 co-tenant에 따라 길어지는 현상**이다. 이게 중요한 이유는 AI-RAN L1이 sustained 평균 throughput만 필요한 workload가 아니라 frame cadence를 맞춰야 하는 workload이기 때문이다. 평균 GPU busy가 낮아도, 특정 boundary가 수백 us~ms 단위로 흔들리면 p99/p999 deadline이 바로 깨진다.

그림에서 빨간 막대는 kernel만 보고 잰 p99 gap이고, 파란 막대는 kernel/memcpy/memset을 모두 activity로 merge한 뒤 잰 p99 gap이다. 두 값의 차이가 클수록 kernel 사이가 "빈 시간"이 아니라 memory op로 채워졌다는 뜻이다. 노란 선은 1ms 이상 kernel gap 중 memcpy/memset이 포함된 비율이다.

가장 선명한 조건은 `L1 2g alone`과 `L1 2g + ChanPred`다. kernel-only p99 gap은 각각 1444us, 1404us까지 커지지만 all-activity p99 gap은 243us, 175us로 훨씬 작다. 동시에 1ms 이상 kernel gap의 96.7%, 99.1%가 memcpy/memset을 포함한다. 즉 2g에서는 L1이 alone이어도 memory/setup boundary가 매우 조밀하게 끼어들고, ChanPred를 같이 두면 그 구조가 더 강해진다.

`L1 3g + NeuralRx`도 같은 방향의 증거다. 3g baseline은 1ms 이상 kernel gap이 run당 19.3개이고 그중 memory 포함 비율이 45.3%인데, NeuralRx가 붙으면 run당 83.0개, memory 포함 비율 72.4%로 늘어난다. 이 결과는 NeuralRx가 단순히 "외부 AI 하나"가 아니라 L1의 copy/convert boundary와 시간적으로 충돌하는 PHY-AI workload라는 해석을 뒷받침한다.

{tables['rootcause']}

## 13. NSYS가 가리키는 실제 문제 지점: convert/copy/memset boundary

![Memory ops]({figs['memory_ops']})

memory activity를 보면 원인이 더 구체화된다. 같은 L1 pipeline에서 memcpy call 수와 memset call 수는 거의 같지만, 총 duration은 partition/workload에 따라 크게 바뀐다. `L1 3g alone`의 memcpy 총 시간은 46.8ms인데 `L1 3g + NeuralRx`에서는 195.1ms로 약 4.2배 증가한다. `L1 3g + ResNet`도 155.2ms, `L1 3g + ResNet+Forecaster`도 140.4ms까지 커진다. 반대로 2g 조건은 memcpy보다 memset 쪽이 더 직접적이다. `L1 3g alone`의 memset은 408.2ms인데 `L1 2g alone`과 `L1 2g + ChanPred`는 각각 814.5ms, 813.9ms로 거의 2배다.

이 차이가 중요하다. 만약 문제가 단순 sustained HBM bandwidth 포화라면 H2D/D2D/GEMM synthetic stress가 일관되게 L1 p99를 망가뜨려야 한다. 하지만 6/1 F와 Dv2에서는 generic cross-partition stress가 대부분 baseline 근처에 머물렀다. 반면 NeuralRx, ResNet 계열, 작은 2g L1에서는 copy/memset boundary가 길어진다. 즉 우리가 주장해야 할 bandwidth 문제는 "평균 GB/s를 많이 썼다"가 아니라, **고정된 device bandwidth와 memory path를 시간적으로 나눠 쓰는 상황에서 L1 boundary가 deterministic하게 보호되지 않는다**는 것이다.

{tables['memory_deep']}

transition level에서도 같은 구조가 보인다. NeuralRx 조건에서는 `copy_complex64_kernel -> convert_kernel`, `convert_kernel -> noise_intf_est`, `convert_kernel -> eq_coef` 같은 L1 stage boundary의 p99 gap이 수십 ms까지 벌어진다. ResNet+ChanPred/Forecaster 조건에서는 `convert_kernel -> ch_est_pre`가 1920번 반복되고, 그 boundary의 평균 memory fraction이 88~90% 수준이다. 즉 tail은 하나의 거대한 kernel이 느려져서 생기는 것이 아니라, 반복적인 L1 stage 사이에서 copy/memset이 끼어드는 방식으로 만들어진다.

{tables['transitions_deep']}

그래서 NSYS 근거로 써야 할 문장은 더 좁고 강해야 한다. "MIG가 bandwidth isolation을 전혀 못 한다"가 아니라, **MIG의 정적 capacity isolation은 L1 kernel boundary의 temporal memory activity를 제어하지 못한다**가 맞다. 이것이 AI-RAN에서 치명적인 이유는 L1은 throughput workload가 아니라 deadline workload이고, NeuralRx/ChanPred/ResNet 같은 PHY-AI workload도 평균 처리량뿐 아니라 per-op tail latency와 fit constraint를 동시에 갖기 때문이다.

## 14. Dv2 replication은 negative result를 강화한다

![Dv2 sanity]({figs['dv2']})

Dv2 반복 실험은 H2D, D2D, compute, launch, ChanPred stress가 baseline 근처에 머무는 것을 보여준다. 이 그림은 주장을 더 조심스럽고 강하게 만든다. 즉, "MIG cross-partition이 항상 깨진다"가 아니라, **generic cross-partition stress는 대체로 안전하지만 AI-RAN PHY-AI composition에서는 정적 MIG가 충분하지 않다**가 맞다.

아래 표는 Dv2 반복 실험의 숫자다. H2D, D2D, compute, launch, ChanPred 모두 p99 mean이 baseline 주변에 있고 CI도 크게 분리되지 않는다. 이 표는 논문에서 매우 중요하다. 왜냐하면 reviewer가 "그냥 MIG가 isolation을 못 하는 것 아닌가?"라고 물을 때, 우리는 "아니다. generic cross-partition stress는 꽤 잘 막힌다. 문제는 AI-RAN의 L1+PHY-AI composition과 static placement다"라고 답할 수 있기 때문이다.

{tables['dv2']}

## 결론

데이터 기반으로 가장 강한 주장은 다음이다.

> MIG는 generic cross-partition throughput isolation에는 효과적일 수 있다. 그러나 AI-RAN은 L1 tail latency와 AI workload service quality를 동시에 보장해야 한다. MIG는 static capacity slicing만 제공하므로, L1 headroom, AI fit/throughput/p99, PHY-AI co-location tail latency 사이의 tradeoff를 안전하게 제어하지 못한다. 따라서 MIG 단독으로는 real-time L1 + PHY-AI consolidation을 위한 충분한 isolation mechanism이 아니다.

## NSYS까지 포함한 최종 해석

지금까지의 데이터는 다음 순서로 읽는 것이 가장 강하다.

1. **MIG는 capacity isolation에는 의미가 있다.** 6/1 F와 5/31 Dv2에서 generic D2D/H2D/GEMM/launch/ChanPred stress는 baseline 주변에 머물렀다. 따라서 "MIG가 모든 cross-partition isolation에 실패한다"는 주장은 데이터와 맞지 않는다.

2. **하지만 AI-RAN이 원하는 것은 capacity isolation만이 아니다.** L1은 frame deadline을 맞춰야 하고, AI workload도 throughput뿐 아니라 per-op p99와 fit constraint를 가진다. 2g L1은 standalone부터 headroom이 작고, AI workload는 작은 slice에서 fit 실패나 throughput scaling 문제를 보인다.

3. **NSYS는 failure mechanism이 단순 idle gap이 아니라 memory-filled kernel boundary라는 점을 보여준다.** kernel-only gap만 보면 긴 idle처럼 보이지만, kernel/memcpy/memset activity를 merge하면 2g 조건의 1ms 이상 kernel gap 대부분이 memory op로 채워져 있다. 즉 문제는 GPU가 놀아서가 아니라, L1의 convert/copy/memset boundary가 temporal하게 보호되지 않는다는 것이다.

4. **copy/convert/runtime boundary가 workload별로 다르게 흔들린다.** NeuralRx는 3g L1의 memcpy total을 4.2배로 키우고, 2g L1은 memset duration을 3g 대비 거의 2배로 키운다. 이것은 synthetic HBM stress와 PHY-AI workload가 같지 않다는 뜻이다.

5. **가장 치명적인 지점은 same-partition L1+PHY-AI coloc이다.** 6/1 G/H에서 L1과 NeuralRx가 같은 partition에 들어가면 p99가 수백 ms로 폭발한다. 외부 AI 종류를 바꿔도 coloc 이후에는 p99가 이미 높은 영역에 머문다. 이 결과는 MIG가 partition 사이 격리는 줄 수 있어도, 같은 MIG device 내부의 temporal sharing 문제는 해결하지 못한다는 점을 보여준다.

따라서 논문에서 최종 메시지는 이렇게 가져가야 한다.

> MIG는 GPU를 공간적으로 나누는 좋은 capacity isolation 도구지만, AI-RAN의 real-time L1 + PHY-AI consolidation에는 부족하다. 이유는 L1과 AI가 모두 tail-sensitive하고, static partition은 workload phase, copy/memset/runtime boundary, kernel launch gap, PHY-AI coloc behavior를 제어하지 못하기 때문이다. AI-RAN에는 MIG 위에 workload-aware temporal scheduling 또는 admission/control layer가 추가로 필요하다.

## 생성된 source tables

- `data/partition_baseline.csv`
- `data/phase4_phy_ai_p99.csv`
- `data/f_saturation_block_summary.csv`
- `data/g_coloc_l1_p99.csv`
- `data/h_dual_p99.csv`
- `data/ai_throughput_parsed.csv`
- `data/ai_partition_scaling.csv`
- `data/ai_per_op_latency_parsed.csv`
- `data/ai_per_op_p99_delta.csv`
- `data/tradeoff_summary.csv`
- `data/static_partition_sweep.csv`
- `data/ai_throughput_v2_parsed.csv`
- `data/ai_throughput_vs_p99.csv`
- `data/g_coloc_external_dominance.csv`
- `data/nsys_gap_summary.csv`
- `data/memory_ops_pressure.csv`
- `data/nsys_kernel_vs_all_activity_summary.csv`
- `data/nsys_memory_activity_breakdown.csv`
- `data/nsys_selected_rootcause_transitions.csv`
- `data/dv2_sanity.csv`
- `data/nsys_profile_matrix.csv`
- `data/nsys_gap_detail_selected.csv`
- `data/nsys_kernel_gap_selected.csv`
- `data/nsys_runtime_selected.csv`
- `data/dv2_sanity_table.csv`
"""
    (OUT / "MIG_AIRAN_VISUAL_EVIDENCE_KR.md").write_text(md)


def build_notebook() -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MIG / AI-RAN Visual Evidence Notebook\\n",
                    "\\n",
                    "이 노트북은 `build_visual_evidence.py`를 실행해서 source CSV, PNG figure, 한국어 MD 리포트를 재생성합니다.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%run build_visual_evidence.py\\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Generated report\\n",
                    "\\n",
                    "- [MIG_AIRAN_VISUAL_EVIDENCE_KR.md](MIG_AIRAN_VISUAL_EVIDENCE_KR.md)\\n",
                    "- Figures are under `figures/`; source tables are under `data/`.\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "visualize_mig_airan_evidence.ipynb").write_text(json.dumps(nb, indent=2, ensure_ascii=False))


def main() -> None:
    setup()
    rows = l1_summary()
    figs = {
        "partition": plot_partition_baseline(rows),
        "phase4": plot_phase4_neuralrx(rows),
        "f": plot_f_saturation(),
        "g": plot_g_coloc(),
        "h": plot_h_dual(rows),
    }
    ai_rows = parse_ai_throughput()
    figs["ai_scaling"] = plot_ai_partition_scaling(ai_rows)
    ai_latency = parse_ai_latency()
    figs["ai_p99"] = plot_ai_per_op_p99(ai_latency)
    figs["tradeoff"] = plot_tradeoff_summary()
    figs["static_partition"] = plot_static_partition_sweep(rows)
    figs["ai_tput_vs_p99"] = plot_ai_throughput_vs_p99(ai_latency)
    figs["coloc_external"] = plot_coloc_external_dominance()
    figs["nsys_gap"] = plot_nsys_kernel_vs_activity_gap()
    figs["memory_ops"] = plot_nsys_memory_activity_breakdown()
    figs["dv2"] = plot_dv2_sanity()
    build_markdown(figs)
    build_notebook()
    print(f"wrote {OUT / 'MIG_AIRAN_VISUAL_EVIDENCE_KR.md'}")
    print(f"wrote {OUT / 'visualize_mig_airan_evidence.ipynb'}")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
