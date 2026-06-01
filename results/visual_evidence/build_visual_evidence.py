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


def build_markdown(figs: dict[str, str]) -> None:
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

## 12. NSYS gap 분석: tail은 inter-kernel gap에서 보인다

![NSYS gap]({figs['nsys_gap']})

NSYS deep analysis는 L1 kernel 자체의 평균 실행시간만으로는 tail을 설명하기 어렵고, kernel 사이 gap과 burst idle 구간이 중요하다는 점을 보여준다. 그림의 라벨은 실제 조건으로 풀어쓴 것이다. `L1 2g + ChanPred on 3g` 조건은 p99/p999 inter-kernel gap이 가장 크고, `L1 4g + ResNet on 2g`는 상대적으로 작다. `L1 3g + ResNet+ChanPred`, `L1 3g + ResNet+Forecaster`처럼 heterogeneous AI 조합은 중간에 위치한다.

이 결과는 우리가 말하는 bandwidth 문제가 단순히 "몇 GB/s를 썼는가"가 아니라는 점을 뒷받침한다. L1이 일정한 frame cadence를 유지하려면 다음 kernel이 제때 launch되고 copy/convert 단계가 제때 지나가야 한다. 하지만 small partition이나 특정 AI mix에서는 kernel 사이 빈 시간이 길어지고, 이 gap tail이 L1 p99/p999로 나타난다. 그래서 이 문제를 **temporal bandwidth 또는 scheduling headroom 부족**으로 해석하는 것이 sustained HBM bandwidth 하나로 설명하는 것보다 더 정확하다.

## 13. Memory/copy pressure는 workload별로 다르다

![Memory ops]({figs['memory_ops']})

NSYS memory-op summary는 각 workload가 L1 주변에 만드는 copy pressure가 다르다는 점을 보여준다. 이 그림은 `L1 3g alone`을 1.0으로 정규화했다. NeuralRx, Qwen, ResNet, Forecaster는 total memcpy나 memcpy p99가 baseline 대비 크게 달라진다. 반면 모든 workload가 같은 방식으로 나빠지는 것은 아니다.

이 그림의 역할은 mechanism 설명이다. 6/1 F/Dv2에서 generic D2D/H2D saturation이 L1 p99를 크게 망가뜨리지 않았는데, 5/31 NeuralRx나 6/1 coloc에서는 문제가 커졌다. 그 차이는 단순 bandwidth 양만으로는 설명하기 어렵다. PHY-AI는 copy, convert, kernel launch, framework runtime phase가 L1 pipeline과 특정 시간 패턴으로 겹칠 수 있고, 이 temporal overlap이 tail을 만든다.

## 14. Dv2 replication은 negative result를 강화한다

![Dv2 sanity]({figs['dv2']})

Dv2 반복 실험은 H2D, D2D, compute, launch, ChanPred stress가 baseline 근처에 머무는 것을 보여준다. 이 그림은 주장을 더 조심스럽고 강하게 만든다. 즉, "MIG cross-partition이 항상 깨진다"가 아니라, **generic cross-partition stress는 대체로 안전하지만 AI-RAN PHY-AI composition에서는 정적 MIG가 충분하지 않다**가 맞다.

## 결론

데이터 기반으로 가장 강한 주장은 다음이다.

> MIG는 generic cross-partition throughput isolation에는 효과적일 수 있다. 그러나 AI-RAN은 L1 tail latency와 AI workload service quality를 동시에 보장해야 한다. MIG는 static capacity slicing만 제공하므로, L1 headroom, AI fit/throughput/p99, PHY-AI co-location tail latency 사이의 tradeoff를 안전하게 제어하지 못한다. 따라서 MIG 단독으로는 real-time L1 + PHY-AI consolidation을 위한 충분한 isolation mechanism이 아니다.

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
- `data/dv2_sanity.csv`
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
    figs["nsys_gap"] = plot_nsys_gap_summary()
    figs["memory_ops"] = plot_memory_ops_pressure()
    figs["dv2"] = plot_dv2_sanity()
    build_markdown(figs)
    build_notebook()
    print(f"wrote {OUT / 'MIG_AIRAN_VISUAL_EVIDENCE_KR.md'}")
    print(f"wrote {OUT / 'visualize_mig_airan_evidence.ipynb'}")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
