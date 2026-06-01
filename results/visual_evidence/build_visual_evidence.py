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
    build_markdown(figs)
    build_notebook()
    print(f"wrote {OUT / 'MIG_AIRAN_VISUAL_EVIDENCE_KR.md'}")
    print(f"wrote {OUT / 'visualize_mig_airan_evidence.ipynb'}")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
