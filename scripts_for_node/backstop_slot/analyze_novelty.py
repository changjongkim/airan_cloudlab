#!/usr/bin/env python3
"""Novelty evidence: value-rule generality, ablations, AI-load and weak-cell sweeps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
RAW = ROOT / "raw"
JOBS = sys.argv[1:] or ["59103692", "59111015"]


def find(stem: str) -> Path | None:
    for job in JOBS:
        path = RAW / f"{stem}_j{job}.json"
        if path.is_file():
            return path
    return None


def metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    h, a = d["headline"], d["all"]
    ai = h.get("ai_total") or {}
    ttft = [v["ttft_ms"] for v in (d.get("ai") or {}).values() if v and v["ttft_ms"].get("n")]
    return {
        "late_tbs": h["late_tbs"], "decoded_on_time": h["decoded_on_time"],
        "rescues": h["nrx_rescues_on_time"], "nrx_runs": h["nrx_runs"],
        "nrx_late": h.get("nrx_late") or 0, "conv_p99_ms": a["conv_done_ms"]["p99"],
        "ai_slo_tokens_per_s": ai.get("tokens_within_slo_per_s", 0.0),
        "ai_tokens_per_s": ai.get("tokens_per_s", 0.0),
        "ttft_p50_ms": float(np.mean([t["p50"] for t in ttft])) if ttft else None,
    }


def mean_of(stems: list[str]) -> dict | None:
    runs = [metrics(p) for p in (find(s) for s in stems) if p]
    if not runs:
        return None
    out = {k: (float(np.mean([r[k] for r in runs])) if runs[0][k] is not None else None)
           for k in runs[0]}
    out["seeds"] = len(runs)
    out["per_seed_rescues"] = [r["rescues"] for r in runs]
    out["per_seed_slo"] = [r["ai_slo_tokens_per_s"] for r in runs]
    return out


def generality() -> list[dict]:
    rows = []
    for name, path in [("weak_rank1 (evaluation)", RAW / "cbcrc_value_j59103692.json")] + [
            (p, RAW / f"cbcrc_{p}_j{JOBS[-1]}.json")
            for p in ("weak_cdl_e", "weak_wide_snr", "weak_mcs4")]:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        fails = [r for r in data if not r["conv_ok"]]
        rescued = [r for r in fails if r["nrx_ok"]]
        by = {}
        for k in sorted({r["cb_fail"] for r in fails}):
            g = [r for r in fails if r["cb_fail"] == k]
            by[str(k)] = {"conv_fails": len(g), "nrx_rescues": sum(r["nrx_ok"] for r in g)}
        kept = [r for r in fails if r["cb_fail"] <= 1]
        rows.append({
            "profile": name, "tbs": len(data), "conv_fails": len(fails),
            "nrx_rescues": len(rescued),
            "rescues_kept_by_rule": sum(r["nrx_ok"] for r in kept),
            "nrx_runs_with_rule": len(kept),
            "by_failed_code_blocks": by,
        })
    return rows


def main() -> None:
    result = {"generality": generality()}
    ablation = {}
    variants = [
        ("Our Scheme", "f{s}r{r}_c16_rescue_value_backstop_corun"),
        ("- value rule", "a{s}r{r}noval_c16_rescue_value_backstop_corun"),
        ("- deadline admission", "a{s}r{r}noadmit_c16_rescue_value_backstop_corun"),
        ("- lane sharing (partner GPU only)", "a{s}r{r}partner_c16_rescue_value_backstop_corun"),
        ("- latest-start start (conventional failure only)", "a{s}r{r}failonly_c16_rescue_value_backstop_corun"),
        ("- conventional first (NeuralRx at arrival)", "a{s}r{r}arrival_c16_rescue_value_backstop_corun"),
        ("- co-run gating (fixed 50% share)", "f{s}r{r}_c16_rescue_value_static"),
        ("- co-run (AI only when radio idle)", "a{s}r{r}strict_c16_rescue_value_backstop"),
    ]
    for rate in (4, 8):
        ablation[str(rate)] = []
        for label, pattern in variants:
            m = mean_of([pattern.format(s=s, r=rate) for s in (1, 2)])
            if m:
                ablation[str(rate)].append({"variant": label, **m})
    result["ablation_16cells"] = ablation

    load = {}
    for label, pattern, rates in [
        ("Our Scheme", {4: "f{s}r4_c16_rescue_value_backstop_corun", 8: "f{s}r8_c16_rescue_value_backstop_corun",
                        12: "l{s}r12_c16_rescue_value_backstop_corun"}, (4, 8, 12)),
        ("fixed 30% share", {r: f"l{{s}}r{r}s30_c16_rescue_value_static" for r in (4, 8, 12)}, (4, 8, 12)),
        ("fixed 50% share", {4: "f{s}r4_c16_rescue_value_static", 8: "f{s}r8_c16_rescue_value_static",
                             12: "l{s}r12_c16_rescue_value_static"}, (4, 8, 12)),
        ("fixed 70% share", {r: f"l{{s}}r{r}s70_c16_rescue_value_static" for r in (4, 8, 12)}, (4, 8, 12)),
    ]:
        load[label] = {}
        for r in rates:
            m = mean_of([pattern[r].format(s=s) for s in (1, 2)])
            if m:
                load[label][str(r)] = m
    result["ai_load_16cells"] = load

    weak = {}
    for wf in ("025", "075"):
        weak[wf] = {}
        for label, stem in [("Our Scheme", "rescue_value_backstop_corun"),
                            ("our NeuralRx rule + fixed 50% share", "rescue_value_static"),
                            ("both receivers at arrival, deadline drop + fixed 50% share", "parallel_admit_static"),
                            ("no AI", "rescue_value_none")]:
            m = mean_of([f"w{s}f{wf}_c16_{stem}" for s in (1, 2)])
            if m:
                weak[wf][label] = m
    result["weak_fraction_16cells"] = weak
    (ROOT / "novelty_evidence.json").write_text(json.dumps(result, indent=2))

    print("## value-rule generality")
    for g in result["generality"]:
        print(f"- {g['profile']}: conv fails {g['conv_fails']}/{g['tbs']}, NeuralRx rescues "
              f"{g['nrx_rescues']}, kept by rule {g['rescues_kept_by_rule']} with "
              f"{g['nrx_runs_with_rule']} runs; by failed CBs {g['by_failed_code_blocks']}")
    for rate, rows in ablation.items():
        print(f"\n## ablation, 16 cells, AI {rate} req/s per GPU")
        print("| variant | rescues | decoded | NRx runs | NRx late | late TBs | SLO tokens/s | TTFT p50 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['variant']} | {r['rescues']:.0f} | {100*r['decoded_on_time']:.2f}% | "
                  f"{r['nrx_runs']:.0f} | {r['nrx_late']:.0f} | {r['late_tbs']:.0f} | "
                  f"{r['ai_slo_tokens_per_s']:.0f} | {r['ttft_p50_ms'] or 0:.0f} |")
    print("\n## AI load, 16 cells (rescues / SLO tokens per s)")
    for label, byrate in load.items():
        print(f"- {label}: " + ", ".join(
            f"{r} req/s: {m['rescues']:.0f} / {m['ai_slo_tokens_per_s']:.0f}" for r, m in byrate.items()))
    print("\n## weak-cell fraction, 16 cells, AI 8 req/s (rescues / NRx runs / SLO tokens per s)")
    for wf, rows in weak.items():
        for label, m in rows.items():
            print(f"- {wf}: {label}: {m['rescues']:.0f} / {m['nrx_runs']:.0f} / {m['ai_slo_tokens_per_s']:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"Our Scheme": "tab:blue", "fixed 30% share": "tab:olive",
              "fixed 50% share": "tab:orange", "fixed 70% share": "tab:red"}
    for label, byrate in load.items():
        rates = sorted(int(r) for r in byrate)
        axes[0].plot(rates, [byrate[str(r)]["rescues"] for r in rates], marker="o",
                     color=colors[label], label=label)
        axes[1].plot(rates, [byrate[str(r)]["ai_slo_tokens_per_s"] / 1000 for r in rates],
                     marker="o", color=colors[label], label=label)
    axes[0].set_ylabel("TBs rescued by NeuralRx in time")
    axes[1].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
    for ax in axes:
        ax.set_xlabel("AI requests per second per GPU")
        ax.set_xticks([4, 8, 12])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("16 cells: effect of AI load", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "novelty_ai_load.png", dpi=160)
    print(ROOT / "novelty_ai_load.png")


if __name__ == "__main__":
    main()
