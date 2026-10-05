#!/usr/bin/env python3
"""How far the rule is from an offline schedule that knows every candidate in advance.

usage: analyze_optimum.py OUT.json JOB[,JOB]:TAG:CELLS[:OURS] [...]      (OURS: policy code of the rule, default wm)

The share of the GPU time in which AI may run, with no recovery lost against the run without
AI, computed on the measured run without AI (which slots had a candidate, when each neural
receiver run started, how long it took):

  rule, ideal     AI is off on a GPU exactly while its neural receiver runs (the rule with an
                  instant stop and no other overhead):   1 - sum(S) / (N H)
  clairvoyant     the offline schedule knows every later candidate.  Start times stay as in the
                  run without AI, so no candidate is lost.  AI runs next to a neural receiver
                  for as long as the run still ends before the next run on that receiver starts:
                  a run of length S next to AI for a time a takes T = S + (1 - r) a, with
                  r = S alone / S next to AI.  Runs are assigned to receivers so that the cut
                  time is smallest (the receiver whose stretched run ends first; optimal for
                  fixed start times because the cost max(0, f - s) is convex).  AI is off for
                  (S - r T) / (1 - r) of each run.
  no rule         AI is never off (low priority alone; it loses recoveries).

Measured, per policy: AI served, recoveries kept, and the share of the time in which an AI piece
was on the GPU.  The clairvoyant AI served is the AI served by the rule scaled by the AI time
(clairvoyant / rule, measured), and at most what low priority alone serves (an upper limit for any
policy with low-priority AI; where that run is missing, at most the offered load).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from analyze_sweep import metrics
from loss_trace import RAW, trace

SLOW_RUNS = ("59210955", "59313958", "fa", "16")       # r = S alone / S next to AI: full load, low priority alone
OFFERED_PER_GPU = 53.0e3 / 4                           # tokens/s offered per GPU (32 requests/s)


def slowdown() -> float:
    alone, slowed = [], []
    for job in SLOW_RUNS[:2]:
        for path in sorted(RAW.glob(f"{SLOW_RUNS[2]}[0-9]n_c{SLOW_RUNS[3]}_*_j{job}.json")):
            t = trace(path.stem)
            alone.append(t["run_ms"][t["start_ms"] >= 0])
        for path in sorted(RAW.glob(f"{SLOW_RUNS[2]}[0-9]r32p100_c{SLOW_RUNS[3]}_*_j{job}.json")):
            t = trace(path.stem)
            slowed.append(t["run_ms"][t["start_ms"] >= 0])
    return float(np.median(np.concatenate(alone)) / np.median(np.concatenate(slowed)))


def offline(t: dict, r: float) -> dict:
    """Shares of the GPU time in which AI may run with no recovery lost (see module docstring)."""
    ran = t["start_ms"] >= 0
    period_ms = float(t["period_ms"])
    start = t["period"][ran] * period_ms + t["start_ms"][ran]
    length = t["run_ms"][ran]
    order = np.argsort(start, kind="stable")
    start, length = start[order], length[order]
    lanes = int(t["lanes"])
    horizon = (int(t["periods"]) - int(t["skip"])) * period_ms
    free_min = np.full(lanes, -1e18)          # end of the current run if it is not stretched
    free_max = np.full(lanes, -1e18)          # end if AI runs next to it all the time
    last = [None] * lanes                     # index of the current run of each receiver
    hold = length / r                         # T of every run; cut below when the receiver is needed
    cut_runs = 0
    for j, (s, S) in enumerate(zip(start, length)):
        usable = np.flatnonzero(free_min <= s + 1e-9)
        if len(usable) == 0:                  # measured starts overlap by rounding; take the earliest
            usable = np.array([int(np.argmin(free_min))])
        lane = int(usable[np.argmin(free_max[usable])])
        if last[lane] is not None and free_max[lane] > s:
            i = last[lane]
            hold[i] = max(length[i], s - start[i])
            cut_runs += 1
        free_min[lane], free_max[lane], last[lane] = s + S, s + S / r, j
    off = np.maximum(0.0, (length - r * hold) / (1.0 - r))
    return {"runs": int(len(start)), "busy_share": float(length.sum() / (lanes * horizon)),
            "rule_ideal": float(1.0 - length.sum() / (lanes * horizon)),
            "clairvoyant": float(1.0 - off.sum() / (lanes * horizon)),
            "cut_runs_share": cut_runs / max(1, len(start)),
            "off_per_cut_run_ms": float(off.sum() / max(1, cut_runs))}


def ai_on_share(path: Path) -> float:
    """Share of the time in which an AI piece was on the GPU (mean over the GPUs)."""
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    period = float(config["period_ms"]) * 1e6
    head = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = head[2] - head[0] * period
    begin, end = epoch + int(config.get("skip_periods", 20)) * period, epoch + int(config["periods"]) * period
    shares = []
    for file in sorted(work.glob("ai*.json")):
        pieces = json.loads(file.read_text())["pieces"]
        on = sum(max(0.0, min(p[1], end) - max(p[0], begin)) for p in pieces)
        shares.append(on / (end - begin))
    return float(np.mean(shares)) if shares else 0.0


def main() -> None:
    out_path = Path(sys.argv[1])
    r = slowdown()
    print(f"neural receiver alone / next to low-priority AI: r = {r:.3f}\n")
    print("| condition | seeds | neural receiver busy | AI time: rule, ideal | rule, measured | clairvoyant | "
          "runs cut by the clairvoyant | AI served: rule | low priority alone | clairvoyant (estimate) | rule / clairvoyant |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    out = {"slowdown_r": r, "conditions": {}}
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        jobs, tag, cells = parts[0].split(","), parts[1], parts[2]
        ours = parts[3] if len(parts) > 3 else "wm"
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r32([a-z0-9]+))_c{cells}_")
        model, rule_on, rule_tok, rule_kept, free_on, free_tok, free_kept = [], [], [], [], [], [], []
        others: dict[str, list] = {}
        refs = {}
        for job in jobs:
            for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
                m = pattern.match(path.name)
                if not m:
                    continue
                if m.group(2) is None:
                    refs[(job, m.group(1))] = metrics(path)
                    model.append(offline(trace(path.stem), r))
        for job in jobs:
            for path in sorted(RAW.glob(f"{tag}[0-9]r32*_c{cells}_*_j{job}.json")):
                m = pattern.match(path.name)
                if not m or (job, m.group(1)) not in refs:
                    continue
                kept = lambda v: 100.0 * v["recovered"] / max(1, refs[(job, m.group(1))]["recovered"])
                if m.group(2) == ours:
                    v = metrics(path)
                    rule_on.append(ai_on_share(path)), rule_tok.append(v["ai_slo"]), rule_kept.append(kept(v))
                elif m.group(2) == "p100":
                    v = metrics(path)
                    free_on.append(ai_on_share(path)), free_tok.append(v["ai_slo"]), free_kept.append(kept(v))
                if m.group(2) in os.environ.get("OTHERS", "").split(","):
                    v = metrics(path)
                    others.setdefault(m.group(2), []).append((v["ai_slo"], kept(v)))
        if not model or not rule_tok:
            continue
        mean = lambda values: float(np.mean(values)) if values else float("nan")
        row = {"seeds": len(model), "busy": mean([m["busy_share"] for m in model]),
               "rule_ideal": mean([m["rule_ideal"] for m in model]), "clairvoyant": mean([m["clairvoyant"] for m in model]),
               "cut_runs_share": mean([m["cut_runs_share"] for m in model]),
               "rule_measured_on": mean(rule_on), "rule_tokens": mean(rule_tok), "rule_kept": mean(rule_kept),
               "free_on": mean(free_on), "free_tokens": mean(free_tok), "free_kept": mean(free_kept)}
        gpus = int(json.loads(next(iter(sorted(RAW.glob(f"{tag}[0-9]n_c{cells}_*_j{jobs[0]}.json")))).read_text())["config"]["num_gpus"])
        cap = row["free_tokens"] if free_tok else OFFERED_PER_GPU * gpus
        estimate = min(cap, row["rule_tokens"] * row["clairvoyant"] / max(1e-9, row["rule_measured_on"]))
        row["clairvoyant_tokens"], row["cap_tokens"], row["cap_is_measured"] = estimate, cap, bool(free_tok)
        row["others"] = {code: {"tokens": float(np.mean([a for a, _ in values])), "kept": float(np.mean([b for _, b in values])),
                                "seeds": len(values), "of_clairvoyant": float(np.mean([a for a, _ in values])) / estimate}
                         for code, values in others.items()}
        out["conditions"][f"{tag} c{cells}"] = row
        k = lambda v: "–" if v != v else f"{v / 1e3:.1f}k"
        print(f"| {tag} c{cells} | {row['seeds']} | {100 * row['busy']:.0f}% | {100 * row['rule_ideal']:.0f}% | "
              f"{100 * row['rule_measured_on']:.0f}% | {100 * row['clairvoyant']:.1f}% | {100 * row['cut_runs_share']:.1f}% | "
              f"{k(row['rule_tokens'])} @ {row['rule_kept']:.1f}% | "
              + (f"{k(row['free_tokens'])} @ {row['free_kept']:.1f}% (AI on {100 * row['free_on']:.0f}%)" if free_tok else f"– (offered {k(cap)})")
              + f" | {k(estimate)} | " + ("–" if estimate != estimate else f"{100 * row['rule_tokens'] / estimate:.0f}%") + " |")
    for name, row in out["conditions"].items():
        if row["others"]:
            print(f"\n{name}: other measured policies against the clairvoyant estimate ({row['clairvoyant_tokens'] / 1e3:.1f}k tokens/s)")
            print("| policy | seeds | AI served | recoveries kept | share of the clairvoyant |")
            print("|---|---|---|---|---|")
            print(f"| the rule | {row['seeds']} | {row['rule_tokens'] / 1e3:.1f}k | {row['rule_kept']:.1f}% | {100 * row['rule_tokens'] / row['clairvoyant_tokens']:.0f}% |")
            for code, v in sorted(row["others"].items(), key=lambda kv: -kv[1]["kept"]):
                print(f"| {code} | {v['seeds']} | {v['tokens'] / 1e3:.1f}k | {v['kept']:.1f}% | {100 * v['of_clairvoyant']:.0f}% |")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
