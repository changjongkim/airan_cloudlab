#!/usr/bin/env python3
"""Scheduling metrics per policy from the per-process records of the runs: protection of the radio
work, latency of the AI requests, and use of the GPU time.

usage: analyze_sched.py JOB[,JOB] TAG CELLS POLICY ...      (POLICY n = the run without AI)
       RATE=16 analyze_sched.py ...     only the runs with this AI rate code (tags with several offered AI loads)
       WARMUP_S=5 analyze_sched.py ...  leave out the first seconds of every run (the processes start cold: in a
                                        100-s run nearly all TBs past the L1 deadline are in the first 3.5 s);
                                        the output file gets the suffix _w5

Per policy (all seeds; a seed that ran in two jobs keeps the run of the first job listed, as in
analyze_sweep.py):
  radio   L1 latency of a TB (samples arrive -> conventional result) p50 / p99 / p99.9 and the share
          past the L1 deadline; neural receiver run time p50 / p99 and the share past the recovery
          deadline; recoveries kept against the run without AI of the same seed
  AI      tokens served within the time limit; latency of a request (arrival -> prefill done)
          p50 / p99; share of the placed requests inside the limit; share of the arriving requests
          that the dispatcher rejects
  GPU     share of the time in which a GPU has work of any kind on it (active), radio work
          (conventional receiver or neural receiver), an AI piece, an AI piece next to a neural
          receiver, an AI piece next to a conventional receiver, and nothing (idle); mean of the GPUs

Writes results/backstop_slot/sched_TAG_cCELLS_jJOBS.json and prints a table.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
RAW = ROOT / "raw"
STEP = 20_000            # ns


def mask(spans, origin: int, n: int) -> np.ndarray:
    g = np.zeros(n + 1, dtype=np.int32)
    for a, b in spans:
        i, j = max(0, (a - origin) // STEP), min(n, (b - origin) // STEP)
        if j > i:
            g[i] += 1
            g[j] -= 1
    return np.cumsum(g[:-1]) > 0


def one(path: Path) -> dict:
    result = json.loads(path.read_text())
    config, head = result["config"], result["headline"]
    work = Path(str(path)[:-5] + "_work")
    period = int(float(config["period_ms"]) * 1e6)
    skip, periods, gpus = int(config.get("skip_periods", 20)), int(config["periods"]), int(config["num_gpus"])
    skip = max(skip, int(float(os.environ.get("WARMUP_S", "0")) * 1e9 / period))
    deadline = float(config["deadline_ms"])
    recovery_deadline = float(config.get("rescue_deadline_ms", config["deadline_ms"]))
    first = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = first[2] - first[0] * period
    origin, n = epoch + skip * period, (periods - skip) * period // STEP
    seconds = (periods - skip) * period / 1e9

    l1, conv_spans, conv_good = [], {g: [] for g in range(gpus)}, {}
    for cell in config["cells"]:
        g, ues = int(cell["gpu"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{int(cell['cell'])}.json").read_text())["records"]:
            if r[0] < skip:
                continue
            l1.extend([(r[4] - r[2]) / 1e6] * ues)
            conv_spans[g].append((r[3], r[4]))
            conv_good[(int(cell["cell"]), r[0])] = int(r[9])
    runs, done, use = [], [], {k: [] for k in ("active", "radio", "ai", "ai_nrx", "ai_conv", "idle", "nrx", "conv")}
    latency, placed, placed_run, recovered, tokens_in_limit = [], 0, 0, 0, 0
    slo = float((config.get("ai") or {}).get("slo_ms", 200.0))
    for g in range(gpus):
        lane = [r for r in json.loads((work / f"lane{g}.json").read_text())["records"] if r[1] >= skip]
        runs += [(r[3] - r[2]) / 1e6 for r in lane]
        done += [(r[3] - (epoch + r[1] * period)) / 1e6 for r in lane]
        for r in lane:      # TBs that the conventional receiver failed and the neural receiver decoded in time
            if r[3] - (epoch + r[1] * period) <= recovery_deadline * 1e6:
                recovered += bin(int(r[7]) & ~conv_good.get((int(r[0]), int(r[1])), 0)).count("1")
        pieces = []
        for file in sorted(work.glob(f"ai{g}.json")) + sorted(work.glob(f"ai{g}s*.json")):
            data = json.loads(file.read_text())
            pieces += [(p[0], p[1]) for p in data["pieces"]]
            for arrival, length, _, finished in data["requests"]:
                placed_run += 1
                if epoch + arrival < origin:
                    continue
                placed += 1
                if finished is not None:
                    latency.append((finished - (int(data.get("epoch_ns", epoch)) + arrival)) / 1e6)
                    tokens_in_limit += int(length) * int(latency[-1] <= slo)
        conv, nrx, ai = mask(conv_spans[g], origin, n), mask([(r[2], r[3]) for r in lane], origin, n), mask(pieces, origin, n)
        radio = conv | nrx
        for key, value in (("active", radio | ai), ("radio", radio), ("ai", ai), ("ai_nrx", ai & nrx), ("ai_conv", ai & conv),
                           ("idle", ~(radio | ai)), ("nrx", nrx), ("conv", conv)):
            use[key].append(float(value.mean()))
    counters = (result.get("controller") or {}).get("counters") or {}
    # The controller counts the rejected requests of the whole run: the requests that arrive in the window are
    # taken as the same share of all arrivals as the window is of the run.
    rejected = int(counters.get("ai_rejected", 0))
    arrivals = (placed_run + rejected) * (periods - skip) / max(1, periods - int(config.get("skip_periods", 20)))
    rejected = max(0.0, arrivals - placed) if skip > int(config.get("skip_periods", 20)) else rejected
    l1, runs, done, latency = np.array(l1), np.array(runs), np.array(done), np.array(latency)
    ai_total = head.get("ai_total") or {}
    return {
        "l1_p50_ms": float(np.percentile(l1, 50)), "l1_p99_ms": float(np.percentile(l1, 99)),
        "l1_p999_ms": float(np.percentile(l1, 99.9)), "l1_late_pct": float(100.0 * (l1 > deadline).mean()),
        "l1_p9999_ms": float(np.percentile(l1, 99.99)), "l1_max_ms": float(l1.max()),
        "l1_late_tbs": int((l1 > deadline).sum()), "l1_tbs": int(len(l1)),
        "nrx_run_p50_ms": float(np.percentile(runs, 50)) if len(runs) else None,
        "nrx_run_p99_ms": float(np.percentile(runs, 99)) if len(runs) else None,
        "nrx_late_pct": float(100.0 * (done > recovery_deadline).mean()) if len(done) else None,
        "recovered": recovered, "recovered_whole_run": head["nrx_rescues_on_time"],
        "ai_slo": tokens_in_limit / seconds, "ai_slo_whole_run": ai_total.get("tokens_within_slo_per_s", 0.0),
        "ai_all": ai_total.get("tokens_per_s", 0.0),
        "ai_p50_ms": float(np.percentile(latency, 50)) if len(latency) else None,
        "ai_p99_ms": float(np.percentile(latency, 99)) if len(latency) else None,
        "ai_in_limit_pct": float(100.0 * (latency <= slo).sum() / placed) if placed else None,
        "ai_rejected_pct": float(100.0 * rejected / (placed + rejected)) if placed + rejected else None,
        "ai_gpu_s_per_s": float(ai_total.get("gpu_busy_ms", 0.0)) / 1e3 / seconds / gpus,
        **{f"gpu_{k}_pct": float(100.0 * np.mean(v)) for k, v in use.items()},
    }


def main() -> None:
    jobs, tag, cells, policies = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4:]
    rate = os.environ.get("RATE", "")
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r{rate or '[0-9]+'}([a-z][a-z0-9]*))_c{cells}_")
    paths: dict[str, dict[int, Path]] = {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if m:
                paths.setdefault(m.group(2) or "n", {}).setdefault(int(m.group(1)), path)
    out, reference, base = [], {}, None
    for policy in ["n"] + [p for p in policies if p != "n"]:
        by_seed = {seed: one(path) for seed, path in sorted(paths.get(policy, {}).items())}
        if not by_seed:
            continue
        if policy == "n":
            reference = {seed: m["recovered"] for seed, m in by_seed.items()}
        row = {"policy": policy, "seeds": sorted(by_seed)}
        for m in by_seed.values():
            m.pop("recovered_pct", None)
        for seed, m in by_seed.items():
            m["recovered_pct"] = 100.0 * m["recovered"] / reference[seed] if reference.get(seed) else None
        for key in next(iter(by_seed.values())):
            values = [m[key] for m in by_seed.values() if m.get(key) is not None]
            if values and key not in ("recovered", "l1_late_tbs", "l1_tbs"):
                row[key] = float(np.mean(values))
                row[key + "_min"], row[key + "_max"] = float(min(values)), float(max(values))
        # L1 misses of all seeds together, with a 95% interval (Wilson) and the test against the run without AI
        late, tbs = sum(m["l1_late_tbs"] for m in by_seed.values()), sum(m["l1_tbs"] for m in by_seed.values())
        z, share = 1.96, late / tbs
        centre = (share + z * z / (2 * tbs)) / (1 + z * z / tbs)
        half = z * np.sqrt(share * (1 - share) / tbs + z * z / (4 * tbs * tbs)) / (1 + z * z / tbs)
        row.update(l1_late_tbs=late, l1_tbs=tbs, l1_late_pooled_pct=100.0 * share,
                   l1_late_low_pct=100.0 * (centre - half), l1_late_high_pct=100.0 * (centre + half))
        if policy == "n":
            base = (late, tbs)
        elif base:
            pooled = (late + base[0]) / (tbs + base[1])
            se = np.sqrt(pooled * (1 - pooled) * (1 / tbs + 1 / base[1]))
            row["l1_late_z_vs_no_ai"] = float((share - base[0] / base[1]) / se) if se > 0 else 0.0
        out.append(row)
    warm = os.environ.get("WARMUP_S", "")
    target = ROOT / f"sched_{tag}{'r' + rate if rate else ''}_c{cells}_j{'_'.join(jobs)}{'_w' + warm if warm else ''}.json"
    target.write_text(json.dumps(out, indent=2))
    f = lambda r, k, s: format(r[k], s) if r.get(k) is not None else "–"
    print("| policy | seeds | AI in limit (k tok/s) | recoveries kept | L1 p50 / p99 / p99.9 (ms) | L1 late | NeuralRx run p50 / p99 (ms) | "
          "AI request p50 / p99 (ms) | in limit | rejected | GPU active | radio | AI | AI next to NeuralRx | idle |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['policy']} | {len(r['seeds'])} | {r['ai_slo'] / 1e3:.1f} | {f(r, 'recovered_pct', '.1f')}% | "
              f"{r['l1_p50_ms']:.2f} / {r['l1_p99_ms']:.2f} / {r['l1_p999_ms']:.2f} | {r['l1_late_pct']:.3f}% | "
              f"{f(r, 'nrx_run_p50_ms', '.2f')} / {f(r, 'nrx_run_p99_ms', '.2f')} | {f(r, 'ai_p50_ms', '.0f')} / {f(r, 'ai_p99_ms', '.0f')} | "
              f"{f(r, 'ai_in_limit_pct', '.1f')}% | {f(r, 'ai_rejected_pct', '.1f')}% | {r['gpu_active_pct']:.1f}% | {r['gpu_radio_pct']:.1f}% | "
              f"{r['gpu_ai_pct']:.1f}% | {r['gpu_ai_nrx_pct']:.1f}% | {r['gpu_idle_pct']:.1f}% |")
    print()
    print("| policy | L1 late TBs / TBs | pooled | 95% interval | z against no AI | L1 p99.99 (ms) | L1 max (ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['policy']} | {r['l1_late_tbs']} / {r['l1_tbs']} | {r['l1_late_pooled_pct']:.4f}% | "
              f"{r['l1_late_low_pct']:.4f}-{r['l1_late_high_pct']:.4f}% | {f(r, 'l1_late_z_vs_no_ai', '+.1f')} | "
              f"{r['l1_p9999_ms']:.2f} | {r['l1_max_ms']:.2f} |")
    print(target)


if __name__ == "__main__":
    main()
