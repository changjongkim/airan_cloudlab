#!/usr/bin/env python3
"""Where the GPU time goes with the SM slice option (v21, v22).

For every run of the given tags: the share of GPU time in which the neural receiver runs, in which the AI runs on
the whole GPU, and in which it runs on its slice (split into the time next to the neural receiver and the time after
the neural receiver has ended), and how long an AI piece takes against its bound on the whole GPU.

usage: analyze_slice.py OUT.json JOB TAG[,TAG..] [SEEDS]      (run names <tag><seed><policy>_c*_j<JOB>)
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(intervals):
    return sum(b - a for a, b in intervals)


def pick(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else None


def one_run(work: Path) -> dict:
    lanes = [json.loads(p.read_text()) for p in sorted(work.glob("lane*.json"))]
    span = nrx_t = whole_t = next_t = after_t = 0
    on_whole, next_to, alone = [], [], []
    tokens = 0.0
    factor = 1.0
    for path in sorted(work.glob("ai*.json")):
        d = json.loads(path.read_text())
        tokens += d["summary"]["tokens_within_slo_per_s"]
        g = d["gpu"]
        nrx = merge([(r[2], r[3]) for lane in lanes if lane["gpu"] == g for r in lane["records"]])
        if not nrx:
            continue
        starts = [a for a, _ in nrx]
        span += nrx[-1][1] - nrx[0][0]
        nrx_t += total(nrx)
        for p in d["pieces"]:
            a, b, sliced = p[0], p[1], len(p) > 5 and p[5]
            if not sliced:
                whole_t += b - a
                on_whole.append((b - a) / p[3])
                continue
            k = max(0, bisect.bisect_right(starts, a) - 1)
            shared = 0
            while k < len(nrx) and nrx[k][0] < b:
                shared += max(0, min(b, nrx[k][1]) - max(a, nrx[k][0]))
                k += 1
            next_t += shared
            after_t += (b - a) - shared
            # p[3] is the bound on the slice (bound on the whole GPU x slice_factor)
            factor = float(d["summary"].get("slice_factor", 0.0)) or factor
            if shared >= 0.98 * (b - a):
                next_to.append((b - a) / p[3])
            elif shared == 0:
                alone.append((b - a) / p[3])
    runs = sorted((r[3] - r[2]) / 1e6 for lane in lanes for r in lane["records"])
    return {"ai_tokens_per_s": tokens, "lane_sms": sorted({lane.get("sms", 0) for lane in lanes}),
            "nrx_busy": nrx_t / span if span else None, "ai_whole": whole_t / span if span else None,
            "ai_slice_next_to_nrx": next_t / span if span else None, "ai_slice_after_nrx": after_t / span if span else None,
            "nrx_run_ms": {"p50": pick(runs, 0.5), "p90": pick(runs, 0.9), "p99": pick(runs, 0.99), "n": len(runs)},
            "piece_wall_over_bound": {"whole_gpu_p50": pick(on_whole, 0.5), "slice_next_to_nrx_p50": pick(next_to, 0.5),
                                      "slice_alone_p50": pick(alone, 0.5),
                                      "n": [len(on_whole), len(next_to), len(alone)]}}


def main() -> None:
    out, job, tags = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
    seeds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["1", "2"]
    report = {}
    for tag in tags:
        for seed in seeds:
            for work in sorted(RAW.glob(f"{tag}{seed}*_j{job}_work")):
                policy = work.name[len(tag) + len(seed):].split("_c")[0]
                result = RAW / (work.name[:-5] + ".json")
                if not result.is_file():
                    continue                    # the run has not ended
                config = json.loads(result.read_text())["config"]
                r = one_run(work)
                r["slice_sms"] = config["ai"].get("slice_sms", 0)
                r["slice_factor"] = config["ai"].get("slice_factor", 1.0)
                # bounds on the slice already hold slice_factor: give the ratios against the whole-GPU bound
                for key in ("slice_next_to_nrx_p50", "slice_alone_p50"):
                    if r["piece_wall_over_bound"][key] is not None:
                        r["piece_wall_over_bound"][key] *= r["slice_factor"]
                report.setdefault(tag, {}).setdefault(policy, {})[seed] = r
    out.write_text(json.dumps(report, indent=1))
    pct = lambda v: "   - " if v is None else f"{100 * v:5.1f}"
    num = lambda v: "  - " if v is None else f"{v:4.1f}"
    print("share of GPU time (%): neural receiver | AI on the whole GPU | AI on the slice next to it | AI on the slice after it ended")
    for tag, policies in report.items():
        for policy, by_seed in policies.items():
            rows = list(by_seed.values())
            mean = lambda f: (lambda v: sum(v) / len(v) if v else None)([f(r) for r in rows if f(r) is not None])
            w = lambda key: mean(lambda r: r["piece_wall_over_bound"][key])
            print(f"{tag} {policy:9s} AI {mean(lambda r: r['ai_tokens_per_s']) / 1e3:5.1f}k | {pct(mean(lambda r: r['nrx_busy']))} "
                  f"{pct(mean(lambda r: r['ai_whole']))} {pct(mean(lambda r: r['ai_slice_next_to_nrx']))} "
                  f"{pct(mean(lambda r: r['ai_slice_after_nrx']))} | neural receiver run p50 {mean(lambda r: r['nrx_run_ms']['p50']):.2f} "
                  f"p99 {mean(lambda r: r['nrx_run_ms']['p99']):.2f} ms on {rows[0]['lane_sms']} | piece time / whole-GPU bound: "
                  f"whole {num(w('whole_gpu_p50'))}, slice next to it {num(w('slice_next_to_nrx_p50'))}, slice alone {num(w('slice_alone_p50'))}"
                  f" ({len(rows)} seeds)")


if __name__ == "__main__":
    main()
