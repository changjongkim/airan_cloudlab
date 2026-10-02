#!/usr/bin/env python3
"""One sweep condition: every policy against the no-AI run of the same seed.

usage: analyze_sweep.py JOB[,JOB...] TAG CELLS [--json]
Runs are named  <TAG><seed>n  (no AI) and  <TAG><seed>r<AI rate><policy>  by v13.sh.
Per policy and AI rate, mean over seeds (range in the JSON):
  AI served      prompt tokens per second finished within the time limit
  L1 late        TBs whose conventional result missed the L1 deadline
  recovered      TBs the neural receiver decoded by the recovery deadline, per second,
                 and as a share of the no-AI run
  NRx on time    neural receiver runs that finished by the recovery deadline
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
POLICY_NAMES = {
    "vf": "Our Scheme", "i": "Radio-idle only", "p100": "Low priority, no cap",
    "v4": "Our Scheme of v7-v12 (no priority, no lane reserve)", "ve": "Ours, AI never next to NeuralRx",
    "vd": "Ours, lane reserve 2", "vc": "Ours, 70% cap", "vb": "Ours, 70% cap, lane reserve 2",
    "vg": "Ours, pieces of 2.5 ms", "vh": "Ours, pieces of 1.5 ms", "vi": "Ours, pieces of 2.5 ms, lane reserve 2",
    "vj": "Ours, pieces of 1.5 ms, lane reserve 2",
    "va": "Ours, 70% cap, AI never next to NeuralRx", "v8": "Ours, 70% cap, reserve 2, 128-token units next to conventional",
}


def policy_name(code: str) -> str:
    if code in POLICY_NAMES:
        return POLICY_NAMES[code]
    if code[0] in "sp" and code[1:].isdigit():
        return f"Fixed {code[1:]}%" + (" + low priority" if code[0] == "p" else "")
    if code[0] in "de":
        shares, lag = code[1:].split("l")
        late = "" if lag == "0" else f", {int(lag) * 2.5 / 1000:g} s late"
        return f"Share follows load {shares.replace('x', '/')}%" + (" + low priority" if code[0] == "e" else "") + late
    return code


def metrics(path: Path) -> dict:
    result = json.loads(path.read_text())
    config, head, everything = result["config"], result["headline"], result["all"]
    seconds = (int(config["periods"]) - int(config.get("skip_periods", 20))) * float(config["period_ms"]) / 1e3
    ai = head.get("ai_total") or {}
    runs = everything["nrx_runs"]
    ttft = [v["ttft_ms"].get("p99") for v in (result.get("ai") or {}).values() if v and v["ttft_ms"].get("n")]
    return {
        "tbs": head["tbs"],
        "tbs_per_s": head["tbs"] / seconds,
        "l1_late_pct": 100.0 * head["late_tbs"] / max(1, head["tbs"]),
        "l1_p999_ms": everything["conv_done_ms"].get("p999"),
        "conv_fail_tbs": everything["conv_fail_tbs"],
        "recovered": head["nrx_rescues_on_time"],
        "recovered_per_s": head["nrx_rescues_on_time"] / seconds,
        "nrx_runs_per_s": runs / seconds,
        "nrx_on_time_pct": 100.0 * (1.0 - everything["nrx_late"] / runs) if runs else None,
        "decoded_pct": 100.0 * head["decoded_final"],
        "ai_slo": ai.get("tokens_within_slo_per_s", 0.0),
        "ai_all": ai.get("tokens_per_s", 0.0),
        "ai_ttft_p99_ms": max(ttft) if ttft else None,
    }


def collect(jobs: list[str], tag: str, cells: str) -> tuple[dict, dict]:
    refs, rows = {}, {}
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
    for job in jobs:
        for path in sorted((ROOT / "raw").glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m:
                continue
            seed = int(m.group(1))
            if m.group(2) is None:
                refs[(job, seed)] = metrics(path)
                refs.setdefault(seed, refs[(job, seed)])      # the first job listed gives the no-AI row
            else:
                # A seed that was run again in a later job keeps the run of the first job listed.
                rows.setdefault((int(m.group(2)), m.group(3)), {}).setdefault(seed, dict(metrics(path), job=job))
    return refs, rows


def summarize(refs: dict, rows: dict) -> list[dict]:
    out = []
    if refs:
        keys = ("l1_late_pct", "l1_p999_ms", "recovered_per_s", "nrx_runs_per_s", "nrx_on_time_pct", "decoded_pct",
                "tbs_per_s", "conv_fail_tbs")
        by_seed = {k: v for k, v in refs.items() if isinstance(k, int)}
        out.append({"policy": "n", "name": "No AI", "ai_rate": 0, "seeds": sorted(by_seed),
                    **{k: float(np.mean([r[k] for r in by_seed.values() if r[k] is not None] or [np.nan])) for k in keys},
                    "ai_slo": 0.0, "recovered_pct": 100.0})
    for (rate, policy), by_seed in sorted(rows.items()):
        items = []
        for seed, m in by_seed.items():
            ref = refs.get((m.get("job"), seed), refs.get(seed))     # the no-AI run of the same job if there is one
            m = dict(m)
            m["recovered_pct"] = 100.0 * m["recovered"] / ref["recovered"] if ref and ref["recovered"] else None
            items.append(m)
        row = {"policy": policy, "name": policy_name(policy), "ai_rate": rate, "seeds": sorted(by_seed)}
        for key in ("ai_slo", "ai_all", "l1_late_pct", "l1_p999_ms", "recovered_per_s", "recovered_pct",
                    "nrx_runs_per_s", "nrx_on_time_pct", "decoded_pct", "ai_ttft_p99_ms", "tbs_per_s"):
            values = [i[key] for i in items if i.get(key) is not None]
            if values:
                row[key] = float(np.mean(values))
                row[key + "_min"], row[key + "_max"] = float(min(values)), float(max(values))
        out.append(row)
    return out


def fmt(row: dict, key: str, spec: str, scale: float = 1.0) -> str:
    return format(row[key] * scale, spec) if row.get(key) is not None else "–"


def main() -> None:
    jobs, tag, cells = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    refs, rows = collect(jobs, tag, cells)
    out = summarize(refs, rows)
    print("| AI load | policy | seeds | AI served (tokens/s) | L1 late | L1 p99.9 ms | recovered /s | recovered kept | NRx runs /s | NRx on time |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in out:
        print(f"| {row['ai_rate']} | {row['name']} | {len(row['seeds'])} | {fmt(row, 'ai_slo', '.1f', 1e-3)}k | "
              f"{fmt(row, 'l1_late_pct', '.3f')}% | {fmt(row, 'l1_p999_ms', '.2f')} | {fmt(row, 'recovered_per_s', '.1f')} | "
              f"{fmt(row, 'recovered_pct', '.1f')}% | {fmt(row, 'nrx_runs_per_s', '.1f')} | {fmt(row, 'nrx_on_time_pct', '.1f')}% |")
    (ROOT / f"sweep_{tag}_c{cells}_j{'_'.join(jobs)}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
