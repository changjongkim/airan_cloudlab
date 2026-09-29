#!/usr/bin/env python3.11
"""Compare the C177 single-GPU runs with the paired C176 multi-GPU runs.

Runs pair by (arrival pattern, policy, execution mode, offered rate). Both
placements use the same radio protocol, seeds, bounds, traces, and MPS limits,
so the comparison isolates the placement. For every run the script reads the
raw logs for the paths that co-location can change: the NeuralRx kernel and
release-to-result path, the recovery path, the AI execution per prompt
length, and the radio commit time.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RATE = re.compile(r"^c17[67](?:r(?P<rate>\d+))?$")


def stats(values) -> dict | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[round((len(values) - 1) * q)]
    return {"count": len(values), "p50": pick(0.5), "p99": pick(0.99), "max": values[-1],
            "mean": sum(values) / len(values)}


def raw_paths(raw: Path, label: str) -> dict:
    """Per-run samples of every path that co-location can change."""
    coordinator = json.loads((raw / f"{label}_coordinator.json").read_text())
    worker = json.loads((raw / f"{label}_nrx_worker.json").read_text())
    records = []
    for owner in sorted(raw.glob(f"{label}_home*_owner.json")):
        records += json.loads(owner.read_text())["records"]
    attempts = [a for r in coordinator["rounds"] for a in r["lease_attempts"] if a["launched"]]
    samples = {
        "nrx_path_ms": [r.get("nrx_release_to_complete_ms") for r in records],
        "recovery_path_ms": [r["prepad_path_ms"] for r in coordinator["physical_recoveries"]],
        "recovery_gpu_ms": [r["conventional_gpu_ms"] for r in coordinator["physical_recoveries"]],
        "commit_ms": [r.get("release_to_commit_ms") for r in records],
    }
    for attempt in attempts:
        samples.setdefault(f"ai_execution_ms_{attempt['context_length']}", []).append(attempt["execution_ms"])
        samples.setdefault(f"ai_gpu_ms_{attempt['context_length']}", []).append(attempt["gpu_ms"])
    return {"samples": samples, "nrx_kernel_ms": worker["nrx_gpu_ms"],
            "timely_nrx_successes": sum(bool(r.get("timely_success")) for r in records)}


def paths(measured: dict) -> dict:
    out = {name: stats(values) for name, values in sorted(measured["samples"].items())}
    out["nrx_kernel_ms"] = measured["nrx_kernel_ms"]
    out["timely_nrx_successes"] = measured["timely_nrx_successes"]
    return out


def key(run: dict) -> tuple:
    rate = RATE.match(run["prefix"])["rate"]
    return (run["pattern"], run["policy"], run["mode"], rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--multi", type=Path, nargs="+", required=True, help="C176 campaign JSONs")
    parser.add_argument("--single", type=Path, nargs="+", required=True, help="C177 campaign JSONs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {"multi": {}, "single": {}}
    for placement, files in (("multi", args.multi), ("single", args.single)):
        for path in files:
            for run in json.loads(path.read_text())["runs"]:
                runs[placement][key(run)] = run

    fields = ("on_time_tokens_per_s", "requests_on_time", "requests_served", "ttft_p50_ms", "ttft_p99_ms",
              "contract_broken_periods", "physical_guard_misses", "radio_deadline_misses", "radio_commits",
              "required_recoveries", "late_launches")
    pairs = []
    pooled = {"multi": {}, "single": {}}
    for k in sorted(set(runs["multi"]) & set(runs["single"]), key=lambda x: (x[3] or "", x[2], x[0], x[1])):
        multi, single = runs["multi"][k], runs["single"][k]
        measured = {"multi": raw_paths(args.raw, multi["label"]), "single": raw_paths(args.raw, single["label"])}
        if k[2] == "natural":
            for placement in ("multi", "single"):
                for name, values in measured[placement]["samples"].items():
                    pooled[placement].setdefault(name, []).extend(values)
        pairs.append({
            "pattern": k[0], "policy": k[1], "mode": k[2], "offered_rate_per_s": single["offered_rate_per_s"],
            "rate_prefix": k[3],
            "multi": {"label": multi["label"], "host": multi["host"], "gpu_models": multi.get("gpu_models"),
                      **{f: multi[f] for f in fields}, "paths": paths(measured["multi"])},
            "single": {"label": single["label"], "host": single["host"], "gpu_models": single.get("gpu_models"),
                       **{f: single[f] for f in fields}, "paths": paths(measured["single"])},
            "same_nrx_outcomes": multi["outcome_sequence_sha256"] == single["outcome_sequence_sha256"],
        })

    gains = []
    for placement in ("multi", "single"):
        for (pattern, policy, mode, rate), run in runs[placement].items():
            if policy != "backstop":
                continue
            other = runs[placement].get((pattern, "recovery_first", mode, rate))
            if other:
                gains.append({"placement": placement, "pattern": pattern, "mode": mode, "rate_prefix": rate,
                              "backstop_over_recovery_first": run["on_time_tokens_per_s"]
                              / max(other["on_time_tokens_per_s"], 1e-9)})

    safe = all(p[s]["contract_broken_periods"] == 0 and p[s]["physical_guard_misses"] == 0
               and p[s]["radio_deadline_misses"] == 0
               for p in pairs for s in ("multi", "single") if p["policy"] != "idle_time")
    result = {
        "schema": "softwall-c177-placement-comparison-v1",
        "pairs": pairs,
        "backstop_over_recovery_first": sorted(gains, key=lambda g: (g["placement"], g["mode"], g["rate_prefix"] or "",
                                                                      g["pattern"])),
        "safe_policies_zero_violations": safe,
        "pooled_natural_paths": {placement: {name: stats(values) for name, values in sorted(samples.items())}
                                 for placement, samples in pooled.items()},
        "unpaired": {p: sorted("|".join(str(x) for x in k) for k in set(runs[p]) - set(runs[o]))
                     for p, o in (("multi", "single"), ("single", "multi"))},
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("pairs:", len(pairs), "safe policies zero violations:", safe)
    for p in pairs:
        m, s = p["multi"], p["single"]
        print(f"{p['mode']:7s} {p['pattern']:10s} r={p['rate_prefix'] or '-':2s} {p['policy']:15s} "
              f"tok/s {m['on_time_tokens_per_s']:6.1f} -> {s['on_time_tokens_per_s']:6.1f}  "
              f"NRx p99 {m['paths']['nrx_path_ms']['p99']:5.1f} -> {s['paths']['nrx_path_ms']['p99']:5.1f}  "
              f"miss {m['radio_deadline_misses']}->{s['radio_deadline_misses']}  same_nrx={p['same_nrx_outcomes']}")


if __name__ == "__main__":
    main()
