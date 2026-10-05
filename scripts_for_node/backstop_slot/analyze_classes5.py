#!/usr/bin/env python3
"""Runs with several kinds of AI work (ai_worker5): radio results and what every class got.

usage: analyze_classes5.py JOB[,JOB] TAG CELLS
Per policy (mean over seeds): recovered TBs kept and L1 misses, then per class
  chat     prompt tokens whose first token met its limit (per s), output tokens on time (per s),
           share of the finished responses with every token on time
  prefill  prompt tokens within the limit (per s)
  encoder  items within the limit (per s)
  batch    prompt tokens (per s), no limit
and the share of the offered requests that the admission rejected.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np

from analyze_sweep import ROOT, metrics, policy_name


def class_rows(path) -> dict:
    result = json.loads(path.read_text())
    seconds = int(result["config"]["periods"]) * float(result["config"]["period_ms"]) / 1e3
    out: dict[str, dict] = {}
    for summary in (result.get("ai") or {}).values():
        for name, c in (summary or {}).get("classes", {}).items():
            row = out.setdefault(name, {"kind": c["kind"], "arrived": 0, "rejected": 0, "limit": 0.0, "extra": 0.0,
                                        "responses": 0, "responses_ok": 0, "delay": []})
            row["arrived"] += c.get("arrived", 0)
            row["rejected"] += c.get("rejected", 0)
            if c["kind"] == "encoder":
                row["limit"] += c.get("items_within_limit_per_s", 0.0)
                row["delay"].append(c["latency_ms"].get("p99", np.nan))
            elif c["kind"] == "batch":
                row["limit"] += c.get("prompt_tokens_per_s", 0.0)
            else:
                row["limit"] += c.get("prompt_tokens_within_limit_per_s", 0.0)
                row["delay"].append(c["ttft_ms"].get("p99", np.nan))
                if c["kind"] == "chat":
                    row["extra"] += c.get("output_tokens_on_time_per_s", 0.0)
                    row["responses"] += c.get("responses_completed", 0)
                    row["responses_ok"] += c.get("responses_fully_on_time", 0)
    # With server-level dispatch the controller admits or rejects; workers only see what they got.
    for name, c in ((result.get("controller") or {}).get("ai_classes") or {}).items():
        if name in out:
            out[name]["arrived"] = c["arrived"]
            out[name]["rejected"] += c["rejected"]
    return out


def main() -> None:
    jobs, tag, cells = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
    refs, runs = {}, {}
    for job in jobs:
        for path in sorted((ROOT / "raw").glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m:
                continue
            if m.group(2) is None:
                refs[(job, int(m.group(1)))] = metrics(path)
            else:      # a policy keeps the runs of the first job listed that has it; reference: the no-AI run of that job
                runs.setdefault(m.group(3), {}).setdefault(int(m.group(1)), (metrics(path), class_rows(path), job))
    names = sorted({n for by_seed in runs.values() for _, c, _ in by_seed.values() for n in c})
    print("| policy | seeds | recovered kept | L1 late | " + " | ".join(names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    out = {}
    for policy, by_seed in sorted(runs.items()):
        seeds = [s for s in by_seed if (by_seed[s][2], s) in refs]
        if not seeds:
            continue
        kept = float(np.mean([100 * by_seed[s][0]["recovered"] / max(1, refs[(by_seed[s][2], s)]["recovered"]) for s in seeds]))
        l1 = float(np.mean([by_seed[s][0]["l1_late_pct"] for s in seeds]))
        name = {"wm": "Antiphase", "wn": "All unit sizes next to the conventional receiver",
                "wr": "All unit sizes, next to NeuralRx while 3 free"}.get(policy) or policy_name(policy)
        cells_out, row = [], {"name": name, "seeds": len(seeds),
                              "recovered_pct": kept, "l1_late_pct": l1, "classes": {}}
        for name in names:
            items = [by_seed[s][1].get(name) for s in seeds if by_seed[s][1].get(name)]
            if not items:
                cells_out.append("–")
                continue
            kind = items[0]["kind"]
            limit = float(np.mean([i["limit"] for i in items]))
            rejected = 100 * sum(i["rejected"] for i in items) / max(1, sum(i["arrived"] for i in items))
            entry = {"kind": kind, "within_limit_per_s": limit, "rejected_pct": rejected}
            if kind == "chat":
                extra = float(np.mean([i["extra"] for i in items]))
                ok = 100 * sum(i["responses_ok"] for i in items) / max(1, sum(i["responses"] for i in items))
                entry.update({"output_tokens_on_time_per_s": extra, "responses_on_time_pct": ok})
                cells_out.append(f"{limit / 1e3:.1f}k prompt, {extra:.0f} out/s, {ok:.0f}% responses on time, {rejected:.0f}% rejected")
            elif kind == "encoder":
                cells_out.append(f"{limit:.0f} items/s, {rejected:.0f}% rejected")
            elif kind == "batch":
                cells_out.append(f"{limit / 1e3:.1f}k tokens/s")
            else:
                cells_out.append(f"{limit / 1e3:.1f}k tokens/s, {rejected:.0f}% rejected")
            row["classes"][name] = entry
        out[policy] = row
        print(f"| {row['name']} | {len(seeds)} | {kept:.1f}% | {l1:.3f}% | " + " | ".join(cells_out) + " |")
    if refs:
        print(f"\nno AI: L1 late {np.mean([r['l1_late_pct'] for r in refs.values()]):.3f}%, "
              f"recovered {np.mean([r['recovered_per_s'] for r in refs.values()]):.0f} TBs/s")
    (ROOT / f"classes5_{tag}_c{cells}_j{'_'.join(jobs)}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
