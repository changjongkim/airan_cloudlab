"""Server-level arrivals of several kinds of AI work (classes5), shared by controller and workers.

``plan5`` gives one arrival list for the whole server (rate = per-GPU rate x GPUs), in arrival
order.  The controller assigns every request to a GPU (global dispatch) and the worker of that
GPU takes it; both build the same list from the same config, so the shared request table only
carries the assignment.  The filler class (kind ``batch``) has no arrivals: every GPU has an
always-full queue of its own.

``first_work_ns`` is the GPU time, in unit bounds, a request needs until its time limit is met:
prompt processing for the LLM classes, the per-item share of a batch for the encoder classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def tier_of(limit_ms: float) -> int:
    """Admission counts only the backlog of work whose time limit is as short or shorter:
    tier 0 up to 100 ms, tier 1 up to 250 ms, tier 2 above."""
    return 0 if limit_ms <= 100.0 else 1 if limit_ms <= 250.0 else 2


def class_costs(spec: dict, bounds_all: dict) -> dict:
    """Unit bounds (ns) of a class, from the bounds file of ``bench_units5.py``."""
    model = bounds_all[spec["model"]]
    units = model["units_ms"]
    ns = lambda ms: int(ms * 1e6)
    if spec["kind"] == "encoder":
        batches = tuple(int(b) for b in spec.get("batches", (1, 4, 16)))
        return {"layers": int(model["layers"]), "batches": batches,
                "cost": {b: {u: ns(units[str(b)][u]["bound"]) for u in ("embed", "layer", "pool")} for b in batches}}
    chunks = tuple(int(c) for c in spec.get("chunks", (128, 512, 1024)))
    groups = int(model.get("decode_groups", 4))
    return {"layers": int(model["layers"]), "chunks": chunks, "groups": groups,
            "cost": {c: {u: ns(units[str(c)][u]["bound"]) for u in ("prep", "layer", "head")} for c in chunks},
            "decode_cost": [ns(units["decode"][f"dec{g}"]["bound"]) for g in range(groups)]}


def first_work_ns(spec: dict, costs: dict, prompt_len: int) -> int:
    if spec["kind"] == "encoder":
        b = costs["batches"][min(1, len(costs["batches"]) - 1)]
        cost = costs["cost"][b]
        return (cost["embed"] + costs["layers"] * cost["layer"] + cost["pool"]) // max(1, b // 2)   # batches are half full on average
    chunks, work, done = costs["chunks"], 0, 0
    c = chunks[0]
    while done < prompt_len:
        c = next((x for x in chunks if x >= prompt_len - done), chunks[-1])
        work += costs["cost"][c]["prep"] + costs["layers"] * costs["cost"][c]["layer"]
        done += c
    return work + costs["cost"][c]["head"]


def plan5(ai: dict, gpus: int, horizon_ns: int) -> list[dict]:
    bounds_all = json.loads(Path(ai["bounds_file"]).read_text())
    pairs = json.loads(Path(ai["length_pairs"]).read_text())["pairs"]
    out = []
    for index, spec in enumerate(ai["classes5"]):
        if spec["kind"] == "batch":
            continue
        costs = class_costs(spec, bounds_all)
        rng = np.random.default_rng(int(ai["seed"]) + 999 + 17 * index)
        rate = float(spec["rate_per_s"]) * gpus
        limit = int(float(spec.get("slo_ms", spec.get("ttft_ms", 0))) * 1e6)
        if spec["kind"] == "encoder":       # items may wait this long for a batch (ai_worker5)
            limit -= int(float(spec.get("batch_delay_ms", 0.3 * float(spec.get("slo_ms", 0)))) * 1e6)
        cap = int(spec.get("response_max", 64)) if spec["kind"] == "chat" else 0
        t = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t * 1e9 >= horizon_ns:
                break
            prompt, response = pairs[int(rng.integers(len(pairs)))]
            prompt = int(max(1, min(prompt, 1024)))
            out.append({"arrival_ns": int(t * 1e9), "cls": index, "prompt_len": prompt,
                        "response": int(min(response, cap)), "limit_ns": limit,
                        "tier": tier_of(float(spec.get("slo_ms", spec.get("ttft_ms", 0)))),
                        "work_ns": first_work_ns(spec, costs, prompt)})
    out.sort(key=lambda r: (r["arrival_ns"], r["cls"]))
    return out
