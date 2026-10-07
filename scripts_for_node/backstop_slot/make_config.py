#!/usr/bin/env python3
"""Write one slot-scale run configuration.

Cells are spread round-robin over the GPUs.  Every other cell on a GPU hosts
a weak rank-1 UE (the weak fraction sets how many), and its NeuralRx runs on
the next GPU so the two receivers of one TB never share a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab"
# Qwen2.5-0.5B unit bounds; filled from bench_units (isolated p99.9 plus margin).
# Measured in job 59151256 (results/backstop_slot/raw/qwen05_units_j59151256.json).
UNIT_BOUNDS_05B = {"128": {"prep": 0.12, "layer": 0.30, "head": 0.30},
                   "512": {"prep": 0.12, "layer": 0.53, "head": 0.42},
                   "1024": {"prep": 0.14, "layer": 0.78, "head": 0.90}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--weak-fraction", type=float, default=0.5)
    parser.add_argument("--nrx-policy", choices=("off", "parallel", "parallel_admit", "rescue_naive", "rescue", "rescue_value"),
                        required=True)
    parser.add_argument("--ai-policy", default="none",
                        choices=("none", "backstop", "backstop_corun", "backstop_units", "idle", "static"))
    parser.add_argument("--periods", type=int, default=2000)
    parser.add_argument("--period-ms", type=float, default=2.5)
    parser.add_argument("--deadline-ms", type=float, default=4.0)
    parser.add_argument("--nrx-bound-ms", type=float, default=2.5)
    parser.add_argument("--nrx-bound-corun-ms", type=float, default=2.8)
    parser.add_argument("--rescue-deadline-ms", type=float, default=None,
                        help="deadline for a NeuralRx rescue (default: the L1 deadline)")
    parser.add_argument("--lanes-per-gpu", type=int, default=1)
    parser.add_argument("--ai-max-piece-ms", type=float, default=1.0)
    parser.add_argument("--ai-force-chunk", type=int, default=0)
    parser.add_argument("--ai-dispatch", choices=("per_gpu", "global"), default="per_gpu")
    parser.add_argument("--conv-by-active", default=None,
                        help="unit sizes allowed next to the conventional receiver by the number of active cells of the "
                             "GPU in the slot, e.g. 4:128,512/5:128 (above the largest number: none)")
    parser.add_argument("--unit-gating", default=None,
                        help="e.g. 128:2.8,512:3.2,1024:3.6/conv=128,512 (NeuralRx co-run bound per AI chunk)")
    parser.add_argument("--conv-budget", default=None,
                        help="conventional delay budget, e.g. alone=3.0/margin=0.2/128:0.5,512:1.0 "
                             "(alone: conventional completion without overlapping AI; "
                             "chunk:alpha = conventional delay per ms of overlapping AI)")
    parser.add_argument("--ai-chunk-choice", choices=("budget", "sustained"), default="budget")
    parser.add_argument("--gated-mps-pct", type=int, default=0,
                        help="MPS share cap of the AI worker under a granted policy (0: none)")
    parser.add_argument("--ai-adaptive-bound", type=int, default=0,
                        help="1: the AI worker scales its unit bounds by measured co-run time")
    parser.add_argument("--controller", default=None, help="controller script (e.g. controller3.py)")
    parser.add_argument("--conv-runtime", choices=("per_cell", "group"), default="per_cell")
    parser.add_argument("--group-size", type=int, default=1, help="cells per conventional process (group runtime)")
    parser.add_argument("--activity-prob", type=float, default=1.0,
                        help="probability that a cell carries a TB in an uplink period")
    parser.add_argument("--activity-mode", choices=("bernoulli", "bursty", "phased", "steps"), default="bernoulli")
    parser.add_argument("--activity-phase", type=int, default=800,
                        help="phase length in periods (phased: full load and --activity-prob alternate)")
    parser.add_argument("--activity-high", type=float, default=1.0,
                        help="activity probability of the busy phases (phased)")
    parser.add_argument("--activity-burst", type=float, default=8.0, help="mean busy run in periods (bursty)")
    parser.add_argument("--activity-levels", default="0.25,0.5,0.75,1.0",
                        help="activity probabilities the load jumps between (steps)")
    parser.add_argument("--ai-arrival-cv", type=float, default=1.0,
                        help="coefficient of variation of AI inter-arrival times (1: Poisson)")
    parser.add_argument("--dynamic-levels", default=None,
                        help="load fractions that separate the dynamic shares, high to low, e.g. 0.875,0.625")
    parser.add_argument("--nrx-max-cb-fail", type=int, default=1)
    parser.add_argument("--nrx-ls-input", choices=("nvlabs", "example"), default="nvlabs",
                        help="layout of the LS estimate given to NeuralRx (example = runs before 2026-10-01)")
    parser.add_argument("--nrx-primary-cells", type=int, default=0,
                        help="weak cells whose every TB is decoded by NeuralRx from arrival")
    parser.add_argument("--nrx-bound-busy", default=None,
                        help="NeuralRx bound alone,with a second lane on the GPU, e.g. 2.8,4.5")
    parser.add_argument("--conv-alone-by-load", default=None,
                        help="conventional completion without AI by active cells on the GPU, e.g. 4:1.6,8:2.4")
    parser.add_argument("--ai-classes", default=None,
                        help="job classes for ai_worker3, e.g. chat:1.5B:8:200,small:0.5B:8:100,batch:1.5B")
    parser.add_argument("--ai-class-order", choices=("urgency", "fifo"), default="urgency")
    parser.add_argument("--ai-chunks", default=None,
                        help="e.g. 128,512: use the multi-chunk AI worker with these chunk sizes")
    parser.add_argument("--weak-profile", default="weak_rank1",
                        help="profile of the weak cells (nv_mu2: two-UE MU-MIMO, NVlabs neural receiver)")
    parser.add_argument("--dataset", default=f"{ROOT}/run_state/backstop_slot/dataset_v1")
    parser.add_argument("--conv-ldpc-iterations", type=int, default=0,
                        help="LDPC iteration limit of the conventional receiver (0: cuPHY's table, 10 here)")
    parser.add_argument("--nrx-ldpc-iterations", type=int, default=10,
                        help="LDPC iterations after the neural receiver")
    parser.add_argument("--dynamic-shares", default=None,
                        help="dynamic-share baseline: low,high MPS share (%%), e.g. 10,30 (ai-policy static)")
    parser.add_argument("--dynamic-lag", type=int, default=0,
                        help="periods by which the dynamic-share baseline sees the radio load late")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--engine", default="/softwall_runtime/engines/neural_rx_fp16_full.trt")
    parser.add_argument("--ai-rate", type=float, default=4.0, help="requests/s per GPU")
    parser.add_argument("--ai-chunk", type=int, default=128)
    parser.add_argument("--ai-slo-ms", type=float, default=200.0)
    parser.add_argument("--ai-admission", type=int, default=0,
                        help="1: admit an AI request only if it is predicted to meet its SLO")
    parser.add_argument("--nrx-run-ms", default="6.3,128:7.4,512:7.9,1024:7.9",
                        help="median NeuralRx run alone, then next to AI of each unit size (reuse-time rule)")
    parser.add_argument("--nrx-reuse-alone-ms", type=float, default=6.7,
                        help="NeuralRx run length used to predict when a run ends (reuse-time rule)")
    parser.add_argument("--ai-stop-check", type=int, default=0,
                        help="1: the AI worker reads the controller's stop word between unit groups")
    parser.add_argument("--yyr", default=None,
                        help="YinYangRAN-style baseline: estimator.json,kept target,l1 target,quantile,reconfig_ms,decision_periods "
                             "(with --dynamic-shares)")
    parser.add_argument("--ring", type=int, default=128,
                        help="slots of the pool that every cell replays in a loop (128: the pattern of failed "
                             "TBs repeats every 0.32 s)")
    parser.add_argument("--ai-classes5", type=Path, default=None,
                        help="JSON with classes5, bounds_file, length_pairs: several models and kinds of AI work (ai_worker5)")
    parser.add_argument("--ai-admission-fraction", type=float, default=1.0,
                        help="admit a request if it is predicted to finish within this share of its time limit")
    parser.add_argument("--static-mps-pct", type=int, default=30)
    parser.add_argument("--idle-budget-ms", type=float, default=1.0)
    parser.add_argument("--nrx-runtime", choices=("lanes", "per_cell"), default="lanes")
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--ring-offset", type=int, default=0,
                        help="shifts every cell's window into the slot pool")
    parser.add_argument("--ai-mps-priority", type=int, default=None,
                        help="CUDA_MPS_CLIENT_PRIORITY for AI workers (1 = below normal)")
    parser.add_argument("--nrx-flags", default=None,
                        help="override NeuralRx mechanisms, e.g. value=0,admit=0,start=fail_only")
    parser.add_argument("--la", default=None,
                        help="closed-loop link adaptation of the two-user cells: 'MCS,MCS,...:TARGET[:DOWN[:DELAY[:START]]]' "
                             "(MCS levels with a ring each in the dataset, target share of TBs that need a retransmission, "
                             "pointer step down per such TB in levels, feedback delay in periods, start level)")
    parser.add_argument("--la-states", default=None,
                        help="channel states of the two-user cells with --la: 'DATASET,DATASET,...:PHASE' (one dataset "
                             "directory per state with the rings of every MCS level; a cell stays PHASE periods in a "
                             "state and then moves to another one at random; PHASE 0: the two-user cells take the "
                             "datasets in turn and keep them)")
    parser.add_argument("--ai-extra", type=Path, help="JSON merged into the config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    la = None
    if args.la:
        fields = args.la.split(":")
        levels = [int(m) for m in fields[0].split(",")]
        la = {"levels": [f"nv_mu2_m{m}" for m in levels], "mcs": levels, "target": float(fields[1]),
              "down": float(fields[2]) if len(fields) > 2 else 0.25, "delay": int(fields[3]) if len(fields) > 3 else 5,
              "start": float(fields[4]) if len(fields) > 4 else (len(levels) - 1) / 2.0}
        args.weak_profile = la["levels"][-1]         # the cell's profile is its highest level
        if args.la_states:
            dirs, phase = args.la_states.rsplit(":", 1)
            la.update(datasets=[str(Path(d).resolve()) for d in dirs.split(",")], phase=int(phase), state_seed=args.seed)
            for d in la["datasets"]:
                for name in la["levels"]:
                    if not (Path(d) / f"{name}_meta.json").is_file():
                        raise SystemExit(f"--la-states: no ring of {name} in {d}")

    weak_total = round(args.cells * args.weak_fraction)
    cells = []
    per_gpu: dict[int, int] = {}
    order = sorted(range(args.cells), key=lambda c: (c // args.gpus, c % args.gpus))
    weak_set = set()
    # Alternate weak cells across GPUs so every GPU gets an equal share.
    for cell in order:
        if len(weak_set) < weak_total and (cell // args.gpus) % 2 == 0:
            weak_set.add(cell)
    for cell in order:
        if len(weak_set) < weak_total and cell not in weak_set:
            weak_set.add(cell)
    for cell in range(args.cells):
        gpu = cell % args.gpus
        per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
        weak = cell in weak_set
        cells.append({
            "cell": cell,
            "gpu": gpu,
            "profile": args.weak_profile if weak else "strong_rank2",
            "num_ue": 2 if weak and args.weak_profile.startswith("nv_mu2") else 1,
            "nrx_gpu": (gpu + 1) % args.gpus if weak else None,
            "offset": 97 * cell + args.ring_offset,
        })
    config = {
        "schema": "backstop-slot-config-v1",
        "cells": cells,
        "num_gpus": args.gpus,
        "periods": args.periods,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "report_extra_deadline_ms": 0.5,
        "nrx_policy": args.nrx_policy,
        "nrx_bound_ms": args.nrx_bound_ms,
        "ai_policy": args.ai_policy,
        "engine": args.engine,
        "dataset": args.dataset,
        "ring": args.ring,
        "skip_periods": 20,
        "tdd_pattern": "DDDSU, 30 kHz SCS, one UL slot per 2.5 ms",
        "deadline_basis": "testMAC UL indication at T0+4.5 ms; data is ready at slot end T0+0.5 ms",
        "nrx_runtime": args.nrx_runtime,
        "rescue_deadline_ms": args.rescue_deadline_ms if args.rescue_deadline_ms else args.deadline_ms,
        "lanes_per_gpu": args.lanes_per_gpu,
        "ai_guard_us": 50,
        "ai_idle_budget_ms": args.idle_budget_ms,
        "ai_min_budget_ms": 0.36,
        "ai_max_piece_ms": args.ai_max_piece_ms,
        "nrx_bound_corun_ms": args.nrx_bound_corun_ms,
        "nrx_max_cb_fail": 1,
        "ai": {
            "model": "Qwen/Qwen2.5-1.5B",
            "chunk": args.ai_chunk,
            "rate_per_s": args.ai_rate,
            "seed": args.seed,
            "unit_bound_ms": {"prep": 0.30, "layer": 0.35, "head": 0.36},
            "trace": f"{ROOT}/results/softwall_multigpu/c176_traces/c176_trace_gamma_cv1.json",
            "max_prompt": 1024,
            "slo_ms": args.ai_slo_ms,
            "slo_admission": bool(args.ai_admission),
            "prior_units_per_ms": 1.5,
            "static_mps_pct": args.static_mps_pct,
            "static_units_per_piece": 8,
            "mps_client_priority": args.ai_mps_priority,
            "overrun_tol_us": 100,
            "pythonpath": "/softwall_runtime/python:/backstop_slot",
            "hf_home": "/softwall_runtime/cache/huggingface",
        },
    }
    if args.ai_chunks:
        chunks = [int(c) for c in args.ai_chunks.split(",")]
        config["ai"]["worker"] = "ai_worker2.py"
        config["ai"]["chunks"] = chunks
        # Bounds from the measured unit GPU time (p99.9 plus margin), A100 80GB.
        measured = {128: (0.30, 0.35, 0.36), 256: (0.15, 0.46, 0.36),
                    512: (0.15, 0.72, 0.36), 1024: (0.15, 1.10, 0.38)}
        by_chunk = {str(c): dict(zip(("prep", "layer", "head"), measured[c])) for c in chunks}
        config["ai"]["unit_bound_ms_by_chunk"] = by_chunk
        config["ai"]["force_chunk"] = args.ai_force_chunk
        config["ai"]["dispatch"] = args.ai_dispatch
    if args.unit_gating:
        bounds_part, _, conv_part = args.unit_gating.replace(";", "/").partition("/")
        config["ai_unit_gating"] = {
            "nrx_bound_corun_ms": {k: float(v) for k, v in (x.split(":") for x in bounds_part.split(","))},
            "conv_safe_chunks": [int(c) for c in conv_part.replace("conv=", "").replace("+yield", "").replace("+queue", "").replace("+nrxbudget", "").split("+")[0].split(",") if c],
            "yield_to_waiting_nrx": "+yield" in conv_part,
            "queue_check": "+queue" in conv_part,
            "nrx_budget": "+nrxbudget" in conv_part,
        }
        for part in conv_part.split("+"):
            if part.startswith("reserve"):
                config["ai_unit_gating"]["free_lane_reserve"] = int(part[len("reserve"):])
        if "+reuse" in conv_part:
            alone, _, per_chunk = args.nrx_run_ms.partition(",")
            config["ai_unit_gating"].update({
                "reuse_rule": True, "run_alone_ms": float(alone), "reuse_alone_ms": args.nrx_reuse_alone_ms,
                "run_ai_ms": {k: float(v) for k, v in (x.split(":") for x in per_chunk.split(","))}})
    if args.conv_by_active:
        config.setdefault("ai_unit_gating", {})["conv_safe_by_active"] = {
            part.split(":")[0]: [int(c) for c in part.split(":")[1].split(",") if c]
            for part in args.conv_by_active.split("/")}
    if args.conv_budget:
        alone, margin, alphas = 0.0, 0.2, {}
        for part in args.conv_budget.split("/"):
            if part.startswith("alone="):
                alone = float(part[6:])
            elif part.startswith("margin="):
                margin = float(part[7:])
            elif part:
                alphas = {k: float(v) for k, v in (x.split(":") for x in part.split(","))}
        config.setdefault("ai_unit_gating", {})["conv_budget"] = {
            "conv_alone_ms": alone, "margin_ms": margin, "alpha": alphas}
    if args.ai_adaptive_bound:
        config["ai"]["adaptive_unit_bound"] = True
    config["ai"]["chunk_choice"] = args.ai_chunk_choice
    if args.gated_mps_pct:
        config["ai"]["gated_mps_pct"] = args.gated_mps_pct
    config["nrx_max_cb_fail"] = args.nrx_max_cb_fail
    config["nrx_ls_input"] = args.nrx_ls_input
    config["conv_ldpc_iterations"] = args.conv_ldpc_iterations
    config["nrx_ldpc_iterations"] = args.nrx_ldpc_iterations
    if args.nrx_primary_cells:
        chosen = [c for c in config["cells"] if c["nrx_gpu"] is not None][:args.nrx_primary_cells]
        for cell in chosen:
            cell["nrx_mode"] = "primary"
    if args.controller:
        config["controller"] = args.controller
    if args.conv_runtime == "group":
        config["conv_runtime"] = "group"
        config["conv_group_size"] = args.group_size
    if args.activity_prob < 1.0 or args.activity_mode == "steps":
        config["activity"] = {"prob": args.activity_prob, "mode": args.activity_mode,
                              "burst_periods": args.activity_burst, "seed": args.seed,
                              "phase_periods": args.activity_phase, "phase_high": args.activity_high,
                              "levels": [float(v) for v in args.activity_levels.split(",")]}
    if args.ai_classes5:
        config["ai"].update(json.loads(args.ai_classes5.read_text()))
        config["ai"]["worker"] = "ai_worker5.py"      # --ai-dispatch global: one arrival list for the server
    if args.ai_stop_check:
        config["ai"]["stop_check"] = True
    if args.ai_admission_fraction != 1.0:
        config["ai"]["admission_fraction"] = args.ai_admission_fraction
    if args.ai_arrival_cv != 1.0:
        config["ai"]["arrival_cv"] = args.ai_arrival_cv
    if args.nrx_bound_busy:
        config["nrx_bound_by_busy_ms"] = [float(v) for v in args.nrx_bound_busy.split(",")]
    if args.conv_alone_by_load:
        config.setdefault("ai_unit_gating", {}).setdefault("conv_budget", {"margin_ms": 0.2, "alpha": {}})[
            "conv_alone_ms"] = {k: float(v) for k, v in (x.split(":") for x in args.conv_alone_by_load.split(","))}
    if args.ai_classes:
        models = {"1.5B": "Qwen/Qwen2.5-1.5B", "0.5B": "Qwen/Qwen2.5-0.5B"}
        classes = []
        for item in args.ai_classes.split(","):
            parts = item.split(":")
            spec = {"name": parts[0], "model": models[parts[1]]}
            if len(parts) > 2:
                spec.update({"kind": "interactive", "rate_per_s": float(parts[2]), "slo_ms": float(parts[3])})
            else:
                spec.update({"kind": "batch", "prompt_len": 1024})
            classes.append(spec)
        config["ai"].update({
            "worker": "ai_worker3.py", "classes": classes, "class_order": args.ai_class_order,
            "chunks": [int(c) for c in (args.ai_chunks or "128,512,1024").split(",")],
            # Isolated unit GPU time (p99.9 plus margin) per model, A100 80GB.
            "unit_bound_ms_by_model": {
                "Qwen/Qwen2.5-1.5B": {"128": {"prep": 0.30, "layer": 0.35, "head": 0.36},
                                      "512": {"prep": 0.15, "layer": 0.72, "head": 0.36},
                                      "1024": {"prep": 0.15, "layer": 1.10, "head": 0.38}},
                "Qwen/Qwen2.5-0.5B": UNIT_BOUNDS_05B,
            },
        })
    if args.dynamic_shares:
        config["dynamic_share"] = {"shares": [int(s) for s in args.dynamic_shares.split(",")],
                                   "lag_periods": args.dynamic_lag, "window_periods": 40}
        if args.dynamic_levels:
            config["dynamic_share"]["levels"] = [float(v) for v in args.dynamic_levels.split(",")]
        if args.yyr:
            estimator, target, l1_target, quantile, reconfig, decision = args.yyr.split(",")
            config["dynamic_share"].update({"mode": "yyr", "estimator": estimator, "target": float(target),
                                            "l1_target": float(l1_target), "quantile": float(quantile),
                                            "reconfig_ms": float(reconfig), "decision_periods": int(decision)})
    if args.nrx_flags:
        flags = {}
        for item in args.nrx_flags.split(","):
            key, value = item.split("=")
            flags[key] = bool(int(value)) if key in ("skip", "admit", "value") else value
        config["nrx_flags"] = flags
    if la:
        if la.get("datasets") and la["phase"] <= 0:      # fixed states: the two-user cells take the datasets in turn
            two_user = [c["cell"] for c in cells if c["nrx_gpu"] is not None]
            la["assign"] = {str(cell): i % len(la["datasets"]) for i, cell in enumerate(two_user)}
        config["la"] = la
    if args.ai_extra:
        config.update(json.loads(args.ai_extra.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
