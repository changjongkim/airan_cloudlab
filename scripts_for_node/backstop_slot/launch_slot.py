#!/usr/bin/env python3
"""Launch one slot-scale run: per-cell radio workers and optional AI workers.

The run configuration names the cells, their GPUs and profiles, the NeuralRx
start policy and the AI policy.  Every worker maps one shared state file and
reads CLOCK_MONOTONIC, so uplink period k is released at the same instant in
all processes: TDD cells are synchronized, so their uplink slots coincide.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from slot_state import H_ABORT, H_EPOCH_NS, SlotState, now_ns, unlink_quietly
from summarize_slot import summarize

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cells = config["cells"]
    num_cells = len(cells)
    num_gpus = int(config.get("num_gpus", 4))
    periods = int(config["periods"])
    period_ns = int(round(float(config["period_ms"]) * 1e6))
    deadline_ns = int(round(float(config["deadline_ms"]) * 1e6))
    nrx_bound_ns = int(round(float(config.get("nrx_bound_ms", 0.0)) * 1e6))
    policy = config["nrx_policy"]               # off | parallel | rescue
    dataset = Path(config["dataset"])
    work = args.output.parent / (args.output.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    state_path = Path("/dev/shm") / f"backstop_slot_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.getpid()}"
    state = SlotState(state_path, num_cells, periods, num_gpus, create=True)

    python = sys.executable
    processes = []
    ready = 0
    common = [
        "--state", str(state_path), "--num-cells", str(num_cells),
        "--num-gpus", str(num_gpus), "--periods", str(periods),
        "--period-ns", str(period_ns),
    ]
    runtime = config.get("nrx_runtime", "lanes")
    weak_cells = [c for c in cells if c.get("nrx_gpu") is not None]
    group_runtime = config.get("conv_runtime") == "group"
    if group_runtime:
        # Several cells per conventional process: one cuPHY cell-group call per period.
        size = max(1, int(config.get("conv_group_size", 1)))
        for gpu in range(num_gpus):
            mine = [int(c["cell"]) for c in cells if int(c["gpu"]) == gpu]
            for start in range(0, len(mine), size):
                group = mine[start:start + size]
                for index in group:
                    unlink_quietly(work / f"cell{index}.ipc")
                command = [
                    python, str(HERE / "conv_group_worker.py"), "--gpu", str(gpu),
                    "--cells", ",".join(str(i) for i in group), "--config", str(args.config),
                    "--work", str(work), "--ready-slot", str(ready), *common,
                ]
                processes.append(("convgroup", group[0], subprocess.Popen(
                    command, stdout=open(work / f"convgroup{group[0]}.log", "w"), stderr=subprocess.STDOUT
                )))
                ready += 1
    for cell in ([] if group_runtime else cells):
        index = int(cell["cell"])
        ipc_file = work / f"cell{index}.ipc"
        unlink_quietly(ipc_file)
        wants_nrx = policy != "off" and cell.get("nrx_gpu") is not None
        command = [
            python, str(HERE / "conv_worker.py"), "--cell", str(index),
            "--gpu", str(cell["gpu"]), "--profile", cell["profile"],
            "--dataset", str(dataset), "--ring", str(config.get("ring", 128)),
            "--offset", str(cell.get("offset", 97 * index)),
            "--ready-slot", str(ready), "--output", str(work / f"conv{index}.json"),
            "--config", str(args.config),
            *common,
        ]
        if wants_nrx:
            command += ["--ipc-file", str(ipc_file)]
        processes.append(("conv", index, subprocess.Popen(
            command, stdout=open(work / f"conv{index}.log", "w"), stderr=subprocess.STDOUT
        )))
        ready += 1
        if wants_nrx and runtime == "per_cell":
            command = [
                python, str(HERE / "nrx_worker.py"), "--cell", str(index),
                "--gpu", str(cell["nrx_gpu"]), "--profile", cell["profile"],
                "--dataset", str(dataset), "--engine", config["engine"],
                "--ipc-file", str(ipc_file), "--policy", policy,
                "--deadline-ns", str(deadline_ns), "--nrx-bound-ns", str(nrx_bound_ns),
                "--ready-slot", str(ready), "--output", str(work / f"nrx{index}.json"),
                *common,
            ]
            processes.append(("nrx", index, subprocess.Popen(
                command, stdout=open(work / f"nrx{index}.log", "w"), stderr=subprocess.STDOUT
            )))
            ready += 1
    if runtime == "lanes" and policy != "off" and weak_cells:
        per_gpu = int(config.get("lanes_per_gpu", 1))
        for lane_id in range(num_gpus * per_gpu):
            gpu = lane_id // per_gpu
            command = [
                python, str(HERE / "nrx_lane.py"), "--gpu", str(gpu), "--lane", str(lane_id),
                "--config", str(args.config), "--work", str(work),
                "--ready-slot", str(ready), "--output", str(work / f"lane{lane_id}.json"),
                *common,
            ]
            processes.append(("lane", lane_id, subprocess.Popen(
                command, stdout=open(work / f"lane{lane_id}.log", "w"), stderr=subprocess.STDOUT
            )))
            ready += 1
    if runtime == "lanes":
        command = [
            python, str(HERE / config.get("controller", "controller.py")), "--config", str(args.config),
            "--ready-slot", str(ready), "--output", str(work / "controller.json"), *common,
        ]
        processes.append(("controller", 0, subprocess.Popen(
            command, stdout=open(work / "controller.log", "w"), stderr=subprocess.STDOUT
        )))
        ready += 1
    if config.get("ai_policy", "none") != "none":
        ai_env = dict(os.environ, PYTHONPATH=config["ai"]["pythonpath"],
                      HF_HOME=config["ai"]["hf_home"], HF_HUB_OFFLINE="1",
                      PYTHONNOUSERSITE="1")
        if config["ai_policy"] in ("static", "backstop_corun"):
            ai_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(config["ai"]["static_mps_pct"])
        elif config["ai"].get("gated_mps_pct"):
            # Granted policies may also cap the AI share: less interference per overlapped ms.
            ai_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(config["ai"]["gated_mps_pct"])
        if config["ai"].get("mps_client_priority") is not None:
            ai_env["CUDA_MPS_CLIENT_PRIORITY"] = str(config["ai"]["mps_client_priority"])
        for gpu in range(num_gpus if config.get("dynamic_share") else 0):
            # Dynamic-share baseline: a pre-loaded worker per share; the controller picks one.
            for share_id, pct in enumerate(config["dynamic_share"]["shares"], start=1):
                command = [
                    python, str(HERE / "ai_worker_dyn.py"), "--gpu", str(gpu), "--share-id", str(share_id),
                    "--config", str(args.config), "--ready-slot", str(ready),
                    "--output", str(work / f"ai{gpu}s{pct}.json"), *common,
                ]
                processes.append(("ai", gpu * 10 + share_id, subprocess.Popen(
                    command, env=dict(ai_env, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(pct)),
                    stdout=open(work / f"ai{gpu}s{pct}.log", "w"), stderr=subprocess.STDOUT
                )))
                ready += 1
        for gpu in range(0 if config.get("dynamic_share") else num_gpus):
            command = [
                python, str(HERE / config["ai"].get("worker", "ai_worker.py")), "--gpu", str(gpu),
                "--config", str(args.config), "--ready-slot", str(ready),
                "--output", str(work / f"ai{gpu}.json"), *common,
            ]
            processes.append(("ai", gpu, subprocess.Popen(
                command, env=ai_env, stdout=open(work / f"ai{gpu}.log", "w"),
                stderr=subprocess.STDOUT
            )))
            ready += 1

    try:
        deadline = time.monotonic() + float(config.get("ready_timeout_s", 900))
        while state.ready_count(ready) < ready:
            failed = [(kind, index, p.returncode) for kind, index, p in processes
                      if p.poll() is not None]
            if failed:
                raise RuntimeError(f"worker exited before readiness: {failed}")
            if time.monotonic() > deadline:
                raise TimeoutError("workers did not become ready")
            time.sleep(0.05)
        epoch = now_ns() + int(config.get("epoch_lead_ms", 200)) * 1_000_000
        state.header[H_EPOCH_NS] = epoch
        for kind, index, process in processes:
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"{kind}{index} exited with {code}")
    except BaseException:
        state.header[H_ABORT] = 1
        for _, _, process in processes:
            if process.poll() is None:
                process.kill()
        raise
    finally:
        try:
            import numpy as np
            np.save(work / "final_cells.npy", np.asarray(state.cells))
        except Exception:
            pass
        unlink_quietly(state_path)

    result = summarize(config, work, epoch)
    result.update({
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": config,
    })
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["headline"], indent=2), flush=True)


if __name__ == "__main__":
    main()
