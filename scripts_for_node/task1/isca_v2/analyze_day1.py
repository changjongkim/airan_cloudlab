#!/usr/bin/env python3
"""Build a compact, provenance-friendly summary of a DART Day-1 campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def latest_status(root):
    result = {}
    path = root / "STATUS.tsv"
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 4:
            result[fields[1]] = {
                "utc": fields[0], "status": fields[2], "importance": fields[3]
            }
    return result


def summarize_placement(root):
    groups = {}
    for path in (root / "04_fixed_placement").glob("p2p_overlap_*.json"):
        item = load(path)
        mode = item["mode"]
        group = groups.setdefault(mode, [])
        group.append(item)
    result = {}
    for mode, items in groups.items():
        result[mode] = {
            "trials": len(items),
            "l1_active_mean_ms_median": median([
                item["metrics"]["l1_active_ms"]["mean"] for item in items
            ]),
            "l1_active_p99_ms_median": median([
                item["metrics"]["l1_active_ms"]["p99"] for item in items
            ]),
            "e2e_mean_ms_median": median([
                item["metrics"]["e2e_ms"]["mean"] for item in items
            ]),
            "nrx_mean_ms_median": median([
                item["metrics"]["nrx_ms"]["mean"] for item in items
            ]),
            "transport_mean_us_median": median([
                item["metrics"]["transport_us"]["mean"] for item in items
            ]),
            "throughput_slots_s_median": median([
                item["completion_throughput_slots_s"] for item in items
            ]),
        }
    return result


def summarize_gdr(root):
    items = [load(path) for path in (root / "05_fixed_gdr").glob("l1prod_*.json")]
    return {
        "trials": len(items),
        "chain_mean_ms_median": median([item.get("mean_ms") for item in items]),
        "chain_p99_ms_median": median([item.get("p99_ms") for item in items]),
        "transport": items[0].get("transport") if items else None,
    }


def summarize_background(root):
    rows = []
    corrected = root / "10b_background_suite_corrected"
    base = corrected if (corrected / "COMPLETE").is_file() else root / "10_background_suite"
    if (base / "INVALID_DEVICE_MAPPING").exists():
        return rows
    for nrx_path in base.glob("*/*/nrx_timeline.json"):
        nrx = load(nrx_path)
        workload = nrx_path.parents[1].name
        policy = nrx_path.parent.name
        background_path = nrx_path.with_name("background_timeline.json")
        background = load(background_path) if background_path.is_file() else {}
        burst = [
            phase for phase in nrx.get("phase_results", [])
            if phase.get("kind") == "burst"
        ]
        burst = burst[0] if burst else {}
        units = background.get("units", background.get("iterations", []))
        wall_s = background.get("wall_s") or 0
        rows.append({
            "workload": workload,
            "policy": policy,
            "nrx_p99_ms": nrx.get("latency_ms", {}).get("p99"),
            "nrx_miss5": nrx.get("deadline_miss_ratio", {}).get("5ms"),
            "max_outstanding": nrx.get("max_outstanding"),
            "backlog_at_window_end": nrx.get("backlog_at_window_end"),
            "burst_p99_ms": burst.get("latency_ms", {}).get("p99"),
            "burst_miss5": burst.get("deadline_miss_ratio", {}).get("5ms"),
            "burst_assignments": burst.get("per_endpoint_requests"),
            "background_units": len(units),
            "background_units_s": len(units) / wall_s if wall_s else None,
        })
    return sorted(rows, key=lambda item: (item["workload"], item["policy"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    output_json = Path(args.output_json or root / "SUMMARY.json")
    output_md = Path(args.output_md or root / "SUMMARY.md")

    summary = {
        "schema": "dart-day1-summary-v0",
        "results_root": str(root),
        "status": latest_status(root),
        "placement": summarize_placement(root),
        "gdr": summarize_gdr(root),
        "background": summarize_background(root),
    }
    direct_path = root / "02_nrx_profile/direct_graph.json"
    if direct_path.is_file():
        direct = load(direct_path)
        summary["nrx_direct"] = direct.get("direct")
    wrapper_path = root / "02_nrx_profile/wrapper_breakdown.json"
    if wrapper_path.is_file():
        summary["nrx_wrapper_breakdown"] = load(wrapper_path).get("metrics")
    corrected_policy = root / "08b_hardware_policy_corrected/hardware_policy.json"
    policy_path = (
        corrected_policy if corrected_policy.is_file()
        else root / "08_hardware_policy/hardware_policy.json"
    )
    if policy_path.is_file():
        policy = load(policy_path)
        summary["hardware_policy"] = {
            "source": str(policy_path),
            "endpoints": policy.get("endpoints"),
            "runs": policy.get("runs"),
        }
    replica_path = root / "09_replica_sweep/replica_sweep.json"
    if replica_path.is_file():
        replica = load(replica_path)
        summary["replica_sweep"] = replica.get("configurations")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(output_json)

    lines = [
        "# DART-Rx Day-1 summary",
        "",
        "이 파일은 완료된 artifact만 집계한다. P2P와 GDR 수치는 서로 다른 "
        "pipeline timing boundary일 수 있으므로 transport-only 수치로 해석하지 않는다.",
        "",
        "## Job status",
        "",
        "| Job | Latest status | Importance |",
        "|---|---:|---:|",
    ]
    for job, item in sorted(summary["status"].items()):
        lines.append(f"| {job} | {item['status']} | {item['importance']} |")
    lines += ["", "## Fixed-topology placement", ""]
    if summary["placement"]:
        lines += [
            "| Mode | trials | L1 active mean (ms) | L1 p99 (ms) | "
            "E2E mean (ms) | transport mean (µs) | slots/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, item in sorted(summary["placement"].items()):
            def show(key):
                value = item.get(key)
                return "-" if value is None else f"{value:.3f}"
            lines.append(
                f"| {mode} | {item['trials']} | "
                f"{show('l1_active_mean_ms_median')} | "
                f"{show('l1_active_p99_ms_median')} | "
                f"{show('e2e_mean_ms_median')} | "
                f"{show('transport_mean_us_median')} | "
                f"{show('throughput_slots_s_median')} |"
            )
    lines += ["", "## Background reclaim", ""]
    if summary["background"]:
        lines += [
            "| Workload | Policy | burst p99 (ms) | burst miss >5ms | "
            "background units/s |",
            "|---|---|---:|---:|---:|",
        ]
        for item in summary["background"]:
            def show(value, digits=4):
                return "-" if value is None else f"{value:.{digits}f}"
            lines.append(
                f"| {item['workload']} | {item['policy']} | "
                f"{show(item['burst_p99_ms'], 3)} | "
                f"{show(item['burst_miss5'], 6)} | "
                f"{show(item['background_units_s'], 3)} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- `background_gated.py`의 ResNet/BERT/Whisper는 실제 모델 구조와 "
        "framework kernel을 사용하지만 synthetic input/random weights이며 정확도 실험이 아니다.",
        "- control replay는 measured component profile을 사용한 deterministic model이며 "
        "hardware performance 결과가 아니다.",
        "- 최종 claim은 hardware policy, burst deadline miss, background utility, "
        "그리고 Nsight trace가 함께 지지할 때만 채택한다.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DART-ANALYZE] OK json={output_json} md={output_md}")


if __name__ == "__main__":
    main()
