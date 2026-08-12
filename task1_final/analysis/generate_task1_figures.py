#!/usr/bin/env python3
"""Generate the Task 1 latency/throughput figures from chain/SUMMARY.txt.

The parser deliberately treats (config, pct) as the record key so the config2
MPS sweep is not collapsed.  Future config4_rdma/config7_rdma rows are picked
up automatically and labelled as CPU-buffer RDMA (not GPUDirect RDMA).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = SCRIPT_DIR.parent / "chain" / "SUMMARY.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "figures"


@dataclass(frozen=True)
class Result:
    config: str
    pct: float | None
    mean_ms: float
    p95_ms: float
    p99_ms: float
    qwen_itps: float
    notes: str

    @property
    def key(self) -> tuple[str, float | None]:
        return self.config, self.pct


def as_float(value: str | None) -> float:
    """Parse an optional numeric CSV field, returning NaN when unavailable."""
    if value is None:
        return math.nan
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"-", "na", "n/a", "nan", "none"}:
        return math.nan
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


def read_summary(path: Path) -> list[Result]:
    """Read SUMMARY.txt and preserve repeated configs at different MPS caps."""
    with path.open(newline="", encoding="utf-8") as handle:
        meaningful = (
            line for line in handle if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(meaningful)
        required = {"config", "pct", "mean_ms", "p95_ms", "p99_ms", "qwen_itps"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"SUMMARY header is missing fields: {', '.join(missing)}")

        by_key: dict[tuple[str, float | None], Result] = {}
        for line_number, row in enumerate(reader, start=2):
            config = (row.get("config") or "").strip()
            if not config:
                continue
            pct_value = as_float(row.get("pct"))
            pct = None if math.isnan(pct_value) else pct_value
            result = Result(
                config=config,
                pct=pct,
                mean_ms=as_float(row.get("mean_ms")),
                p95_ms=as_float(row.get("p95_ms")),
                p99_ms=as_float(row.get("p99_ms")),
                qwen_itps=as_float(row.get("qwen_itps")),
                notes=(row.get("notes") or "").strip(),
            )
            if math.isnan(result.mean_ms):
                print(f"warning: skipping line {line_number}: missing mean_ms")
                continue
            if result.key in by_key:
                print(f"warning: duplicate {result.key}; keeping the last row")
            by_key[result.key] = result
    if not by_key:
        raise ValueError(f"No usable result rows found in {path}")
    return list(by_key.values())


def config_number(config: str) -> int:
    digits = "".join(character for character in config.lower().split("_")[0] if character.isdigit())
    return int(digits) if digits else 999


def category(result: Result) -> str:
    name = result.config.lower().replace("-", "_")
    if "gdr" in name:
        return "Cross-partition · GPUDirect RDMA staging"
    if "rdma" in name:
        return "Cross-partition · CPU-buffer RDMA"
    if name in {"config4", "config7"} or "shm" in name:
        return "Cross-partition · CPU shared memory"
    if name == "config2" or result.pct is not None or "mps" in name:
        return "Full-GPU MPS"
    if name in {"config1", "config3"} or "mig" in name:
        return "MIG same-partition"
    return "Standalone baseline"


def short_label(result: Result) -> str:
    name = result.config.lower().replace("-", "_")
    if name == "config5":
        return "cuPHY only\n(cfg5)"
    if name == "config6":
        return "cuPHY+NRx\nstandalone (cfg6)"
    if name == "config1":
        return "MIG same\n(cfg1)"
    if name == "config3":
        return "MIG same\n(cfg3)"
    if name == "config4":
        return "Cross SHM\n(cfg4)"
    if name == "config7":
        return "Cross SHM\n(cfg7)"
    if "gdr" in name:
        base = name.replace("config", "cfg").replace("_gdr", "")
        return f"Cross GPUDirect\nRDMA staging ({base})"
    if "rdma" in name:
        base = name.replace("config", "cfg").replace("_rdma", "")
        return f"Cross CPU-buffer\nRDMA ({base})"
    if result.pct is not None:
        pct = f"{result.pct:g}%"
        return f"MPS {pct}\n({result.config})"
    return result.config


def point_label(result: Result) -> str:
    name = result.config.lower().replace("-", "_")
    fixed_labels = {
        "config1": "cfg1 MIG",
        "config3": "cfg3 MIG",
        "config4": "cfg4 SHM",
        "config7": "cfg7 SHM",
        "config4_rdma": "cfg4 CPU-RDMA",
        "config7_rdma": "cfg7 CPU-RDMA",
        "config4_gdr": "cfg4 GDR staging",
        "config7_gdr": "cfg7 GDR staging",
    }
    if name in fixed_labels:
        return fixed_labels[name]
    if result.pct is not None:
        return f"MPS {result.pct:g}%"
    if "rdma" in name:
        return name.replace("config", "cfg").replace("_rdma", " CPU-RDMA")
    return name.replace("config", "cfg")


def annotation_spec(result: Result) -> tuple[tuple[int, int], str, bool]:
    """Return a deterministic label offset, alignment, and leader-line flag.

    The eight MIG/cross-partition points have almost identical throughput and
    tightly packed latency.  Fan their labels to both sides so repeated runs
    produce readable figures without relying on optional text-layout packages.
    """
    name = result.config.lower().replace("-", "_")
    clustered = {
        "config1": ((-14, -28), "right", True),
        "config3": ((14, -28), "left", True),
        "config4_gdr": ((-14, -10), "right", True),
        "config7_gdr": ((14, -10), "left", True),
        "config4_rdma": ((-14, 9), "right", True),
        "config7": ((14, 8), "left", True),
        "config4": ((-14, 29), "right", True),
        "config7_rdma": ((14, 29), "left", True),
    }
    return clustered.get(name, ((7, 9), "left", False))


def annotate_point(axis: plt.Axes, result: Result) -> None:
    offset, horizontal_alignment, leader = annotation_spec(result)
    axis.annotate(
        point_label(result),
        (result.qwen_itps, result.mean_ms),
        xytext=offset,
        textcoords="offset points",
        ha=horizontal_alignment,
        va="center",
        fontsize=8,
        bbox=(
            {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.86}
            if leader
            else None
        ),
        arrowprops=(
            {"arrowstyle": "-", "color": "#777777", "lw": 0.65, "shrinkA": 1, "shrinkB": 3}
            if leader
            else None
        ),
        zorder=6,
    )


def ordered(results: Iterable[Result]) -> list[Result]:
    group_order = {
        "Standalone baseline": 0,
        "MIG same-partition": 1,
        "Cross-partition · CPU shared memory": 2,
        "Cross-partition · CPU-buffer RDMA": 3,
        "Cross-partition · GPUDirect RDMA staging": 4,
        "Full-GPU MPS": 5,
    }

    def key(result: Result) -> tuple[float, int, float]:
        pct = result.pct if result.pct is not None else -1.0
        return group_order[category(result)], config_number(result.config), pct

    return sorted(results, key=key)


CATEGORY_COLORS = {
    "Standalone baseline": "#7f8c8d",
    "MIG same-partition": "#2a9d8f",
    "Cross-partition · CPU shared memory": "#e9c46a",
    "Cross-partition · CPU-buffer RDMA": "#277da1",
    "Cross-partition · GPUDirect RDMA staging": "#7b2cbf",
    "Full-GPU MPS": "#e76f51",
}


def style_axes(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(path)


def plot_latency_bars(results: list[Result], output: Path) -> None:
    rows = ordered(results)
    figure, axis = plt.subplots(figsize=(max(12.0, len(rows) * 1.25), 6.8))
    x_positions = list(range(len(rows)))
    width = 0.25
    metrics = [
        ("Mean", [row.mean_ms for row in rows], "#264653"),
        ("p95", [row.p95_ms for row in rows], "#4f86a6"),
        ("p99", [row.p99_ms for row in rows], "#90c2d7"),
    ]
    for index, (metric, values, color) in enumerate(metrics):
        offsets = [x + (index - 1) * width for x in x_positions]
        axis.bar(offsets, values, width=width, label=metric, color=color, edgecolor="white")

    axis.set_title("L1 Latency by Isolation and IPC Configuration")
    axis.set_ylabel("Latency (ms; lower is better)")
    axis.set_xticks(x_positions, [short_label(row) for row in rows], rotation=28, ha="right")
    axis.legend(ncol=3, frameon=False, loc="upper left")
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, output)


def plot_throughput_latency(results: list[Result], output: Path) -> None:
    rows = [row for row in ordered(results) if not math.isnan(row.qwen_itps)]
    figure, axis = plt.subplots(figsize=(10.8, 7.0))

    for group, color in CATEGORY_COLORS.items():
        points = [row for row in rows if category(row) == group]
        if not points:
            continue
        axis.scatter(
            [row.qwen_itps for row in points],
            [row.mean_ms for row in points],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=group,
            zorder=3,
        )

    mps = sorted(
        (row for row in rows if category(row) == "Full-GPU MPS"),
        key=lambda row: row.pct if row.pct is not None else math.inf,
    )
    if len(mps) > 1:
        axis.plot(
            [row.qwen_itps for row in mps],
            [row.mean_ms for row in mps],
            color=CATEGORY_COLORS["Full-GPU MPS"],
            linewidth=1.8,
            alpha=0.75,
            zorder=2,
        )

    for row in rows:
        annotate_point(axis, row)

    axis.set_title("AI Throughput vs. L1 Latency")
    axis.set_xlabel("Qwen throughput (iterations/s; higher is better)")
    axis.set_ylabel("L1 mean latency (ms; lower is better)")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, output)


def pareto_front(rows: list[Result]) -> list[Result]:
    """Return non-dominated points for max throughput and min latency."""
    front: list[Result] = []
    for candidate in rows:
        dominated = any(
            other.qwen_itps >= candidate.qwen_itps
            and other.mean_ms <= candidate.mean_ms
            and (
                other.qwen_itps > candidate.qwen_itps
                or other.mean_ms < candidate.mean_ms
            )
            for other in rows
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: row.qwen_itps)


def plot_pareto(results: list[Result], output: Path) -> None:
    rows = [row for row in ordered(results) if not math.isnan(row.qwen_itps)]
    frontier = pareto_front(rows)
    frontier_keys = {row.key for row in frontier}
    markers = {
        "Standalone baseline": "o",
        "MIG same-partition": "o",
        "Cross-partition · CPU shared memory": "s",
        "Cross-partition · CPU-buffer RDMA": "D",
        "Cross-partition · GPUDirect RDMA staging": "P",
        "Full-GPU MPS": "^",
    }

    figure, axis = plt.subplots(figsize=(10.8, 7.0))
    for row in rows:
        group = category(row)
        is_frontier = row.key in frontier_keys
        axis.scatter(
            row.qwen_itps,
            row.mean_ms,
            marker=markers[group],
            s=115 if is_frontier else 78,
            facecolor=CATEGORY_COLORS[group] if is_frontier else "none",
            edgecolor=CATEGORY_COLORS[group],
            linewidth=1.8 if is_frontier else 1.2,
            alpha=1.0 if is_frontier else 0.65,
            zorder=4 if is_frontier else 3,
        )
        annotate_point(axis, row)

    if len(frontier) > 1:
        axis.plot(
            [row.qwen_itps for row in frontier],
            [row.mean_ms for row in frontier],
            linestyle="--",
            linewidth=1.5,
            color="#222222",
            alpha=0.75,
            label="Empirical Pareto frontier",
            zorder=2,
        )

    standalone = next(
        (row for row in results if row.config.lower() == "config6"), None
    )
    if standalone is not None:
        axis.axhline(
            standalone.mean_ms,
            color=CATEGORY_COLORS["Standalone baseline"],
            linestyle=":",
            linewidth=1.4,
            label=f"Standalone cuPHY+NRx: {standalone.mean_ms:.1f} ms",
        )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker=markers[group],
            linestyle="none",
            markerfacecolor=CATEGORY_COLORS[group],
            markeredgecolor=CATEGORY_COLORS[group],
            markersize=7,
            label=group,
        )
        for group in CATEGORY_COLORS
        if any(category(row) == group for row in rows)
    ]
    legend_items.append(
        Line2D([0], [0], color="#222222", linestyle="--", label="Empirical Pareto frontier")
    )
    if standalone is not None:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color=CATEGORY_COLORS["Standalone baseline"],
                linestyle=":",
                label=f"Standalone cuPHY+NRx ({standalone.mean_ms:.1f} ms)",
            )
        )
    axis.legend(handles=legend_items, frameon=False, fontsize=8, loc="upper left")
    axis.set_title("Isolation and Transport Pareto Overview")
    axis.set_xlabel("Qwen throughput (iterations/s; higher is better)")
    axis.set_ylabel("L1 mean latency (ms; lower is better)")
    style_axes(axis)
    figure.tight_layout()
    save_figure(figure, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = read_summary(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(results)} unique observations from {args.summary}")
    plot_latency_bars(results, args.output_dir / "task1_l1_latency_grouped.png")
    plot_throughput_latency(results, args.output_dir / "task1_qwen_vs_l1.png")
    plot_pareto(results, args.output_dir / "task1_isolation_transport_pareto.png")


if __name__ == "__main__":
    main()
