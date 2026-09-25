#!/usr/bin/env python3.11
"""Build the C162 feasibility-boundary and scheduler-latency paper figure."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


CONTEXTS = (64, 128, 256, 512)
COLORS = {64: "#0072B2", 128: "#009E73", 256: "#E69F00", 512: "#CC79A7"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--scalability", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    scale = json.loads(args.scalability.read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)

    rows = grid["qualified_rows"]
    frontier = []
    for context in CONTEXTS:
        for unresolved in range(5):
            safe = [row["decision_time_ms"] for row in rows
                    if row["accepted_debts"] == 4
                    and row["unresolved_debts"] == unresolved
                    and row["context_length"] == context
                    and row["state"] == "QSU"]
            if safe:
                frontier.append((context, unresolved, max(safe)))
    frontier_path = args.data_dir / "frontier.tsv"
    frontier_path.write_text("\n".join(
        f"{context}\t{unresolved}\t{latest}"
        for context, unresolved, latest in frontier
    ) + "\n")
    safe_points = args.data_dir / "physical_safe.tsv"
    safe_points.write_text(
        "46\t2\tE3/128\n46\t1\tE5/512\n88\t1\tE6a/64\n"
    )
    reject_points = args.data_dir / "physical_reject.tsv"
    reject_points.write_text("46\t2\tE4/256\n89\t1\tE6b/64\n")
    latency_path = args.data_dir / "latency.tsv"
    by_debt = scale["large_grid"]["by_debt"]
    latency_path.write_text("\n".join(
        "\t".join(map(str, (
            debt,
            by_debt[str(debt)]["decision_us"]["p99"] / 1000,
            by_debt[str(debt)]["decision_us"]["max"] / 1000,
            by_debt[str(debt)]["verify_us"]["p99"] / 1000,
        ))) for debt in (1, 2, 4, 8, 16, 32, 64)
    ) + "\n")

    gp = args.data_dir / "c162_figure.gnuplot"
    plots = ", \\\n+".join(
        f"'{frontier_path}' using (int($1)=={context}?$3:1/0):2 with linespoints lw 2 pt 7 ps 0.7 lc rgb '{COLORS[context]}' title 'context {context}'"
        for context in CONTEXTS
    )
    gp.write_text(f"""
if (term eq 'svg') set terminal svg size 1250,500 enhanced font 'Arial,12'
if (term eq 'pdf') set terminal pdfcairo size 12.5in,5in enhanced font 'Arial,12'
if (term eq 'png') set terminal pngcairo size 1875,750 enhanced font 'Arial,16'
set output out
set multiplot layout 1,2 margins 0.07,0.98,0.14,0.91 spacing 0.11,0.03
set key top right opaque
set grid ytics xtics lc rgb '#dddddd'
set title '(a) Certified AI-admission frontier (accepted debt = 4)'
set xlabel 'Decision time after release (ms)'
set ylabel 'Unresolved recovery debts'
set xrange [40:120]
set yrange [-0.25:4.25]
set ytics 0,1,4
plot {plots}, \\
     '{safe_points}' using ($1-0.2):2 with points pt 7 ps 1.3 lc rgb '#0072B2' title 'physical admit', \\
     '{reject_points}' using ($1+0.2):2 with points pt 5 ps 1.5 lw 2 lc rgb '#D55E00' title 'physical reject', \\
     '{safe_points}' using ($1-0.2):2:3 with labels offset char 0.6,0.7 tc rgb '#333333' notitle, \\
     '{reject_points}' using ($1+0.2):2:3 with labels offset char 0.6,-0.8 tc rgb '#333333' notitle
unset logscale x
set title '(b) Certified scheduler and verifier latency'
set xlabel 'Recovery debts'
set ylabel 'Latency (ms)'
set xrange [0.8:80]
set yrange [0:5.25]
set logscale x 2
set xtics ('1' 1,'2' 2,'4' 4,'8' 8,'16' 16,'32' 32,'64' 64)
set key top left opaque
plot '{latency_path}' using 1:2 with linespoints lw 2 pt 7 lc rgb '#0072B2' title 'scheduler p99', \\
     '{latency_path}' using 1:3 with linespoints lw 2 pt 5 lc rgb '#D55E00' title 'scheduler max', \\
     '{latency_path}' using 1:4 with linespoints lw 2 pt 9 lc rgb '#009E73' title 'verifier p99', \\
     5 with lines dt 2 lw 2 lc rgb '#555555' title '5 ms control budget'
unset multiplot
""".strip() + "\n")
    for term, output in (("svg", args.svg), ("pdf", args.pdf), ("png", args.png)):
        subprocess.run([
            "gnuplot", "-e", f"term='{term}';out='{output}'", str(gp)
        ], check=True)
    print(json.dumps({
        "status": "C162_FIGURE_BUILT",
        "svg": str(args.svg), "pdf": str(args.pdf), "png": str(args.png),
        "frontier_points": len(frontier),
    }, indent=2))


if __name__ == "__main__":
    main()
