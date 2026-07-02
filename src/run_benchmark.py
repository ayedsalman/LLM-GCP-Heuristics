"""
run_benchmark.py

End-to-end pipeline: load generated graph instances, run every
sequential heuristic on every instance, and write a per-instance
results CSV in the format expected by analysis/statistical_analysis.py.

Usage
-----
    python data/generate_graphs.py --out-dir data/graphs
    python src/run_benchmark.py --graphs-dir data/graphs --scale medium \
        --out results/seq_medium.csv

Then analyze:
    python analysis/statistical_analysis.py \
        --input results/seq_medium.csv --alpha 0.05 --baseline DSATUR
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import networkx as nx

from heuristics.sequential_heuristics import (
    HEURISTICS,
    greedy_color,
    first_fit_coloring,
    rlf_coloring,
    num_colors_used,
    verify_valid_coloring,
)

# The "best available heuristic per model" selection used in the paper's
# headline results tables -- adjust per scale if a different variant wins
# on your own re-run (this is exactly the situation Section 4.1.3
# describes: refinement outcomes are scale-dependent).
BEST_PER_MODEL = {
    "DSATUR": "DSATUR",
    "LLaMA": "LLaMA_H3",
    "Claude": "Claude_H3A2A",
    "GPT4o": "GPT4o_H4",
    "DeepSeek": "DeepSeek_H1",
    "GPT51": "GPT51_H1",
}


def run_one_graph(G: nx.Graph) -> dict:
    """Returns {algorithm_name: color_count} for one graph instance."""
    row = {}

    for label, heuristic_name in BEST_PER_MODEL.items():
        coloring = greedy_color(G, heuristic_name)
        assert verify_valid_coloring(G, coloring), f"{heuristic_name} produced invalid coloring"
        row[label] = num_colors_used(coloring)

    ff = first_fit_coloring(G)
    assert verify_valid_coloring(G, ff)
    row["FirstFit"] = num_colors_used(ff)

    rlf = rlf_coloring(G)
    assert verify_valid_coloring(G, rlf)
    row["RLF"] = num_colors_used(rlf)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", required=True)
    parser.add_argument("--scale", required=True, choices=["small", "medium", "large"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-limit-warn", type=float, default=30.0,
                         help="print a warning if any single instance exceeds this many seconds")
    args = parser.parse_args()

    manifest_path = os.path.join(args.graphs_dir, "manifest.csv")
    with open(manifest_path) as f:
        manifest = [row for row in csv.DictReader(f) if row["scale"] == args.scale]

    if not manifest:
        raise SystemExit(f"No instances found for scale={args.scale} in {manifest_path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["instance_id"] + list(BEST_PER_MODEL.keys()) + ["FirstFit", "RLF"]

    with open(args.out, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, entry in enumerate(manifest):
            path = os.path.join(args.graphs_dir, entry["instance_id"])
            G = nx.read_edgelist(path, nodetype=int)

            t0 = time.perf_counter()
            row = run_one_graph(G)
            elapsed = time.perf_counter() - t0
            if elapsed > args.time_limit_warn:
                print(f"[warn] {entry['instance_id']} took {elapsed:.1f}s")

            row["instance_id"] = entry["instance_id"]
            writer.writerow(row)
            print(f"[{i+1}/{len(manifest)}] {entry['instance_id']} done ({elapsed:.2f}s)")

    print(f"\nResults written to {args.out}")
    print(
        "Next: python analysis/statistical_analysis.py "
        f"--input {args.out} --alpha 0.05 --baseline DSATUR"
    )


if __name__ == "__main__":
    main()
