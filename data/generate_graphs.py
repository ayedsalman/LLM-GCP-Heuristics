"""
generate_graphs.py

Generates the 140-instance synthetic benchmark suite described in the
paper (Table: Synthetic Graph Dataset Generation Parameters and
Instance Counts), spanning three graph families and three scales.

    ER (Erdos-Renyi):        uniform random connectivity
    BA (Barabasi-Albert):    preferential attachment, scale-free hubs
    WS (Watts-Strogatz):     small-world, high local clustering

Usage
-----
    python generate_graphs.py --out-dir data/graphs --seed 42

Produces one .gpickle (networkx graph) or .edgelist file per instance,
named <family>_<scale>_<index>.edgelist, plus a manifest CSV listing
every instance and its generation parameters.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import networkx as nx

# Exactly matches the paper's dataset table.
DATASET_SPEC = [
    # (family, scale, n_range, param_fn, count)
    ("ER", "small",  (30, 120),   lambda rng: {"p": rng.choice([0.10, 0.20])}, 20),
    ("ER", "medium", (500, 1000), lambda rng: {"p": 0.05},                    20),
    ("ER", "large",  (2000, 5000), lambda rng: {"p": rng.choice([0.01, 0.02])}, 10),

    ("BA", "small",  (30, 120),   lambda rng: {"m": rng.choice([2, 4])}, 20),
    ("BA", "medium", (500, 1000), lambda rng: {"m": 4},                 20),
    ("BA", "large",  (2000, 5000), lambda rng: {"m": 4},                10),

    ("WS", "small",  (120, 120),   lambda rng: {"k": 8,  "p": 0.1}, 10),
    ("WS", "medium", (500, 1000),  lambda rng: {"k": 20, "p": 0.1}, 20),
    ("WS", "large",  (2000, 5000), lambda rng: {"k": 20, "p": 0.1}, 10),
]


def make_graph(family: str, n: int, params: dict, rng: random.Random) -> nx.Graph:
    if family == "ER":
        return nx.erdos_renyi_graph(n, params["p"], seed=rng.randint(0, 2**31))
    if family == "BA":
        m = min(params["m"], n - 1)  # m must be < n
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 2**31))
    if family == "WS":
        k = min(params["k"], n - 1 if (n - 1) % 2 == 0 else n - 2)
        return nx.watts_strogatz_graph(n, k, params["p"], seed=rng.randint(0, 2**31))
    raise ValueError(f"unknown family: {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/graphs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    manifest_rows = []

    for family, scale, (n_lo, n_hi), param_fn, count in DATASET_SPEC:
        for idx in range(count):
            n = rng.randint(n_lo, n_hi) if n_lo != n_hi else n_lo
            params = param_fn(rng)
            g = make_graph(family, n, params, rng)

            fname = f"{family}_{scale}_{idx:03d}.edgelist"
            path = os.path.join(args.out_dir, fname)
            nx.write_edgelist(g, path, data=False)

            manifest_rows.append({
                "instance_id": fname,
                "family": family,
                "scale": scale,
                "n_vertices": g.number_of_nodes(),
                "n_edges": g.number_of_edges(),
                "params": params,
            })

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} instances into {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
