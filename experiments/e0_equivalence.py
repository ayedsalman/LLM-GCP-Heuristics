"""S0.1 / E1 equivalence report: the new evaluators reproduce the original code.

A. Original implementation (src/heuristics/sequential_heuristics.py, unchanged)
   vs the incremental evaluator: identical colour of every vertex, for DSATUR
   and the 12 formulas of Table 6, plus First-Fit; on all small and medium VAL
   instances (the original naive loop is too slow for the 30 large ones in a
   reasonable time; use --legacy-large to include them).
B. Independent full-argmax reference (all uncoloured scores recomputed and the
   argmax taken every step, lowest id on ties) vs the incremental evaluator: all
   140 VAL instances. This checks the heap/lazy-deletion logic everywhere.
C. Numba engine (used for controls/tuning) vs the exact engine: colour counts
   for every strictly linear formula on all 140 VAL instances.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import common  # noqa: F401
import numpy as np
import pandas as pd
import networkx as nx

from heuristics import sequential_heuristics as L
from gcpbench.datasets import load_manifest
from gcpbench.fastlinear import LinearEvaluator
from gcpbench.formulas import paper_formulas
from gcpbench.graph import Graph, n_colors
from gcpbench.greedy import greedy_formula
from gcpbench.classical import first_fit

SET = "data/VAL"


def argmax_reference(g: Graph, f):
    """Full recomputation of every uncoloured vertex's score at every step
    (vectorised: numpy float64/int64 element-wise ops give the same IEEE results
    as the scalar expression), argmax with the lowest id on ties."""
    n = g.n
    deg_l, cc_l, th_l = g.static_features()
    deg = np.asarray(deg_l, dtype=np.int64)
    cc = np.asarray(cc_l, dtype=np.float64)
    th = np.asarray(th_l, dtype=np.int64)
    color = [-1] * n
    sat = np.zeros(n, dtype=np.int64)
    ucn = deg.copy()
    nbr = [set() for _ in range(n)]
    unc = np.ones(n, dtype=bool)
    for _ in range(n):
        s = np.broadcast_to(np.asarray(f.fn(sat, deg, ucn, cc, th)), (n,)).astype(np.float64, copy=True)
        s[~unc] = -np.inf
        bv = int(np.flatnonzero(s == s.max())[0])
        c = 0
        while c in nbr[bv]:
            c += 1
        color[bv] = c
        unc[bv] = False
        for u in g.adj[bv]:
            if color[u] == -1:
                ucn[u] -= 1
                if c not in nbr[u]:
                    nbr[u].add(c)
                    sat[u] += 1
    return color


def part_a(row):
    G = nx.read_edgelist(f"{SET}/{row['instance_id']}", nodetype=int)
    g = Graph.from_networkx(G)
    out = []
    for name, f in paper_formulas().items():
        ref = L.greedy_color(G, name)
        new = greedy_formula(g, f)
        same = all(ref[g.labels[i]] == new[i] for i in range(g.n))
        out.append({"part": "A", "instance_id": row["instance_id"], "scale": row["scale"],
                    "heuristic": name, "identical_coloring": same,
                    "colors_ref": L.num_colors_used(ref), "colors_new": n_colors(new)})
    ref = L.first_fit_coloring(G)
    new = first_fit(g)
    out.append({"part": "A", "instance_id": row["instance_id"], "scale": row["scale"], "heuristic": "FirstFit",
                "identical_coloring": all(ref[g.labels[i]] == new[i] for i in range(g.n)),
                "colors_ref": L.num_colors_used(ref), "colors_new": n_colors(new)})
    return out


def part_b(row):
    G = nx.read_edgelist(f"{SET}/{row['instance_id']}", nodetype=int)
    g = Graph.from_networkx(G)
    out = []
    for name, f in paper_formulas().items():
        ref = argmax_reference(g, f)
        new = greedy_formula(g, f)
        out.append({"part": "B", "instance_id": row["instance_id"], "scale": row["scale"], "heuristic": name,
                    "identical_coloring": ref == new, "colors_ref": n_colors(ref), "colors_new": n_colors(new)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--legacy-large", action="store_true")
    a = ap.parse_args()
    rows = load_manifest(SET)
    rows_a = [r for r in rows if a.legacy_large or r["scale"] != "large"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [x for chunk in ex.map(part_a, rows_a) for x in chunk]
        res += [x for chunk in ex.map(part_b, rows) for x in chunk]
    # part C
    gs = []
    for r in rows:
        G = nx.read_edgelist(f"{SET}/{r['instance_id']}", nodetype=int)
        g = Graph.from_networkx(G)
        g.name = r["instance_id"]
        gs.append((r, g))
    ev = LinearEvaluator([g for _, g in gs])
    for name, f in paper_formulas().items():
        w = f.coefficient_vector()
        if w is None:
            continue
        for i, (r, g) in enumerate(gs):
            res.append({"part": "C", "instance_id": r["instance_id"], "scale": r["scale"], "heuristic": name,
                        "identical_coloring": None, "colors_ref": n_colors(greedy_formula(g, f)),
                        "colors_new": ev.colors_one(i, w)})
    df = pd.DataFrame(res)
    df["same_count"] = df["colors_ref"] == df["colors_new"]
    common.save(df, "e0_equivalence_detail.csv")
    summ = df.groupby(["part", "scale"]).agg(pairs=("same_count", "size"),
                                             identical_colorings=("identical_coloring", "sum"),
                                             identical_counts=("same_count", "sum")).reset_index()
    common.save(summ, "e0_equivalence_summary.csv")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
