"""E-D2: exact and bounded optimality gaps (plan D3, D4, D5; replaces Table 10).

small instances   exact chi with CP-SAT (time limit per instance); unsolved
                  instances keep their [LB, UB] interval.
medium / large    LB = clique number; UB = best TabuCol colouring within a
                  wall-clock budget (protocol: best of 5 runs of 300 s medium /
                  900 s large; --tabu-* lower this for a quick pass, and the
                  budget actually used is written to e_d_budget.json).
PLANTED           chi = k by construction.

Gaps are reported per set x family x scale for every heuristic in
seq_colors_long.csv (run e_a_main.py first): absolute and relative gap to
chi (small, PLANTED) or to the best-known UB (medium/large), plus the share of
instances that reach chi / the UB.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import common
import numpy as np
import pandas as pd

from gcpbench.bounds import clique_lb, exact_chi, tabucol_ub
from gcpbench.datasets import load_manifest
from gcpbench.graph import read_graph


def _one(args):
    set_name, row, cp_limit, tabu_med, tabu_large, runs = args
    path = os.path.join(common.SETS[set_name], row["instance_id"])
    g = read_graph(path)
    t0 = time.time()
    out = {"set": set_name, "instance_id": row["instance_id"], "family": row["family"],
           "scale": row["scale"], "n": g.n, "m": g.m}
    if set_name == "PLANTED":
        k = int(row["chi_planted"])
        out.update(lb=k, ub=k, chi=k, status="PLANTED", method="construction")
    elif row["scale"] == "small":
        lb, ub, st, _ = exact_chi(g, cp_limit)
        out.update(lb=lb, ub=ub, chi=lb if st == "OPTIMAL" else None, status=st, method="CP-SAT")
    else:
        lb, _, complete = clique_lb(g, time_limit=60)
        budget = tabu_med if row["scale"] == "medium" else tabu_large
        ub = min(tabucol_ub(g, budget, seed=s)[0] for s in range(runs))
        out.update(lb=lb, ub=ub, chi=lb if lb == ub else None,
                   status="CLOSED" if lb == ub else "BOUNDED", method=f"clique+TabuCol({runs}x{budget}s)",
                   clique_exact=complete)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="VAL,TEST-A,TEST-B,PLANTED")
    ap.add_argument("--scales", default="small,medium,large")
    ap.add_argument("--cp-limit", type=float, default=60)
    ap.add_argument("--tabu-medium", type=float, default=300)
    ap.add_argument("--tabu-large", type=float, default=900)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="bounds.csv")
    a = ap.parse_args()
    jobs = []
    for s in a.sets.split(","):
        for row in load_manifest(common.SETS[s]):
            if s == "PLANTED" or row["scale"] in a.scales.split(","):
                jobs.append((s, row, a.cp_limit, a.tabu_medium, a.tabu_large, a.runs))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_one, jobs))
    B = pd.DataFrame(res)
    path = os.path.join(common.OUT, a.out)
    if os.path.exists(path):            # merge with earlier passes (e.g. small first, then medium/large)
        old = pd.read_csv(path)
        old = old[~old.set_index(["set", "instance_id"]).index.isin(B.set_index(["set", "instance_id"]).index)]
        B = pd.concat([old, B], ignore_index=True)
    common.save(B, a.out)
    with open(os.path.join(common.OUT, "e_d_budget.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    print(B.groupby(["set", "scale", "status"]).size().to_string())
    gaps(B)


def gaps(B):
    L = pd.read_csv(os.path.join(common.OUT, "seq_colors_long.csv"))
    M = L.merge(B[["set", "instance_id", "lb", "ub", "chi", "status"]], on=["set", "instance_id"])
    M["gap_to_ub"] = M["colors"] - M["ub"]
    M["rel_gap_to_ub"] = M["gap_to_ub"] / M["ub"]
    M["gap_to_lb"] = M["colors"] - M["lb"]
    M["reaches_ub"] = M["colors"] <= M["ub"]
    common.save(M, "gaps_long.csv")
    keep = ["DSATUR", "LLaMA_H3", "Claude_H3A2A", "GPT4o_H4", "DeepSeek_H1", "GPT51_H1", "RLF", "RLF_legacy",
            "FirstFit", "DSATUR_brelaz", "SmallestLast"]
    T = (M[M.algorithm.isin(keep)].groupby(["set", "scale", "family", "algorithm"])
         .agg(N=("colors", "size"), mean_colors=("colors", "mean"), mean_ub=("ub", "mean"),
              mean_lb=("lb", "mean"), abs_gap=("gap_to_ub", "mean"), rel_gap_pct=("rel_gap_to_ub", "mean"),
              share_at_ub=("reaches_ub", "mean")).reset_index())
    T["rel_gap_pct"] *= 100
    common.save(T, "gaps_by_family_scale.csv")
    small = T[(T.scale == "small")]
    print(small[small.set == "TEST-A"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
