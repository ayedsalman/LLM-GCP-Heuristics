"""E-F: GPU kernel ablation with negative controls (plan F1, F2, F3, F5, C5, E7).

One fixed local-maxima kernel (CPU emulator, bit-identical to
src/cuda/parallel_coloring.cu); only the priority function changes:
random (Jones-Plassmann), degree, -ucn, sat, 1e6*sat + deg, the four LLM CUDA
formulas of Table 6, and the anti-priority controls -deg and -sat; plus Luby's
MIS colouring. Instances: VAL, TEST-A, PLANTED, and dense ER graphs
(p = 0.5, n = 250-1000) where colour counts spread out more (F3).

Outputs: mean colours and rounds per set x scale x priority, the per-instance
identity matrix between priorities (identical colourings and identical
counts), and variance over rand() seeds for the three priorities that use it.
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import common
import networkx as nx
import numpy as np
import pandas as pd

from gcpbench.datasets import load_manifest
from gcpbench.graph import Graph, read_graph, verify_coloring
from gcpbench.parallel import PRIORITIES, color_parallel, luby_mis_coloring

RAND_PRIOS = ("random", "GPT4o", "DeepSeek")
DENSE = [(n, s) for n in (250, 500, 1000) for s in range(3)]


def _inst(job):
    set_name, iid, fam, scale, seeds = job
    if set_name == "DENSE":
        n, s = iid
        g = Graph.from_networkx(nx.erdos_renyi_graph(n, 0.5, seed=7000 + 10 * n + s))
        iid = f"ER_dense_n{n}_{s}"
    else:
        g = read_graph(f"{common.SETS[set_name]}/{iid}")
    out, cols = [], {}
    for p in PRIORITIES:
        for s in (seeds if p in RAND_PRIOS else [0]):
            c, R = color_parallel(g, p, seed=s)
            assert verify_coloring(g, c)
            out.append({"set": set_name, "instance_id": iid, "family": fam, "scale": scale, "priority": p,
                        "seed": s, "colors": len(set(c)), "rounds": R})
            if s == 0:
                cols[p] = c
    c, R = luby_mis_coloring(g, 0)
    out.append({"set": set_name, "instance_id": iid, "family": fam, "scale": scale, "priority": "Luby",
                "seed": 0, "colors": len(set(c)), "rounds": R})
    ident = []
    names = list(cols)
    for a, b in itertools.combinations(names, 2):
        ident.append({"set": set_name, "instance_id": iid, "scale": scale, "a": a, "b": b,
                      "same_coloring": cols[a] == cols[b],
                      "same_count": len(set(cols[a])) == len(set(cols[b]))})
    return out, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="VAL,TEST-A,PLANTED,DENSE")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    jobs = []
    for s in a.sets.split(","):
        if s == "DENSE":
            jobs += [("DENSE", ns, "ER-dense", "dense", seeds) for ns in DENSE]
        else:
            jobs += [(s, r["instance_id"], r["family"], r["scale"], seeds) for r in load_manifest(common.SETS[s])]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_inst, jobs, chunksize=4))
    R = pd.DataFrame([x for o, _ in res for x in o])
    I = pd.DataFrame([x for _, i in res for x in i])
    common.save(R, "e_f_ablation_long.csv")
    common.save(I, "e_f_identity_long.csv")
    s0 = R[R.seed == 0]
    means = s0.pivot_table(index=["set", "scale"], columns="priority", values="colors", aggfunc="mean")
    rounds = s0.pivot_table(index=["set", "scale"], columns="priority", values="rounds", aggfunc="mean")
    common.save(means.round(3), "e_f_mean_colors.csv", index=True)
    common.save(rounds.round(1), "e_f_mean_rounds.csv", index=True)
    idm = I.groupby(["a", "b"]).agg(same_coloring=("same_coloring", "mean"),
                                    same_count=("same_count", "mean")).reset_index()
    common.save(idm, "e_f_identity_matrix.csv")
    llm = I[I.a.isin(["GPT4o", "GPT51", "Claude", "DeepSeek"]) & I.b.isin(["GPT4o", "GPT51", "Claude", "DeepSeek"])]
    print("LLM-pair identity (share of instances):")
    print(llm.groupby(["a", "b"])[["same_coloring", "same_count"]].mean().round(3).to_string())
    per = (R[R.priority.isin(RAND_PRIOS)].groupby(["set", "scale", "priority", "instance_id"])["colors"]
           .agg(["mean", "std", "min", "max"]).reset_index())
    per["range"] = per["max"] - per["min"]
    var = (per.groupby(["set", "scale", "priority"])
           .agg(mean_colors=("mean", "mean"), mean_sd_over_seeds=("std", "mean"),
                mean_range_over_seeds=("range", "mean"), share_seed_sensitive=("range", lambda x: (x > 0).mean()))
           .reset_index())
    common.save(var, "e_f_rand_seed_variance.csv")
    print(means.round(2).to_string())


if __name__ == "__main__":
    main()
