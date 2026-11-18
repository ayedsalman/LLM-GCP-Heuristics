"""E-E: equal-footing runtime, scaling and Pareto view (plan E1, E2, E4, E5, S0.6).

Implementations timed on the same instances, one pinned core:
  naive      the submission's code (rescoring every uncoloured vertex at every
             step: O(n(n+m))) -- the setting behind the 15-21x figures;
  incremental gcpbench.greedy (heap with lazy deletion, O((n+m) log n)), the
             same code for DSATUR and every LLM formula;
  numba      gcpbench.fastlinear for strictly linear formulas;
  nx-DSATUR  networkx.greedy_color(strategy="saturation_largest_first"), a
             common off-the-shelf DSATUR, for reference.
Reports median and IQR over --reps repetitions (protocol: 30), the per-instance
runtime ratio to DSATUR within each implementation, log-log slopes of runtime
against m, and the colours-vs-runtime Pareto front per scale.
"""

from __future__ import annotations

import argparse
import os

import common
import numpy as np
import pandas as pd
import networkx as nx

from heuristics import sequential_heuristics as L
from gcpbench.classical import CLASSICAL
from gcpbench.datasets import load_manifest
from gcpbench.fastlinear import LinearEvaluator
from gcpbench.formulas import paper_formulas
from gcpbench.graph import Graph
from gcpbench.greedy import greedy_formula
from gcpbench.timing import pin, time_fn, platform_info

METHODS = ["DSATUR", "LLaMA_H3", "DeepSeek_H1", "Claude_H3A2A", "GPT4o_H4", "GPT51_H1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="TEST-A")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--naive-reps", type=int, default=3)
    ap.add_argument("--naive-max-m", type=int, default=60000,
                    help="skip the O(n(n+m)) naive code above this many edges")
    a = ap.parse_args()
    env = pin(0)
    F = paper_formulas()
    rows = load_manifest(common.SETS[a.set])
    pick = []
    for sc in ("small", "medium", "large"):
        for fam in ("ER", "BA", "WS"):
            pick += [r for r in rows if r["scale"] == sc and r["family"] == fam][:a.per_cell]
    out = []
    for r in pick:
        G = nx.read_edgelist(os.path.join(common.SETS[a.set], r["instance_id"]), nodetype=int)
        g = Graph.from_networkx(G)
        g.static_features()                    # features precomputed outside timing for all
        ev = LinearEvaluator([g])
        base = {"instance_id": r["instance_id"], "family": r["family"], "scale": r["scale"], "n": g.n, "m": g.m}
        for name in METHODS:
            f = F[name]
            out.append({**base, "method": name, "impl": "incremental",
                        **time_fn(lambda: greedy_formula(g, f), reps=a.reps)})
            w = f.coefficient_vector()
            if w is not None:
                out.append({**base, "method": name, "impl": "numba", **time_fn(lambda: ev.colors_one(0, w), reps=a.reps)})
            if g.m <= a.naive_max_m:
                out.append({**base, "method": name, "impl": "naive",
                            **time_fn(lambda: L.greedy_color(G, name), reps=a.naive_reps, warmup=1)})
        for name in ("RLF", "FirstFit", "SmallestLast", "DSATUR_brelaz"):
            out.append({**base, "method": name, "impl": "incremental",
                        **time_fn(lambda: CLASSICAL[name](g), reps=a.reps)})
        if g.m <= a.naive_max_m:
            out.append({**base, "method": "RLF_legacy", "impl": "naive",
                        **time_fn(lambda: L.rlf_coloring(G), reps=a.naive_reps, warmup=1)})
        out.append({**base, "method": "DSATUR", "impl": "networkx",
                    **time_fn(lambda: nx.greedy_color(G, strategy="saturation_largest_first"), reps=a.reps)})
        print(r["instance_id"], "timed", flush=True)
    T = pd.DataFrame(out)
    common.save(T, "e_e_runtime_long.csv")
    # ratio to DSATUR within the same implementation, per instance
    ref = T[T.method == "DSATUR"].set_index(["instance_id", "impl"])["median_s"]
    T["ratio_to_dsatur_same_impl"] = [row.median_s / ref.get((row.instance_id, row.impl), np.nan)
                                      for row in T.itertuples()]
    summ = (T.groupby(["impl", "method", "scale"])
            .agg(median_ms=("median_s", lambda x: 1000 * np.median(x)),
                 ratio_median=("ratio_to_dsatur_same_impl", "median"),
                 ratio_q1=("ratio_to_dsatur_same_impl", lambda x: np.nanquantile(x, .25)),
                 ratio_q3=("ratio_to_dsatur_same_impl", lambda x: np.nanquantile(x, .75)),
                 n_inst=("median_s", "size")).reset_index())
    common.save(summ, "e_e_runtime_summary.csv")
    # log-log slope of runtime vs m (incremental and naive), all instances pooled
    fits = []
    for (impl, method), d in T.groupby(["impl", "method"]):
        d = d[d.m > 0]
        if d["m"].nunique() >= 3:
            k, c = np.polyfit(np.log(d["m"]), np.log(d["median_s"]), 1)
            fits.append({"impl": impl, "method": method, "slope_log_time_vs_log_m": k, "n_points": len(d)})
    common.save(pd.DataFrame(fits), "e_e_scaling_fits.csv")
    common.write_json({**platform_info(), **env, "reps": a.reps, "naive_reps": a.naive_reps,
                       "naive_max_m": a.naive_max_m, "set": a.set, "per_cell": a.per_cell}, "e_e_platform.json")
    print(summ[summ.method.isin(["DSATUR", "LLaMA_H3", "DeepSeek_H1"])].round(3).to_string(index=False))
    pareto()


def pareto():
    """Colours (TEST-A mean, all instances of the scale) vs median incremental runtime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L_ = pd.read_csv(os.path.join(common.OUT, "seq_colors_long.csv"))
    T = pd.read_csv(os.path.join(common.OUT, "e_e_runtime_long.csv"))
    T = T[T.impl == "incremental"]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sc in zip(axes, ("small", "medium", "large")):
        col = L_[(L_.set == "TEST-A") & (L_.scale == sc)].groupby("algorithm")["colors"].mean()
        rt = T[T.scale == sc].groupby("method")["median_s"].median() * 1000
        pts = [(m, col[m], rt[m]) for m in rt.index if m in col.index]
        for m, c, t in pts:
            dominated = any((c2 <= c and t2 <= t) and (c2 < c or t2 < t) for _, c2, t2 in pts)
            rows.append({"scale": sc, "method": m, "mean_colors": c, "median_ms": t, "non_dominated": not dominated})
            ax.scatter(t, c, color="#1f4e79" if not dominated else "#9aa5b1", zorder=3)
            ax.annotate(m, (t, c), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title(f"TEST-A, {sc}")
        ax.set_xlabel("median runtime, ms (incremental, log)")
        ax.set_ylabel("mean colours")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(common.OUT, "fig_pareto_colors_runtime.png"), dpi=160)
    common.save(pd.DataFrame(rows), "e_e_pareto.csv")


if __name__ == "__main__":
    main()
