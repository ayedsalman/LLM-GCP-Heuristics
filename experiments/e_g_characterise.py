"""E-G: characterising the heuristics (plan G1, G2, G3, G7).

G1  scale-aware effective weights. During each run, every ``probe`` steps the
    SD of each feature across the uncoloured vertices is recorded; the
    effective weight of feature j is |w_j| * SD_j (averaged over probes and
    instances), reported as a share of the total. Also the Spearman
    correlation between the score and sat across uncoloured vertices.
G2  ordering similarity with DSATUR: Spearman correlation of the colouring
    positions of the vertices, and the share of instances with exactly
    DSATUR's colour count.
G3  relatedness among the Table-6 formulas: cosine of standardised
    coefficient vectors, mean ordering correlation, correlation of
    per-instance colour counts (heat maps).
G7  DSATUR as a point of the linear family: w = (10000, 1, 0, 0, 0); cosine of
    each standardised formula with DSATUR's direction.
"""

from __future__ import annotations

import argparse
import itertools
import os

import common
import numpy as np
import pandas as pd
from scipy import stats

from gcpbench import FEATURES
from gcpbench.datasets import load_set
from gcpbench.formulas import paper_formulas
from gcpbench.greedy import greedy_scored


def run_probe(g, f, n_probes=40):
    deg, cc, th = (np.asarray(x, float) for x in g.static_features())
    dcc = deg * cc
    rec = []

    def probe(color, sat, ucn):
        col = np.asarray(color)
        unc = col < 0
        if unc.sum() < 3:
            return
        S = np.asarray(sat, float)[unc]
        U = np.asarray(ucn, float)[unc]
        feats = {"sat": S, "deg": deg[unc], "ucn": U, "cc": cc[unc], "twohop": th[unc], "deg*cc": dcc[unc]}
        score = np.asarray(f.fn(S, deg[unc], U, cc[unc], th[unc]), float)
        rho = stats.spearmanr(score, S).statistic if S.std() > 0 and score.std() > 0 else np.nan
        rec.append({**{k: v.std() for k, v in feats.items()}, "rho_score_sat": rho})

    every = max(1, g.n // n_probes)
    _, order = greedy_scored(g, f.fn, dynamic=bool(f.dynamic), return_order=True, probe=probe,
                             probe_every=every)
    return pd.DataFrame(rec), order


def terms(f):
    """Coefficients by term for the Table-6 formulas, including the bilinear deg*cc."""
    if f.coefficients is not None:
        return dict(f.coefficients)
    # bilinear formulas: evaluate the deg*cc coefficient numerically
    base = f.fn(0, 0, 0, 0, 0)
    c = {}
    for j, x in enumerate(FEATURES):
        args = [0] * 5
        args[j] = 1
        v = f.fn(*args) - base
        if v:
            c[x] = v
    c["deg*cc"] = f.fn(0, 1, 0, 1, 0) - base - c.get("deg", 0) - c.get("cc", 0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="TEST-A")
    ap.add_argument("--per-cell", type=int, default=4)
    a = ap.parse_args()
    F = paper_formulas()
    inst = []
    for row, g in load_set(common.SETS[a.set]):
        if sum(1 for r, _ in inst if r["scale"] == row["scale"] and r["family"] == row["family"]) < a.per_cell:
            inst.append((row, g))
    eff, orders = [], {}
    for row, g in inst:
        for name, f in F.items():
            P, order = run_probe(g, f)
            orders[(row["instance_id"], name)] = order
            w = terms(f)
            sd = P.mean()
            e = {k: abs(w.get(k, 0)) * sd.get(k, 0) for k in list(FEATURES) + ["deg*cc"]}
            tot = sum(e.values()) or 1
            eff.append({"instance_id": row["instance_id"], "scale": row["scale"], "family": row["family"],
                        "formula": name, **{f"share_{k}": v / tot for k, v in e.items()},
                        "rho_score_sat": P["rho_score_sat"].mean()})
        print(row["instance_id"], flush=True)
    E = pd.DataFrame(eff)
    common.save(E, "g1_effective_weights_long.csv")
    G1 = E.groupby(["formula", "scale"]).mean(numeric_only=True).round(3)
    common.save(G1, "g1_effective_weights.csv", index=True)

    # G2 ordering similarity with DSATUR
    g2 = []
    for row, g in inst:
        pos_d = np.empty(g.n)
        pos_d[orders[(row["instance_id"], "DSATUR")]] = np.arange(g.n)
        for name in F:
            pos = np.empty(g.n)
            pos[orders[(row["instance_id"], name)]] = np.arange(g.n)
            g2.append({"instance_id": row["instance_id"], "scale": row["scale"], "formula": name,
                       "spearman_order_vs_dsatur": stats.spearmanr(pos, pos_d).statistic})
    G2 = pd.DataFrame(g2)
    Lp = os.path.join(common.OUT, "seq_colors_long.csv")
    L = pd.read_csv(Lp)
    L = L[L.set == a.set]
    W = L.pivot_table(index="instance_id", columns="algorithm", values="colors")
    same = {name: float((W[name] == W["DSATUR"]).mean()) for name in F}
    G2s = G2.groupby(["formula", "scale"])["spearman_order_vs_dsatur"].mean().unstack().round(3)
    G2s["share_same_colors_as_DSATUR_all_TEST"] = pd.Series(same)
    common.save(G2s, "g2_ordering_similarity.csv", index=True)

    # G3 relatedness + G7
    allP = []
    for row, g in inst[: 3 * a.per_cell]:
        P, _ = run_probe(g, F["DSATUR"])
        allP.append(P.mean())
    sd = pd.concat(allP, axis=1).mean(axis=1)
    keys = list(FEATURES) + ["deg*cc"]
    vec = {n: np.array([terms(f).get(k, 0) * sd.get(k, 0) for k in keys]) for n, f in F.items()}
    names = list(F)

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    C = pd.DataFrame([[cos(vec[a_], vec[b]) for b in names] for a_ in names], index=names, columns=names)
    O = pd.DataFrame(np.nan, index=names, columns=names)
    for a_, b in itertools.combinations_with_replacement(names, 2):
        rs = []
        for row, g in inst:
            pa = np.empty(g.n)
            pa[orders[(row["instance_id"], a_)]] = np.arange(g.n)
            pb = np.empty(g.n)
            pb[orders[(row["instance_id"], b)]] = np.arange(g.n)
            rs.append(stats.spearmanr(pa, pb).statistic)
        O.loc[a_, b] = O.loc[b, a_] = np.mean(rs)
    K = W[names].corr(method="spearman")
    common.save(C.round(3), "g3_coef_cosine.csv", index=True)
    common.save(O.round(3), "g3_order_correlation.csv", index=True)
    common.save(K.round(3), "g3_color_correlation.csv", index=True)
    g7 = pd.DataFrame({"cosine_with_DSATUR_direction": C["DSATUR"]})
    common.save(g7.round(4), "g7_dsatur_direction.csv", index=True)
    heatmaps(C, O, K)
    print(G1.reset_index().query("scale=='medium'")[["formula", "share_sat", "share_deg", "share_ucn",
                                                     "share_cc", "share_twohop", "share_deg*cc",
                                                     "rho_score_sat"]].to_string(index=False))
    print(G2s.to_string())


def heatmaps(C, O, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, M, t in zip(axes, (C, O, K), ("coefficient cosine (standardised)", "ordering Spearman",
                                          "per-instance colour-count Spearman")):
        im = ax.imshow(M.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M.index, fontsize=7)
        ax.set_title(t, fontsize=9)
        for i in range(len(M)):
            for j in range(len(M)):
                ax.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(os.path.join(common.OUT, "fig_g3_relatedness.png"), dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
