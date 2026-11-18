"""E-A and the main sequential tables (plan A1-A7, B6, C4, D7, D8, D9, G6).

1. Evaluates the 12 Table-6 formulas and 10 classical baselines on VAL, TEST-A,
   TEST-B and PLANTED (cached; colourings verified).
2. Reproduction check: VAL means vs Table 7 of the submission.
3. Selection optimism (E-A): for each formula, (VAL - TEST-A) mean colours minus
   the same for DSATUR, per scale, with a bootstrap CI (instances resampled
   independently within each set).
4. Primary analysis on TEST-A and TEST-B (B6): paired differences vs DSATUR,
   bootstrap CI, Wilcoxon + Holm, TOST (delta = 0.10); pooled, per scale,
   per family x scale.
5. Friedman/Nemenyi on TEST-A with ONE common algorithm set at every scale and
   the full test table (D7).
6. Strictly linear subset (C4): DeepSeek H1 is bilinear; its best strictly
   linear formula in Table 6 is H1A.
"""

from __future__ import annotations

import argparse

import common
import numpy as np
import pandas as pd

from gcpbench.evaluate import evaluate
from gcpbench.stats import paired_vs_baseline, friedman_nemenyi, test_table_row, bootstrap_ci

SEL = common.PAPER_SELECTED
FRIEDMAN_SET = ["DSATUR", "LLaMA_H3", "Claude_H3A2A", "GPT4o_H4", "DeepSeek_H1", "GPT51_H1",
                "RLF_legacy", "FirstFit"]


def optimism(val, test, h, base="DSATUR", n_boot=10000, seed=0):
    """Diff-in-diff with independent bootstrap of VAL and TEST instances."""
    dv = (val[h] - val[base]).to_numpy(float)
    dt = (test[h] - test[base]).to_numpy(float)
    rng = np.random.default_rng(seed)
    bv = dv[rng.integers(0, len(dv), (n_boot, len(dv)))].mean(1)
    bt = dt[rng.integers(0, len(dt), (n_boot, len(dt)))].mean(1)
    b = bv - bt
    return dv.mean() - dt.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975), dv.mean(), dt.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sets", default="VAL,TEST-A,TEST-B,PLANTED")
    a = ap.parse_args()
    H = common.paper_heuristics()
    longs = {}
    for s in a.sets.split(","):
        print("evaluating", s, flush=True)
        longs[s] = evaluate(common.SETS[s], H, workers=a.workers).assign(set=s)
    allres = pd.concat(longs.values(), ignore_index=True)
    common.save(allres, "seq_colors_long.csv")

    # ---- 2. reproduction of Table 7 on VAL
    val = longs["VAL"]
    m = val.groupby(["scale", "algorithm"])["colors"].mean().unstack(0)[["small", "medium", "large"]]
    rep = []
    names = {"DSATUR": "DSATUR", "RLF": "RLF_legacy", "First-Fit": "FirstFit", **SEL}
    for paper_name, ours in names.items():
        p = common.PAPER_TABLE7[paper_name]
        rep.append({"paper_row": paper_name, "heuristic": ours,
                    **{f"paper_{s}": v for s, v in zip(("small", "medium", "large"), p)},
                    **{f"VAL_{s}": m.loc[ours, s] for s in ("small", "medium", "large")}})
    rep = pd.DataFrame(rep)
    common.save(rep, "table7_reproduction_VAL.csv")
    print(rep.round(2).to_string(index=False))

    # ---- means per set x scale (and family) for every heuristic
    means = allres.groupby(["set", "scale", "algorithm"])["colors"].mean().unstack("algorithm")
    common.save(means.round(4), "seq_means_by_set_scale.csv", index=True)
    fam = allres.groupby(["set", "scale", "family", "algorithm"])["colors"].mean().unstack("algorithm")
    common.save(fam.round(4), "seq_means_by_set_scale_family.csv", index=True)

    # ---- 3. optimism (E-A)
    if "TEST-A" in longs:
        rows = []
        for scale in ("small", "medium", "large", "all"):
            def w(df):
                d = df if scale == "all" else df[df["scale"] == scale]
                return d.pivot_table(index="instance_id", columns="algorithm", values="colors")
            V, T = w(longs["VAL"]), w(longs["TEST-A"])
            for h in [x for x in H if x != "DSATUR"]:
                est, lo, hi, dv, dt = optimism(V, T, h)
                rows.append({"scale": scale, "heuristic": h, "VAL_minus_DSATUR": dv,
                             "TEST_minus_DSATUR": dt, "optimism": est, "ci95_lo": lo, "ci95_hi": hi,
                             "selected_in_paper": h in SEL.values()})
        opt = pd.DataFrame(rows)
        common.save(opt, "e_a_optimism.csv")
        print(opt[opt["selected_in_paper"]].round(3).to_string(index=False))

    # ---- 4. paired analysis vs DSATUR
    tabs = []
    for s in ("TEST-A", "TEST-B", "VAL"):
        if s not in longs:
            continue
        L = longs[s]
        strata = [("all", "all", L)] + [(sc, "all", L[L.scale == sc]) for sc in ("small", "medium", "large")]
        strata += [(sc, f, L[(L.scale == sc) & (L.family == f)]) for sc in ("small", "medium", "large")
                   for f in ("ER", "BA", "WS")]
        for sc, f, d in strata:
            W = d.pivot_table(index="instance_id", columns="algorithm", values="colors")
            t = paired_vs_baseline(W, "DSATUR", n_boot=10000)
            t.insert(0, "family", f)
            t.insert(0, "scale", sc)
            t.insert(0, "set", s)
            tabs.append(t)
    paired = pd.concat(tabs, ignore_index=True)
    common.save(paired, "paired_vs_dsatur.csv")
    show = paired[(paired.set == "TEST-A") & (paired.family == "all")
                  & paired.algorithm.isin(list(SEL.values()) + ["RLF", "RLF_legacy", "DSATUR_brelaz",
                                                                "SmallestLast", "FirstFit"])]
    print(show[["scale", "algorithm", "mean_diff", "ci95_lo", "ci95_hi", "share_equal", "p_holm",
                "p_tost", "verdict"]].round(3).to_string(index=False))

    # ---- 5. Friedman / Nemenyi with a common algorithm set (D7)
    trows = []
    for s in ("VAL", "TEST-A", "TEST-B"):
        if s not in longs:
            continue
        for sc in ("small", "medium", "large"):
            d = longs[s][longs[s].scale == sc]
            W = d.pivot_table(index="instance_id", columns="algorithm", values="colors")[FRIEDMAN_SET]
            res = friedman_nemenyi(W)
            row = test_table_row("Friedman (8 algorithms, as paper small/medium)", s, sc, res)
            row.update({f"rank_{k}": round(v, 3) for k, v in res["avg_ranks"].items()})
            trows.append(row)
    tt = pd.DataFrame(trows)
    common.save(tt, "test_table_friedman.csv")
    print(tt[["set", "scale", "N", "k", "statistic", "df", "p", "CD"]].to_string(index=False))

    # ---- 6. strictly linear subset (C4)
    from gcpbench.formulas import paper_formulas
    lin = {n: f.strictly_linear for n, f in paper_formulas().items()}
    common.write_json(lin, "strict_linearity_table6.json")


if __name__ == "__main__":
    main()
