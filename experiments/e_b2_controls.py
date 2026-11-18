"""E-B2 control baselines under matched selection effort, B3 tuned upper bound,
and the G5 perturbation / headroom study (plan B2, B3, G5, G7).

Arms (PROTOCOL.md section 4), all selected on VAL and reported on TEST-A:
  R-U   20 draws x 5 candidates, coefficients iid U[-1, 1]^5, best of 5 on VAL
  R-D   20 draws x 5 candidates, Dirichlet(1,1,1,1,1), best of 5 on VAL
  HC    sat only; sat + 0.01*ucn; 7 x 7 grid over (w_sat, w_deg); the
        lexicographic DSATUR is taken from e_a_main (exact engine)
  TUNED CMA-ES over the five coefficients on VAL (standardised coordinates),
        R restarts x E evaluations (protocol: 20 x 500; budget used is recorded)
  PERT  each non-zero coefficient of each strictly linear Table-6 formula
        scaled by 0.75, 0.9, 1.1, 1.25 (one at a time)

The LLM "arms" available without new API calls are the single draws reported
in the paper; experiments/run_llm.py produces the 20-seed LLM arms of E-B1,
which this script's summary picks up from results/llm/e_b1_zero_shot.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import common
import numpy as np
import pandas as pd

from gcpbench import FEATURES
from gcpbench.datasets import load_set
from gcpbench.fastlinear import LinearEvaluator
from gcpbench.formulas import paper_formulas
from gcpbench.stats import bootstrap_ci

GRID = [0, 0.01, 0.1, 0.5, 1, 2, 10]


def load_eval(set_name):
    rows, gs = [], []
    for r, g in load_set(common.SETS[set_name]):
        rows.append(r)
        gs.append(g)
    return rows, LinearEvaluator(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-cma", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    vrows, VAL = load_eval("VAL")
    trows, TA = load_eval("TEST-A")
    vscale = np.array([r["scale"] for r in vrows])
    tscale = np.array([r["scale"] for r in trows])
    cache = {}

    def ev(E, w, tag):
        key = (tag, tuple(np.round(np.asarray(w, float), 12)))
        if key not in cache:
            cache[key] = E.colors(w)
        return cache[key]

    def record(arm, draw, cand, w, extra=None):
        v = ev(VAL, w, "V")
        t = ev(TA, w, "T")
        r = {"arm": arm, "draw": draw, "candidate": cand, **{f"w_{f}": float(x) for f, x in zip(FEATURES, w)},
             "VAL_mean": v.mean(), "TEST-A_mean": t.mean()}
        for sc in ("small", "medium", "large"):
            r[f"VAL_{sc}"] = v[vscale == sc].mean()
            r[f"TEST-A_{sc}"] = t[tscale == sc].mean()
        r.update(extra or {})
        return r

    rng = np.random.default_rng(a.seed)
    rows = []
    # ---- random arms with best-of-5 selection on VAL
    for arm in ("R-U", "R-D"):
        for d in range(a.draws):
            cands = rng.uniform(-1, 1, (5, 5)) if arm == "R-U" else rng.dirichlet(np.ones(5), 5)
            for i, w in enumerate(cands):
                rows.append(record(arm, d, i, w))
        print(arm, "done", round(time.time() - t0), "s", flush=True)
    cand_df = pd.DataFrame(rows)
    sel = (cand_df.sort_values(["arm", "draw", "VAL_mean", "candidate"])
           .groupby(["arm", "draw"]).head(1).assign(selected=True))
    common.save(cand_df, "e_b2_random_candidates.csv")

    # ---- handcrafted
    hc = [record("HC", 0, 0, (1, 0, 0, 0, 0), {"label": "sat only"}),
          record("HC", 0, 1, (1, 0, 0.01, 0, 0), {"label": "sat + 0.01 ucn"})]
    grid = []
    for ws, wd in itertools.product(GRID, GRID):
        grid.append(record("GRID", 0, 0, (ws, wd, 0, 0, 0), {"label": f"({ws},{wd})"}))
    grid = pd.DataFrame(grid)
    common.save(grid, "e_b2_grid_sat_deg.csv")
    gbest = grid.sort_values(["VAL_mean", "w_sat", "w_deg"]).head(1).assign(label="grid best (VAL)")
    print("grid done", round(time.time() - t0), "s", flush=True)

    # ---- tuned (CMA-ES on standardised coefficients)
    scales = feature_scales(vrows)
    tuned = []
    if not a.skip_cma:
        import cma
        for r in range(a.restarts):
            x0 = np.random.default_rng(a.seed + r).normal(0, 1, 5)
            es = cma.CMAEvolutionStrategy(x0, 0.5, {"seed": a.seed + r + 1, "maxfevals": a.evals,
                                                    "verbose": -9, "popsize": 10})
            best = (np.inf, None)
            while not es.stop():
                X = es.ask()
                F = [ev(VAL, np.asarray(x) / scales, "V").mean() for x in X]
                es.tell(X, F)
                for x, fx in zip(X, F):
                    if fx < best[0]:
                        best = (fx, np.asarray(x) / scales)
            tuned.append(record("TUNED", r, 0, best[1], {"evals": es.countevals}))
            print(f"CMA restart {r}: VAL {best[0]:.4f}", round(time.time() - t0), "s", flush=True)
    tuned = pd.DataFrame(tuned)

    # ---- LLM formulas through the same engine (strictly linear ones)
    llm = []
    for name, f in paper_formulas().items():
        w = f.coefficient_vector()
        if w is not None:
            llm.append(record("PAPER", 0, 0, w, {"label": name}))
    llm = pd.DataFrame(llm)

    # ---- perturbation (G5)
    pert = []
    for _, r in llm.iterrows():
        if r["label"] == "DSATUR":
            continue
        w0 = np.array([r[f"w_{f}"] for f in FEATURES])
        for j, f in enumerate(FEATURES):
            if w0[j] == 0:
                continue
            for fac in (0.75, 0.9, 1.1, 1.25):
                w = w0.copy()
                w[j] *= fac
                pert.append(record("PERT", 0, 0, w, {"label": r["label"], "feature": f, "factor": fac,
                                                      "base_VAL": r["VAL_mean"], "base_TEST-A": r["TEST-A_mean"]}))
    pert = pd.DataFrame(pert)
    common.save(pert, "g5_perturbation.csv")

    allsel = pd.concat([sel, pd.DataFrame(hc), gbest, tuned, llm], ignore_index=True)
    common.save(allsel, "e_b2_arms.csv")

    # ---- summary across draws
    summ = []
    for arm, grp in allsel.groupby("arm"):
        x = grp["TEST-A_mean"].to_numpy()
        lo, hi = bootstrap_ci(x) if len(x) > 1 else (x[0], x[0])
        summ.append({"arm": arm, "n": len(x), "TEST-A_mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else 0,
                     "ci95_lo": lo, "ci95_hi": hi, "best": x.min(), "worst": x.max(),
                     **{f"TEST-A_{sc}": grp[f"TEST-A_{sc}"].mean() for sc in ("small", "medium", "large")}})
    summ = pd.DataFrame(summ)
    common.save(summ, "e_b2_summary.csv")
    # where do the paper's LLM draws fall within the random best-of-5 distributions?
    pos = []
    for _, r in llm.iterrows():
        for arm in ("R-U", "R-D"):
            x = sel[sel.arm == arm]["TEST-A_mean"].to_numpy()
            pos.append({"formula": r["label"], "arm": arm, "TEST-A_mean": r["TEST-A_mean"],
                        "share_random_draws_at_least_as_good": float((x <= r["TEST-A_mean"]).mean())})
    common.save(pd.DataFrame(pos), "e_b2_llm_vs_random.csv")
    common.write_json({"draws": a.draws, "restarts": len(tuned), "evals_per_restart": a.evals,
                       "feature_scales": dict(zip(FEATURES, scales.tolist())), "seconds": time.time() - t0,
                       "engine": "gcpbench.fastlinear (numba)"}, "e_b2_budget.json")
    print(summ.round(3).to_string(index=False))


def feature_scales(vrows):
    """Standardisation for CMA-ES: SD of deg, ucn(=deg at start), cc, twohop over all
    VAL vertices; sat scale = half the mean DSATUR colour count."""
    degs, ccs, ths = [], [], []
    for r, g in load_set(common.SETS["VAL"]):
        d, c, t = g.feature_arrays()
        degs.append(d)
        ccs.append(c)
        ths.append(t)
    d, c, t = np.concatenate(degs), np.concatenate(ccs), np.concatenate(ths)
    path = os.path.join(common.OUT, "seq_colors_long.csv")
    sat_sd = 3.0
    if os.path.exists(path):
        L = pd.read_csv(path)
        sat_sd = L[(L.set == "VAL") & (L.algorithm == "DSATUR")]["colors"].mean() / 2
    return np.array([sat_sd, d.std(), d.std(), c.std(), t.std()])


if __name__ == "__main__":
    main()
