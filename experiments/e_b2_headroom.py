"""Headroom test (plan B3, G5): the VAL-selected tuned formula and the best
handcrafted rule, re-evaluated with the exact engine on TEST-A and TEST-B, and
compared per instance with DSATUR and with the paper's LLaMA H3 (paired
bootstrap CI, Wilcoxon + Holm, TOST)."""

from __future__ import annotations

import json
import os

import common
import pandas as pd

from gcpbench import FEATURES
from gcpbench.evaluate import evaluate, wide
from gcpbench.stats import paired_vs_baseline


def main():
    arms = pd.read_csv(os.path.join(common.OUT, "e_b2_arms.csv"))
    tuned = arms[arms.arm == "TUNED"].sort_values("VAL_mean").iloc[0]
    hc = arms[arms.arm == "HC"].sort_values("VAL_mean").iloc[0]
    H = {"TUNED_best_VAL": ("linear", json.dumps([float(tuned[f"w_{f}"]) for f in FEATURES])),
         "HC_best_VAL": ("linear", json.dumps([float(hc[f"w_{f}"]) for f in FEATURES])),
         "DSATUR": ("classical", "DSATUR"),
         "LLaMA_H3": ("formula", "1.2*sat - 0.6*cc + 0.1*ucn")}
    out = []
    for s in ("TEST-A", "TEST-B"):
        L = evaluate(common.SETS[s], H)
        W = wide(L)
        for base in ("DSATUR", "LLaMA_H3"):
            t = paired_vs_baseline(W, base, algorithms=["TUNED_best_VAL", "HC_best_VAL"]
                                   + (["LLaMA_H3"] if base == "DSATUR" else []))
            t.insert(0, "set", s)
            out.append(t)
    T = pd.concat(out, ignore_index=True)
    common.save(T, "e_b2_headroom_paired.csv")
    common.write_json({"tuned_w": {f: float(tuned[f"w_{f}"]) for f in FEATURES},
                       "hc_label": hc.get("label"), "hc_w": {f: float(hc[f"w_{f}"]) for f in FEATURES}},
                      "e_b2_headroom_formulas.json")
    print(T[["set", "algorithm", "baseline", "mean_diff", "ci95_lo", "ci95_hi", "p_holm", "p_tost",
             "verdict"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
