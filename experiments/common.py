"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

OUT = os.path.join("results", "revision")
os.makedirs(OUT, exist_ok=True)

SETS = {"VAL": "data/VAL", "TEST-A": "data/TEST-A", "TEST-B": "data/TEST-B", "PLANTED": "data/PLANTED"}

# The heuristic reported for each model in Table 7 of the submission (best available
# per scale was used there; here the Trial-1 selection is kept fixed across scales).
PAPER_SELECTED = {"LLaMA": "LLaMA_H3", "Claude": "Claude_H3A2A", "GPT-4o": "GPT4o_H4",
                  "DeepSeek": "DeepSeek_H1", "GPT-5.1": "GPT51_H1"}
PAPER_TABLE7 = {  # mean colours reported in the manuscript (small, medium, large)
    "LLaMA": (4.24, 9.70, 10.33), "DSATUR": (4.24, 9.83, 10.33), "Claude": (4.26, 10.05, 10.93),
    "GPT-4o": (4.30, 10.03, 10.83), "DeepSeek": (4.50, 9.85, 10.50), "GPT-5.1": (4.76, 11.08, 12.03),
    "RLF": (4.76, 11.13, 12.47), "First-Fit": (4.86, 11.20, 12.07)}

CLASSICAL_KEYS = ["DSATUR", "DSATUR_lex", "DSATUR_brelaz", "DSATUR_sewell", "IDO", "WelshPowell",
                  "SmallestLast", "FirstFit", "RLF", "RLF_legacy"]


def paper_heuristics(include_classical=True):
    from gcpbench.formulas import PAPER_SEQUENTIAL
    h = {name: ("formula", expr) for name, (expr, *_rest) in PAPER_SEQUENTIAL.items()}
    if include_classical:
        h.update({k: ("classical", k) for k in CLASSICAL_KEYS})
    return h


def save(df, name, index=False):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def write_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)
