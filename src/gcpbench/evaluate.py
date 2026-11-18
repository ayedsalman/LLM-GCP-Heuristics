"""Evaluate heuristics on a registered set, with an on-disk cache.

Results are cached per (set lock hash, heuristic key) in
``results/cache/<set>/<key>.json`` so that every number in the revised paper
comes from one evaluation (plan Section 1, "build the infrastructure once").
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .classical import CLASSICAL
from .datasets import load_set, load_manifest
from .formulas import Formula
from .graph import n_colors, verify_coloring
from .greedy import greedy_formula

CACHE_DIR = "results/cache"


def _key(kind: str, payload: str) -> str:
    return hashlib.sha256(f"{kind}:{payload}".encode()).hexdigest()[:20]


def _set_tag(set_dir: str) -> str:
    rows = load_manifest(set_dir)
    h = hashlib.sha256("".join(sorted(r["sha256"] for r in rows)).encode()).hexdigest()[:12]
    return f"{os.path.basename(os.path.normpath(set_dir))}-{h}"


def _run_on_set(args):
    set_dir, kind, payload, name = args
    out = []
    for row, g in load_set(set_dir):
        if kind == "formula":
            f = Formula(name, payload)
            cols = greedy_formula(g, f)
        elif kind == "classical":
            cols = CLASSICAL[payload](g)
        elif kind == "linear":
            from .formulas import linear_formula
            f = linear_formula(name, json.loads(payload))
            cols = greedy_formula(g, f)
        else:
            raise ValueError(kind)
        if not verify_coloring(g, cols):
            raise AssertionError(f"{name} produced an invalid colouring on {row['instance_id']}")
        out.append({"instance_id": row["instance_id"], "family": row["family"], "scale": row["scale"],
                    "colors": n_colors(cols)})
    return out


def evaluate(set_dir: str, heuristics: dict, workers: int = 1, cache: bool = True) -> pd.DataFrame:
    """heuristics: name -> ("formula", expr) | ("classical", key) | ("linear", json-coefs).
    Returns a long DataFrame [instance_id, family, scale, algorithm, colors]."""
    tag = _set_tag(set_dir)
    cdir = os.path.join(CACHE_DIR, tag)
    os.makedirs(cdir, exist_ok=True)
    todo, frames = [], []
    for name, (kind, payload) in heuristics.items():
        path = os.path.join(cdir, _key(kind, payload) + ".json")
        if cache and os.path.exists(path):
            with open(path) as f:
                rows = json.load(f)
            frames.append(pd.DataFrame(rows).assign(algorithm=name))
        else:
            todo.append((name, kind, payload, path))
    if todo:
        args = [(set_dir, k, p, n) for n, k, p, _ in todo]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                results = list(ex.map(_run_on_set, args))
        else:
            results = [_run_on_set(a) for a in args]
        for (name, _, _, path), rows in zip(todo, results):
            with open(path, "w") as f:
                json.dump(rows, f)
            frames.append(pd.DataFrame(rows).assign(algorithm=name))
    return pd.concat(frames, ignore_index=True)


def wide(long: pd.DataFrame, value: str = "colors") -> pd.DataFrame:
    return long.pivot_table(index="instance_id", columns="algorithm", values=value, aggfunc="first")
