"""LLM experiment pipeline: E-B1 (multi-seed zero-shot), E-B3 (uniform refinement
arms T/G/R/N), E-H (budget-matched evolutionary arm). Plan B1, B4, B5, B8, C1, A8.

All development decisions use VAL only; TEST sets are evaluated only for the
heuristic that the frozen selection rule picked (PROTOCOL.md section 2).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from ..datasets import load_set
from ..evaluate import evaluate, wide
from ..formulas import Formula
from ..graph import Graph, verify_coloring
from ..greedy import greedy_formula
from ..validator import validate
from . import prompts
from .explain import describe_instance, describe_successes
from .parse import parse_response
from .registry import append, new_id, save_raw, sha

N_FAILURE_INSTANCES = 5
MAX_REQUERIES = 3


def heldout_graphs():
    """Five small graphs outside every evaluation set (Stage-2 colouring check)."""
    gs = [nx.erdos_renyi_graph(60, 0.15, seed=101), nx.barabasi_albert_graph(60, 3, seed=102),
          nx.watts_strogatz_graph(60, 6, 0.1, seed=103), nx.erdos_renyi_graph(100, 0.1, seed=104),
          nx.barabasi_albert_graph(100, 2, seed=105)]
    return [Graph.from_networkx(g) for g in gs]


_HELDOUT = None


def colouring_check(expr: str) -> bool:
    global _HELDOUT
    if _HELDOUT is None:
        _HELDOUT = heldout_graphs()
    try:
        f = Formula("cand", expr)
        return all(verify_coloring(g, greedy_formula(g, f)) for g in _HELDOUT)
    except Exception:
        return False


@dataclass
class Ctx:
    model: str
    client: object
    val_dir: str
    test_dirs: dict              # name -> dir
    registry: str = "registry/candidates.jsonl"
    temperature: float = 0.7
    workers: int = 1


def query(ctx: Ctx, prompt: str, *, seed: int, experiment: str, arm: str, rnd: int,
          prompt_kind: str, expected: int, parent_id=None, shown=None) -> list[dict]:
    """Query, parse, validate; re-query up to MAX_REQUERIES times while any parsed
    candidate violates the constraint (or fewer than ``expected`` parse).
    All attempts are logged; the candidates of the accepted attempt are returned."""
    best = None
    for attempt in range(MAX_REQUERIES + 1):
        resp = ctx.client.complete(prompt, temperature=ctx.temperature, seed=seed * 100 + attempt)
        qid = new_id("q")
        raw_path = save_raw(qid, resp)
        cands = parse_response(resp.text, expected=expected)
        recs = []
        for c in cands:
            v = validate(c.raw_formula)
            admitted = v.ok and colouring_check(v.expr)
            recs.append({
                "candidate_id": new_id("c"), "query_id": qid, "parent_id": parent_id,
                "model": ctx.model, "model_snapshot": resp.model_snapshot,
                "queried_at_utc": resp.queried_at_utc, "temperature": resp.temperature,
                "seed": seed, "experiment": experiment, "arm": arm, "round": rnd,
                "prompt_kind": prompt_kind, "prompt_sha256": sha(prompt), "attempt": attempt,
                "shown_instances": shown or [], "response_sha256": sha(resp.text), "raw_path": raw_path,
                "index": c.index, "name": c.name, "raw_formula": c.raw_formula, "expr": v.expr,
                "rationale": c.rationale, "tie_break": c.tie_break, "strictly_linear": v.ok,
                "violations": [list(x) for x in v.violations], "features_used": list(v.features_used),
                "coefficients": v.coefficients, "admitted": admitted, "source": "llm",
            })
        n_ok = sum(r["admitted"] for r in recs)
        complete = len(recs) >= expected and n_ok == len(recs)
        for r in recs:
            r["n_requeries"] = attempt
            r["accepted_attempt"] = complete
        append(recs, ctx.registry)
        if best is None or n_ok > sum(r["admitted"] for r in best):
            best = recs
        if complete:
            return recs
    return best or []


def val_means(ctx: Ctx, recs: list[dict]) -> dict:
    heur = {r["candidate_id"]: ("formula", r["expr"]) for r in recs if r["admitted"]}
    if not heur:
        return {}
    long = evaluate(ctx.val_dir, heur, workers=ctx.workers)
    return long.groupby("algorithm")["colors"].mean().to_dict()


def select(recs: list[dict], means: dict) -> dict | None:
    """PROTOCOL.md section 2: min VAL mean; ties -> lower index (runtime tie-break
    is applied by experiments/e_b1_generate.py when exact ties occur)."""
    pool = [r for r in recs if r["candidate_id"] in means]
    if not pool:
        return None
    return min(pool, key=lambda r: (means[r["candidate_id"]], r.get("round", 0), r["index"]))


def test_means(ctx: Ctx, rec: dict) -> dict:
    out = {}
    for name, d in ctx.test_dirs.items():
        long = evaluate(d, {rec["candidate_id"]: ("formula", rec["expr"])}, workers=1)
        out[name] = float(long["colors"].mean())
    return out


# --------------------------------------------------------------- E-B1

def zero_shot(ctx: Ctx, seed: int) -> dict:
    recs = query(ctx, prompts.UNIVERSAL_PROMPT, seed=seed, experiment="E-B1", arm="zero-shot",
                 rnd=0, prompt_kind="universal", expected=5)
    means = val_means(ctx, recs)
    sel = select(recs, means)
    row = {"model": ctx.model, "seed": seed, "n_parsed": len(recs),
           "n_admitted": sum(r["admitted"] for r in recs),
           "n_requeries": recs[0]["n_requeries"] if recs else None,
           "rejected_violation": sum(not r["strictly_linear"] for r in recs),
           "selected": sel["candidate_id"] if sel else None,
           "selected_expr": sel["expr"] if sel else None,
           "val_mean": means.get(sel["candidate_id"]) if sel else None,
           "val_median_of_5": float(np.median(list(means.values()))) if means else None}
    if sel:
        row.update({f"{k}_mean": v for k, v in test_means(ctx, sel).items()})
    return row


# --------------------------------------------------------------- E-B3

def _dsatur_val(ctx: Ctx) -> pd.Series:
    long = evaluate(ctx.val_dir, {"DSATUR": ("classical", "DSATUR")})
    return long.set_index("instance_id")["colors"]


def refine(ctx: Ctx, seed: int, parent: dict, arm: str, rounds: int = 2) -> list[dict]:
    rng = random.Random(f"{ctx.model}-{seed}-{arm}")
    ds = _dsatur_val(ctx)
    rows = []
    for rnd in range(1, rounds + 1):
        par_long = evaluate(ctx.val_dir, {"parent": ("formula", parent["expr"])})
        pv = par_long.set_index("instance_id")
        gap = (pv["colors"] - ds.reindex(pv.index)).sort_index()
        shown = []
        if arm == "T":
            cand = gap[gap > 0].sort_values(ascending=False, kind="stable")
            shown = list(cand.index[:N_FAILURE_INSTANCES])
            if len(shown) < N_FAILURE_INSTANCES:            # parent never worse: show largest gaps
                rest = gap.drop(shown).sort_values(ascending=False, kind="stable")
                shown += list(rest.index[:N_FAILURE_INSTANCES - len(shown)])
        elif arm == "R":
            shown = rng.sample(sorted(gap.index), N_FAILURE_INSTANCES)
        if arm in ("T", "R"):
            meta = {r["instance_id"]: (r, g) for r, g in load_set(ctx.val_dir) if r["instance_id"] in shown}
            fails = "\n".join(describe_instance(meta[i][1], meta[i][0], int(pv.loc[i, "colors"]), int(ds[i]))
                              for i in shown)
            cells = {}
            for (fam, sc), grp in pv.groupby(["family", "scale"]):
                cells[(fam, sc)] = (grp["colors"].mean(), ds.reindex(grp.index).mean())
            prompt = prompts.REFINE_TARGETED.format(formula=parent["expr"],
                                                    successes=describe_successes(cells), failures=fails)
            expected = 2
        elif arm == "G":
            prompt = prompts.REFINE_GENERIC.format(formula=parent["expr"])
            expected = 2
        elif arm == "N":
            prompt = prompts.UNIVERSAL_PROMPT
            expected = 5
        else:
            raise ValueError(arm)
        recs = query(ctx, prompt, seed=seed * 10 + rnd, experiment="E-B3", arm=arm, rnd=rnd,
                     prompt_kind=f"refine-{arm}", expected=expected,
                     parent_id=parent["candidate_id"], shown=shown)
        means = val_means(ctx, recs)
        best_var = select(recs, means)
        parent_mean = float(pv["colors"].mean())
        row = {"model": ctx.model, "seed": seed, "arm": arm, "round": rnd,
               "parent_id": parent["candidate_id"], "parent_expr": parent["expr"],
               "parent_val_mean": parent_mean, "n_admitted": len(means), "shown": json.dumps(shown)}
        if best_var:
            row.update({"variant_id": best_var["candidate_id"], "variant_expr": best_var["expr"],
                        "variant_val_mean": means[best_var["candidate_id"]]})
            for k, v in test_means(ctx, best_var).items():
                row[f"variant_{k}_mean"] = v
            if means[best_var["candidate_id"]] < parent_mean:
                parent = best_var                            # selected for the next round
        for k, v in test_means(ctx, parent).items():
            row[f"selected_{k}_mean"] = v
        row["selected_id"] = parent["candidate_id"]
        rows.append(row)
    return rows


# --------------------------------------------------------------- E-H

def evolve(ctx: Ctx, seed: int, init: list[dict], generations: int = 5, children: int = 5,
           pop_size: int = 5) -> list[dict]:
    """EoH/ReEvo-style loop: show two tournament-selected parents with VAL fitness,
    ask for ``children`` new formulas, keep the best ``pop_size`` (VAL)."""
    rng = random.Random(f"evo-{ctx.model}-{seed}")
    means = val_means(ctx, init)
    pop = sorted([r for r in init if r["candidate_id"] in means],
                 key=lambda r: means[r["candidate_id"]])[:pop_size]
    rows = []
    for gen in range(1, generations + 1):
        if not pop:
            break
        parents = sorted(rng.sample(pop, min(2, len(pop))), key=lambda r: means[r["candidate_id"]])
        ptxt = "\n".join(f"- score(v) = {p['expr']}    (mean colors {means[p['candidate_id']]:.3f})"
                         for p in parents)
        recs = query(ctx, prompts.EVOLVE_PROMPT.format(parents=ptxt, k=children), seed=seed * 100 + gen,
                     experiment="E-H", arm="evolve", rnd=gen, prompt_kind="evolve", expected=children,
                     parent_id=",".join(p["candidate_id"] for p in parents))
        means.update(val_means(ctx, recs))
        pool = pop + [r for r in recs if r["candidate_id"] in means]
        pop = sorted(pool, key=lambda r: (means[r["candidate_id"]], r["round"], r["index"]))[:pop_size]
        best = pop[0]
        row = {"model": ctx.model, "seed": seed, "generation": gen, "best_id": best["candidate_id"],
               "best_expr": best["expr"], "best_val_mean": means[best["candidate_id"]]}
        row.update({f"best_{k}_mean": v for k, v in test_means(ctx, best).items()})
        rows.append(row)
    return rows
