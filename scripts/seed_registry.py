"""Seed registry/candidates.jsonl with the 12 sequential and 4 parallel formulas
printed in Table 6 of the submitted manuscript (source = "paper-table6").

The other 25 sequential candidates of the original study are not printed in
the paper. Add them from the original raw responses with
`python scripts/seed_registry.py --add MODEL TRIAL NAME "FORMULA"` so that
experiment E-A can evaluate all 37 candidates (plan A4).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gcpbench.formulas import PAPER_SEQUENTIAL, PAPER_PARALLEL  # noqa: E402
from gcpbench.validator import validate  # noqa: E402
from gcpbench.llm.registry import append, load, REGISTRY  # noqa: E402


def record(name, model, trial, expr, parallel=False, best=None):
    v = validate(expr, allow=("RANDOM",) if parallel else ())
    strict = validate(expr)
    return {"candidate_id": f"paper-{name}", "source": "paper-table6", "model": model,
            "experiment": "original-study", "arm": "parallel" if parallel else trial, "round": 0,
            "name": name, "raw_formula": expr, "expr": v.expr, "strictly_linear": strict.ok,
            "violations": [list(x) for x in strict.violations], "features_used": list(v.features_used),
            "coefficients": strict.coefficients, "admitted": True, "paper_best_T1": best,
            "model_snapshot": None, "queried_at_utc": None, "temperature": 0.7, "seed": None,
            "rationale": None, "raw_path": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs=4, metavar=("MODEL", "TRIAL", "NAME", "FORMULA"))
    a = ap.parse_args()
    have = {r["candidate_id"] for r in load(REGISTRY)}
    if a.add:
        m, t, n, f = a.add
        append([record(n, m, t, f)])
        print("added", n)
        return
    recs = [record(n, m, t, e, best=b) for n, (e, m, t, b) in PAPER_SEQUENTIAL.items()]
    par_model = {"GPT4o_par": "GPT-4o", "GPT51_par": "GPT-5.1", "Claude_par": "Claude",
                 "DeepSeek_par": "DeepSeek"}
    recs += [record(n, par_model[n], "parallel", e, parallel=True) for n, e in PAPER_PARALLEL.items()]
    recs = [r for r in recs if r["candidate_id"] not in have]
    append(recs)
    print(f"seeded {len(recs)} records into {REGISTRY}")


if __name__ == "__main__":
    main()
