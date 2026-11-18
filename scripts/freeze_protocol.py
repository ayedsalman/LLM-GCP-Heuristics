"""Write registry/protocol_freeze.json: hashes of PROTOCOL.md and the dataset lock,
plus the machine-readable constants of the protocol (plan S0.5).

Run once, commit, and tag (e.g. `git tag protocol-freeze-v1`) before any TEST
evaluation. `--check` exits non-zero if PROTOCOL.md or the lock changed since.
"""
import argparse
import datetime
import hashlib
import json
import sys

FREEZE = "registry/protocol_freeze.json"
CONSTANTS = {
    "selection_rule": "min VAL mean colours; ties: min VAL median runtime; ties: lower candidate index",
    "equivalence_margin_colours": 0.10,
    "alpha": 0.05,
    "bootstrap_resamples": 10000,
    "multiplicity": "Holm across heuristic-vs-DSATUR comparisons of a table",
    "generation_seeds_per_model": 20,
    "min_generation_seeds": 10,
    "candidates_per_query": 5,
    "temperature": 0.7,
    "max_requeries_on_violation": 3,
    "refinement_rounds": 2,
    "refinement_arms": ["T", "G", "R", "N"],
    "random_control_draws": 20,
    "cmaes_restarts": 20,
    "cmaes_evals_per_restart": 500,
    "tie_break": "lowest vertex id",
    "timing_reps": 30,
    "timing_warmup": 3,
    "parallel_rand": "splitmix64(seed ^ v) -> [0,1), fixed per run; seeds 0..9",
    "baseline": "DSATUR = sat*10000 + deg (submitted paper)",
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cur = {"PROTOCOL.md": sha("PROTOCOL.md"), "datasets.lock.json": sha("registry/datasets.lock.json")}
    if a.check:
        with open(FREEZE) as f:
            frozen = json.load(f)
        bad = [k for k in cur if frozen["hashes"][k] != cur[k]]
        print("protocol unchanged" if not bad else f"CHANGED since freeze: {bad}")
        sys.exit(1 if bad else 0)
    out = {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "hashes": cur, "constants": CONSTANTS}
    with open(FREEZE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {FREEZE}")


if __name__ == "__main__":
    main()
