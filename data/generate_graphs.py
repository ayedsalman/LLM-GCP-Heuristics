"""
generate_graphs.py -- benchmark sets for the revised study (plan S0.2, A2, D2, D6).

Backward compatible with the submitted version:

    python data/generate_graphs.py --out-dir data/graphs --seed 42

still produces the original 140-instance suite (Table 4) byte-for-byte,
now relabelled VAL, plus a manifest with per-instance seeds and SHA-256.

New sets (see src/gcpbench/datasets.py for the design):

    python data/generate_graphs.py --set VAL     --out-dir data/VAL
    python data/generate_graphs.py --set TEST-A  --out-dir data/TEST-A
    python data/generate_graphs.py --set TEST-B  --out-dir data/TEST-B
    python data/generate_graphs.py --set PLANTED --out-dir data/PLANTED

    # register the paper's ORIGINAL instance files as VAL (preferred if available)
    python data/generate_graphs.py --register-dir path/to/original_graphs --set VAL

    # lock / verify hashes (run `lock` once, commit registry/datasets.lock.json,
    # tag the commit, and only then evaluate any heuristic on TEST-A/TEST-B/EXT)
    python data/generate_graphs.py --lock
    python data/generate_graphs.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gcpbench import datasets  # noqa: E402

DEFAULT_DIRS = {"VAL": "data/VAL", "TEST-A": "data/TEST-A", "TEST-B": "data/TEST-B",
                "PLANTED": "data/PLANTED", "DIMACS": "data/dimacs"}
LOCK = "registry/datasets.lock.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="VAL", choices=["VAL", "TEST-A", "TEST-B", "PLANTED", "DIMACS"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None, help="master seed (default: per-set constant)")
    ap.add_argument("--register-dir", default=None, help="register existing files as --set")
    ap.add_argument("--all", action="store_true", help="generate VAL, TEST-A, TEST-B, PLANTED")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.lock or args.verify:
        dirs = {k: v for k, v in DEFAULT_DIRS.items() if os.path.exists(os.path.join(v, "manifest.csv"))}
        if args.lock:
            out = datasets.lock(dirs, LOCK)
            for k, v in out["sets"].items():
                print(f"locked {k}: {v['n_instances']} instances, sha256 {v['set_sha256'][:16]}")
            print(f"wrote {LOCK}; commit it and tag the commit before any TEST evaluation.")
        else:
            problems = datasets.verify(dirs, LOCK)
            print("\n".join(problems) if problems else f"all sets match {LOCK}")
            sys.exit(1 if problems else 0)
        return

    sets = ["VAL", "TEST-A", "TEST-B", "PLANTED"] if args.all else [args.set]
    for s in sets:
        out_dir = (args.out_dir if (args.out_dir and not args.all) else DEFAULT_DIRS[s])
        if args.register_dir:
            rows = datasets.register_dir(s, args.register_dir)
            print(f"registered {len(rows)} files in {args.register_dir} as {s}")
            continue
        if s == "DIMACS":
            rows = datasets.register_dimacs(out_dir)
            print(f"registered {len(rows)} DIMACS instances in {out_dir}")
            continue
        legacy = s in ("VAL", "TEST-A")       # same file format/pipeline as the paper
        rows = datasets.generate(s, out_dir, master_seed=args.seed, legacy_edgelist=legacy)
        print(f"{s}: generated {len(rows)} instances into {out_dir}")
    if "TEST-A" in sets and os.path.exists(os.path.join(DEFAULT_DIRS["VAL"], "manifest.csv")):
        overlap = datasets.check_disjoint(DEFAULT_DIRS["VAL"], DEFAULT_DIRS["TEST-A"])
        print(f"VAL/TEST-A shared instances (by hash): {overlap}")
        if overlap:
            sys.exit("TEST-A is not disjoint from VAL")


if __name__ == "__main__":
    main()
