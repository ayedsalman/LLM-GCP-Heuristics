"""Run the LLM experiments of the revision (needs API keys; see registry/models.json).

    # E-B1: 5-seed pilot, then the full run (PROTOCOL.md section 3)
    python experiments/run_llm.py zero-shot --model LLaMA --seeds 0-4
    python experiments/run_llm.py zero-shot --model LLaMA --seeds 0-19

    # E-B3: two refinement rounds per seed, all four arms, every model
    python experiments/run_llm.py refine --model LLaMA --seeds 0-19 --arms T,G,R,N

    # E-H (optional): budget-matched evolutionary arm, 5 seeds
    python experiments/run_llm.py evolve --model LLaMA --seeds 0-4

    # E-C: CUDA reliability audit (on a machine with nvcc and a GPU)
    python experiments/run_llm.py cuda-audit --model Claude --samples 20

    # dry run of the whole pipeline without API calls
    python experiments/run_llm.py zero-shot --model mock --seeds 0-1 --val data/VAL --tests TEST-A=data/TEST-A

Outputs: registry/candidates.jsonl (every candidate, with provenance),
runs/raw/*.json (raw responses), results/llm/*.csv (per-seed summaries).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from gcpbench.llm import pipeline, cuda_audit  # noqa: E402
from gcpbench.llm.clients import make_client  # noqa: E402
from gcpbench.llm.registry import load  # noqa: E402

OUT = "results/llm"


def seeds(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def append_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["zero-shot", "refine", "evolve", "cuda-audit"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--arms", default="T,G,R,N")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--val", default="data/VAL")
    ap.add_argument("--tests", default="TEST-A=data/TEST-A,TEST-B=data/TEST-B")
    ap.add_argument("--registry", default="registry/candidates.jsonl")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-io-contract", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    client = make_client(a.model)
    tests = dict(kv.split("=") for kv in a.tests.split(",") if kv)
    ctx = pipeline.Ctx(a.model, client, a.val, tests, registry=a.registry,
                       temperature=a.temperature, workers=a.workers)

    if a.command == "zero-shot":
        for s in seeds(a.seeds):
            row = pipeline.zero_shot(ctx, s)
            append_csv([row], os.path.join(a.out, "e_b1_zero_shot.csv"))
            print(json.dumps(row, default=str))

    elif a.command in ("refine", "evolve"):
        zs = pd.read_csv(os.path.join(a.out, "e_b1_zero_shot.csv"))
        zs = zs[zs["model"] == a.model]
        regs = {r["candidate_id"]: r for r in load(a.registry, model=a.model)}
        for s in seeds(a.seeds):
            sel = zs[zs["seed"] == s]
            if sel.empty or pd.isna(sel.iloc[-1]["selected"]):
                print(f"seed {s}: no zero-shot selection; run zero-shot first")
                continue
            parent = regs[sel.iloc[-1]["selected"]]
            if a.command == "refine":
                for arm in a.arms.split(","):
                    rows = pipeline.refine(ctx, s, parent, arm, rounds=a.rounds)
                    append_csv(rows, os.path.join(a.out, "e_b3_refine.csv"))
                    print(f"seed {s} arm {arm}: done")
            else:
                init = [r for r in regs.values() if r["experiment"] == "E-B1" and r["seed"] == s
                        and r["query_id"] == parent["query_id"]]
                rows = pipeline.evolve(ctx, s, init)
                append_csv(rows, os.path.join(a.out, "e_h_evolve.csv"))

    else:
        graphs = cuda_audit.audit_graphs()
        recs = []
        for i in range(a.samples):
            r = cuda_audit.audit_sample(client, a.model, i, graphs, io_contract=not a.no_io_contract)
            recs.append(r)
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "e_c_cuda_audit.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
            print(a.model, i, "T0" if r["T0_pass"] else r["T0_defect"], "T1" if r["T1_pass"] else "")
        append_csv(cuda_audit.summarise(recs), os.path.join(a.out, "e_c_cuda_summary.csv"))
        with open(os.path.join(a.out, f"T2_worksheet_{a.model}.md"), "w") as f:
            f.write(cuda_audit.worksheet(recs))


if __name__ == "__main__":
    main()
