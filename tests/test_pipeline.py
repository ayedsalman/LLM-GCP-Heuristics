"""End-to-end dry run of the LLM pipeline with the mock client (no API calls)."""

import csv
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src")]

from gcpbench import datasets  # noqa: E402
from gcpbench.llm import pipeline, cuda_audit  # noqa: E402
from gcpbench.llm.clients import MockClient  # noqa: E402


def _subset(src, dst, k):
    os.makedirs(dst)
    rows = [r for r in csv.DictReader(open(os.path.join(src, "manifest.csv"))) if r["scale"] == "small"][:k]
    for r in rows:
        shutil.copy(os.path.join(src, r["instance_id"]), dst)
    with open(os.path.join(dst, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def test_pipeline_mock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("VAL", "TEST-A"):
        d = os.path.join(ROOT, "data", name)
        if not os.path.exists(os.path.join(d, "ER_small_000.edgelist")):
            datasets.generate(name, d, legacy_edgelist=True)
    _subset(os.path.join(ROOT, "data", "VAL"), "VAL", 6)
    _subset(os.path.join(ROOT, "data", "TEST-A"), "TA", 4)
    ctx = pipeline.Ctx("mock", MockClient("mock", p_violation=0.3), "VAL", {"TEST-A": "TA"},
                       registry="reg.jsonl")
    row = pipeline.zero_shot(ctx, 0)
    assert row["selected"] and row["n_admitted"] >= 1 and "TEST-A_mean" in row
    recs = [json.loads(l) for l in open("reg.jsonl")]
    parent = next(r for r in recs if r["candidate_id"] == row["selected"])
    for arm in "TGRN":
        rows = pipeline.refine(ctx, 0, parent, arm, rounds=2)
        assert len(rows) == 2 and all("selected_TEST-A_mean" in r for r in rows)
        if arm == "T":
            shown = json.loads(rows[0]["shown"])
            assert len(shown) == pipeline.N_FAILURE_INSTANCES - 1 or len(shown) <= 6
            val_ids = {r["instance_id"] for r in csv.DictReader(open("VAL/manifest.csv"))}
            assert set(shown) <= val_ids           # failure instances only from VAL (A8)
    evo = pipeline.evolve(ctx, 0, [r for r in recs if r["query_id"] == parent["query_id"]], generations=2)
    assert len(evo) == 2
    recs = [json.loads(l) for l in open("reg.jsonl")]
    assert all({"model_snapshot", "prompt_sha256", "raw_path", "strictly_linear"} <= set(r) for r in recs)


def test_cuda_audit_toolchain_missing(monkeypatch):
    monkeypatch.setattr(cuda_audit, "COMPILE", ["/nonexistent/nvcc", "{src}", "-o", "{bin}"])
    r = cuda_audit.audit_sample(MockClient("m", cuda_ok=1.0), "m", 0, cuda_audit.audit_graphs(1))
    assert r["T0_defect"] == "toolchain_missing"
    s = cuda_audit.summarise([r])
    assert s[0]["n"] == 1
