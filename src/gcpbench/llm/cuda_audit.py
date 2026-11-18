"""CUDA generation reliability audit (plan E-C, C2, C3).

Tiers, never mixed in one results table:
  T0  unassisted: the first response, compiled and run as returned;
  T1  LLM self-repair: compiler/runtime/checker output fed back, up to 3 turns,
      no human edits;
  T2  human-corrected: only for samples still failing after T1. Done outside
      this script; every fix is recorded in docs/INTERVENTION_LOG.md with its
      diff, lines changed and time spent. ``worksheet`` lists what needs it.

Defect taxonomy: compile_error, runtime_error, invalid_coloring,
non_termination, memory_leak, performance_pathology (> 10x the model's median
kernel time, assigned post hoc), constraint_violation (non-linear, rand(),
unknown feature in the declared formula), no_code, bad_output.

Metrics: pass@1 and pass@3 per tier with the unbiased estimator
1 - C(n-c, k) / C(n, k) (Chen et al., 2021), n = samples per model.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import networkx as nx

from ..graph import Graph, verify_coloring
from ..validator import validate
from . import prompts
from .parse import extract_code, parse_response
from .registry import new_id, sha

NVCC = os.environ.get("NVCC", "nvcc")
COMPILE = [NVCC, "-O3", "-std=c++17", "{src}", "-o", "{bin}"]
SANITIZER = shutil.which("compute-sanitizer")


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def audit_graphs(per_family: int = 5):
    """Held-out verification graphs: 5 per family, not in any evaluation set."""
    out = []
    for i in range(per_family):
        out.append(("ER", nx.erdos_renyi_graph(300 + 50 * i, 0.05, seed=900 + i)))
        out.append(("BA", nx.barabasi_albert_graph(300 + 50 * i, 4, seed=910 + i)))
        out.append(("WS", nx.watts_strogatz_graph(300 + 50 * i, 10, 0.1, seed=920 + i)))
    return [(fam, Graph.from_networkx(g)) for fam, g in out]


@dataclass
class Outcome:
    ok: bool
    category: str | None = None
    details: str = ""
    kernel_ms: list = field(default_factory=list)
    colors: list = field(default_factory=list)


def compile_and_test(code: str, graphs, workdir: str, timeout: int = 60,
                     leak_check: bool = False) -> Outcome:
    src = os.path.join(workdir, "prog.cu")
    binp = os.path.join(workdir, "prog")
    with open(src, "w") as f:
        f.write(code)
    cmd = [a.format(src=src, bin=binp) for a in COMPILE]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return Outcome(False, "toolchain_missing", f"{NVCC} not found")
    if r.returncode != 0:
        return Outcome(False, "compile_error", (r.stderr or r.stdout)[-3000:])
    out = Outcome(True)
    for fam, g in graphs:
        path = os.path.join(workdir, "g.txt")
        with open(path, "w") as f:
            for u, v in g.edges():
                f.write(f"{u} {v}\n")
        run = [binp, path]
        if leak_check and SANITIZER:
            run = [SANITIZER, "--leak-check", "full", "--error-exitcode", "97"] + run
        try:
            p = subprocess.run(run, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Outcome(False, "non_termination", f"timeout {timeout}s on {fam} n={g.n}")
        if p.returncode == 97:
            return Outcome(False, "memory_leak", p.stdout[-2000:] + p.stderr[-2000:])
        if p.returncode != 0:
            return Outcome(False, "runtime_error", (p.stderr or p.stdout)[-3000:])
        lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
        ms = [float(l.split()[1]) for l in lines if l.startswith("KERNEL_MS")]
        try:
            cols = [int(l) for l in lines if not l.startswith("KERNEL_MS")]
        except ValueError:
            return Outcome(False, "bad_output", "\n".join(lines[:20]))
        if len(cols) != g.n:
            return Outcome(False, "bad_output", f"expected {g.n} colour lines, got {len(cols)}")
        if not verify_coloring(g, cols):
            return Outcome(False, "invalid_coloring", f"conflicting edge on {fam} n={g.n}")
        out.kernel_ms += ms
        out.colors.append(len(set(cols)))
    return out


def audit_sample(client, model: str, sample: int, graphs, max_repair: int = 3,
                 io_contract: bool = True, leak_check: bool = True) -> dict:
    prompt = prompts.CUDA_PROMPT + (prompts.CUDA_IO_CONTRACT if io_contract else "")
    messages = [{"role": "user", "content": prompt}]
    resp = client.complete(messages, temperature=0.7, seed=sample)
    rec = {"sample_id": new_id("cu"), "model": model, "sample": sample,
           "model_snapshot": resp.model_snapshot, "queried_at_utc": resp.queried_at_utc,
           "prompt_sha256": sha(prompt), "io_contract": io_contract, "turns": []}
    cands = parse_response(resp.text, expected=5)
    formula = cands[0].raw_formula if cands else None
    v = validate(formula) if formula else None
    rec["declared_formula"] = formula
    rec["constraint_violation"] = (not v.ok) if v else None
    rec["violations"] = [list(x) for x in v.violations] if v else []
    text = resp.text
    with tempfile.TemporaryDirectory() as wd:
        for turn in range(max_repair + 1):
            code = extract_code(text)
            if code is None:
                o = Outcome(False, "no_code", text[-1000:])
            else:
                o = compile_and_test(code, graphs, wd, leak_check=leak_check)
            rec["turns"].append({"turn": turn, "ok": o.ok, "category": o.category,
                                 "details": o.details[-1500:], "code_sha256": sha(code or ""),
                                 "code": code, "kernel_ms": o.kernel_ms, "colors": o.colors})
            if o.ok or o.category == "toolchain_missing" or turn == max_repair:
                break
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": prompts.CUDA_REPAIR.format(
                                       category=o.category, details=o.details[-2500:])}]
            text = client.complete(messages, temperature=0.7, seed=sample * 10 + turn + 1).text
    t = rec["turns"]
    rec["T0_pass"] = t[0]["ok"]
    rec["T1_pass"] = any(x["ok"] for x in t)
    rec["T1_turns_needed"] = next((x["turn"] for x in t if x["ok"]), None)
    rec["needs_T2"] = not rec["T1_pass"]
    rec["T0_defect"] = t[0]["category"]
    return rec


def summarise(records: list[dict]) -> list[dict]:
    by = {}
    for r in records:
        by.setdefault(r["model"], []).append(r)
    rows = []
    for m, rs in by.items():
        n = len(rs)
        c0 = sum(r["T0_pass"] for r in rs)
        c1 = sum(r["T1_pass"] for r in rs)
        med = sorted(ms for r in rs for t in r["turns"] if t["ok"] for ms in t["kernel_ms"])
        median = med[len(med) // 2] if med else None
        patho = sum(1 for r in rs for t in r["turns"][:1]
                    if t["ok"] and median and t["kernel_ms"] and max(t["kernel_ms"]) > 10 * median)
        defects = {}
        for r in rs:
            defects[r["T0_defect"] or "none"] = defects.get(r["T0_defect"] or "none", 0) + 1
        rows.append({"model": m, "n": n, "T0_pass": c0, "T1_pass": c1,
                     "T0_pass@1": pass_at_k(n, c0, 1), "T0_pass@3": pass_at_k(n, c0, 3),
                     "T1_pass@1": pass_at_k(n, c1, 1), "T1_pass@3": pass_at_k(n, c1, 3),
                     "constraint_violation": sum(bool(r["constraint_violation"]) for r in rs),
                     "performance_pathology_T0": patho, "needs_T2": n - c1,
                     "T0_defects": json.dumps(defects, sort_keys=True)})
    return rows


def worksheet(records: list[dict]) -> str:
    lines = ["# T2 worksheet: samples that still fail after LLM self-repair", "",
             "Record every human fix in docs/INTERVENTION_LOG.md (diff, lines changed, minutes).", ""]
    for r in records:
        if r["needs_T2"]:
            last = r["turns"][-1]
            lines.append(f"- {r['model']} sample {r['sample']} ({r['sample_id']}): last defect "
                         f"{last['category']}; code sha {last['code_sha256'][:12]}")
    return "\n".join(lines)
