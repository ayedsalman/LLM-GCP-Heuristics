"""Core tests: validator, exact equivalence with the original code, baselines, stats,
parallel emulator, parser, dataset reproducibility. Run: pytest -q"""

import json
import os
import random
import shutil
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src")]

from heuristics import sequential_heuristics as L  # noqa: E402
from gcpbench.classical import CLASSICAL  # noqa: E402
from gcpbench.formulas import paper_formulas, Formula  # noqa: E402
from gcpbench.graph import Graph, verify_coloring, n_colors  # noqa: E402
from gcpbench.greedy import greedy_formula, greedy_naive  # noqa: E402
from gcpbench.parallel import color_parallel, PRIORITIES, splitmix_unit, luby_mis_coloring  # noqa: E402
from gcpbench.stats import tost_paired, holm, critical_difference, paired_vs_baseline  # noqa: E402
from gcpbench.validator import validate  # noqa: E402
from gcpbench.llm.parse import parse_response, extract_code  # noqa: E402
from gcpbench.llm.cuda_audit import pass_at_k  # noqa: E402


def graphs(k=24, lo=20, hi=80):
    out = []
    for s in range(k):
        r = random.Random(s)
        n = r.randint(lo, hi)
        G = [nx.erdos_renyi_graph(n, r.choice([.1, .2, .4]), seed=s), nx.barabasi_albert_graph(n, r.choice([2, 4]), seed=s),
             nx.watts_strogatz_graph(n, 6, .1, seed=s)][s % 3]
        out.append(G)
    return out


# ------------------------------------------------------------------ validator
@pytest.mark.parametrize("text,ok,codes", [
    ("1.2·saturation(v) − 0.6·clustering_coefficient(v) + 0.1·uncolored_neighbors(v)", True, set()),
    ("0.6 sat + 0.4 deg·cc", False, {"PRODUCT"}),
    ("deg + rand()", False, {"RANDOM"}),
    ("10^6·sat + rand()·cc + deg", False, {"RANDOM", "PRODUCT"}),
    ("max(sat, deg)", False, {"MINMAX"}),
    ("sat if deg > 3 else ucn", False, {"CONDITIONAL"}),
    ("log(deg) + sat", False, {"NONLINEAR"}),
    ("foo + sat", False, {"UNKNOWN"}),
    ("score(v) = -ucn", True, set()),
    ("1e6*sat + 1200*ucn + 17*deg", True, set()),
    ("deg/2 + sat", True, set()),
    ("sat/deg", False, {"PRODUCT"}),
])
def test_validator(text, ok, codes):
    r = validate(text)
    assert r.ok == ok
    assert codes <= r.codes


def test_linear_coefficients():
    r = validate("2 sat + deg - 3 cc")
    assert r.coefficients == {"sat": 2.0, "deg": 1.0, "cc": -3.0}


def test_paper_linearity_flags():
    F = paper_formulas()
    assert not F["DeepSeek_H1"].strictly_linear and not F["DeepSeek_H1B"].strictly_linear
    assert all(F[n].strictly_linear for n in F if n not in ("DeepSeek_H1", "DeepSeek_H1B"))


# ---------------------------------------------------------------- equivalence
def test_incremental_equals_original_code():
    F = paper_formulas()
    for G in graphs():
        g = Graph.from_networkx(G)
        for name, f in F.items():
            ref = L.greedy_color(G, name)
            new = greedy_formula(g, f)
            assert [ref[lab] for lab in g.labels] == new, name


def test_incremental_equals_naive():
    F = paper_formulas()
    for G in graphs(9):
        g = Graph.from_networkx(G)
        for f in F.values():
            assert greedy_naive(g, f.fn) == greedy_formula(g, f)


def test_classical_valid_and_legacy_rlf_matches():
    for G in graphs(12):
        g = Graph.from_networkx(nx.convert_node_labels_to_integers(G))
        for k, fn in CLASSICAL.items():
            assert verify_coloring(g, fn(g)), k
        assert n_colors(CLASSICAL["RLF_legacy"](g)) == L.num_colors_used(L.rlf_coloring(G))


def test_rlf_fix_on_planted():
    from gcpbench.datasets import planted_graph
    G, _ = planted_graph(400, 8, 20, seed=3)
    g = Graph.from_networkx(G)
    assert n_colors(CLASSICAL["RLF"](g)) <= n_colors(CLASSICAL["RLF_legacy"](g))


def test_numba_engine_close_to_exact():
    from gcpbench.fastlinear import LinearEvaluator
    gs = [Graph.from_networkx(G) for G in graphs(9, 60, 200)]
    ev = LinearEvaluator(gs)
    f = paper_formulas()["LLaMA_H3"]
    exact = np.array([n_colors(greedy_formula(g, f)) for g in gs])
    fast = ev.colors(f.coefficient_vector())
    assert (exact == fast).mean() >= 0.8


# ---------------------------------------------------------------------- stats
def test_cd_matches_manuscript():
    for k, n, cd in [(8, 50, 1.485), (10, 30, 2.47), (4, 50, 0.66), (4, 30, 0.86)]:
        assert abs(critical_difference(k, n) - cd) < 0.006


def test_tost_and_holm():
    assert tost_paired(np.zeros(40)).equivalent()
    assert not tost_paired(np.r_[np.ones(20), np.zeros(20)]).equivalent()
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_table():
    d = pd.DataFrame({"DSATUR": [4, 5, 6, 7] * 10, "X": [4, 5, 6, 7] * 10, "Y": [5, 6, 7, 8] * 10})
    t = paired_vs_baseline(d, "DSATUR").set_index("algorithm")
    assert t.loc["X", "equivalent"] and t.loc["Y", "verdict"] == "worse"


# ------------------------------------------------------------------- parallel
def test_splitmix_known_value():
    # splitmix64(0 ^ 0) = 0xE220A8397B1DCDAF
    assert splitmix_unit(0, 1)[0] == (0xE220A8397B1DCDAF >> 11) * 2.0 ** -53


def test_parallel_valid_and_progress():
    for G in graphs(6):
        g = Graph.from_networkx(G)
        for p in PRIORITIES:
            c, R = color_parallel(g, p, seed=1)
            assert verify_coloring(g, c) and 1 <= R <= g.n
        c, R = luby_mis_coloring(g, 0)
        assert verify_coloring(g, c)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_cuda_source_host_build_matches_emulator(tmp_path):
    subprocess.check_call([sys.executable, os.path.join(ROOT, "tests", "cuda_host", "build_host.py")])
    binp = os.path.join(ROOT, "tests", "cuda_host", "pc_host")
    G = nx.barabasi_albert_graph(120, 3, seed=5)
    path = tmp_path / "g.edgelist"
    nx.write_edgelist(G, path, data=False)
    g = Graph.from_networkx(nx.read_edgelist(path, nodetype=int))
    for p in ("GPT51", "Claude", "DeepSeek", "neg_sat"):
        out = subprocess.run([binp, str(path), "--priority", p, "--seed", "3", "--reps", "1", "--warmup", "0",
                              "--dump", str(tmp_path / "c.txt")], capture_output=True, text=True, errors="replace")
        gpu = [int(x) for x in open(tmp_path / "c.txt").read().split()]
        cpu, R = color_parallel(g, p, seed=3)
        assert gpu == cpu and int(out.stdout.split(",")[6]) == R


# ---------------------------------------------------------------------- parse
def test_parse_response():
    text = """### Heuristic 1: Saturation-led
**Formula:** score(v) = 1.2·saturation(v) − 0.6·clustering_coefficient(v) + 0.1·uncolored_neighbors(v)
**Design intuition:** saturation dominates.
**Tie-breaking:** higher degree(v), then lower vertex id.

### Heuristic 2: Degree-led
score(v) = 0.5 * degree(v) + 2 * saturation(v)
Intuition: hubs first.
Tie-break: lowest id.
"""
    c = parse_response(text)
    assert [x.expr for x in c] == ["1.2*sat - 0.6*cc + 0.1*ucn", "0.5 * deg + 2 * sat"]
    assert "saturation" in c[0].rationale and "Tie" in c[0].tie_break
    assert extract_code("x\n```cuda\nint main(){}\n```\n") == "int main(){}\n"


def test_pass_at_k():
    assert pass_at_k(20, 0, 1) == 0 and pass_at_k(20, 20, 3) == 1
    assert pass_at_k(20, 5, 1) == pytest.approx(0.25)


# ------------------------------------------------------------------- datasets
def test_val_reproduces_lock(tmp_path):
    from gcpbench import datasets
    with open(os.path.join(ROOT, "registry", "datasets.lock.json")) as f:
        lock = json.load(f)
    rows = datasets.generate("VAL", str(tmp_path), legacy_edgelist=True)
    got = {r["instance_id"]: r["sha256"] for r in rows if r["scale"] == "small"}
    for k, v in got.items():
        assert lock["sets"]["VAL"]["instances"][k] == v
