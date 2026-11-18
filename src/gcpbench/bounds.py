"""Chromatic-number bounds (plan D3, D4, E-D2).

exact_chi     CP-SAT model (OR-tools) seeded with a maximum clique (fixed to
              distinct colours, symmetry breaking) and the DSATUR upper bound.
              Returns (lb, ub, status); status OPTIMAL means lb == ub == chi.
clique_lb     maximum clique size (networkx.find_cliques with a time guard;
              falls back to the best clique found so far).
tabucol_ub    TabuCol (Hertz & de Werra 1987; Galinier & Hao 1999 tenure
              0.6*f + U{0..9}) driven downwards from the DSATUR colouring,
              numba-compiled, wall-clock budget per instance.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from .classical import dsatur_lex
from .graph import Graph, n_colors, verify_coloring

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


def clique_lb(g: Graph, time_limit: float = 30.0):
    G = g.to_networkx() if g._nx is None else g._nx
    best = []
    t0 = time.time()
    for c in nx.find_cliques(G):
        if len(c) > len(best):
            best = c
        if time.time() - t0 > time_limit:
            return len(best), best, False
    return len(best), best, True


def exact_chi(g: Graph, time_limit: float = 60.0, workers: int = 1):
    from ortools.sat.python import cp_model
    ub_col = dsatur_lex(g)
    ub = n_colors(ub_col)
    lb, clique, _ = clique_lb(g, time_limit=min(10.0, time_limit))
    labels = {lab: i for i, lab in enumerate(g.labels)} if g._nx is not None else None
    if labels:
        clique = [labels[x] for x in clique]
    if lb >= ub:
        return lb, ub, "OPTIMAL", ub_col
    m = cp_model.CpModel()
    K = ub
    x = [[m.NewBoolVar(f"x{v}_{c}") for c in range(K)] for v in range(g.n)]
    y = [m.NewBoolVar(f"y{c}") for c in range(K)]
    for v in range(g.n):
        m.AddExactlyOne(x[v])
    for u, v in g.edges():
        for c in range(K):
            m.AddBoolOr([x[u][c].Not(), x[v][c].Not()])
    for v in range(g.n):
        for c in range(K):
            m.AddImplication(x[v][c], y[c])
    for c in range(K - 1):
        m.AddImplication(y[c + 1], y[c])
    for i, v in enumerate(clique):          # symmetry breaking with the clique
        m.Add(x[v][i] == 1)
    m.Add(sum(y) >= lb)
    for v in range(g.n):                    # warm start from the DSATUR colouring
        m.AddHint(x[v][ub_col[v]], 1)
    m.Minimize(sum(y))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    st = s.Solve(m)
    if st == cp_model.OPTIMAL:
        cols = [next(c for c in range(K) if s.Value(x[v][c])) for v in range(g.n)]
        k = int(round(s.ObjectiveValue()))
        return k, k, "OPTIMAL", cols
    if st == cp_model.FEASIBLE:
        return (max(lb, int(np.ceil(s.BestObjectiveBound() - 1e-9))),
                min(ub, int(round(s.ObjectiveValue()))), "FEASIBLE", None)
    return lb, ub, "UNKNOWN", ub_col


@njit(cache=True)
def _tabucol(indptr, indices, k, col, max_iter, seed):
    np.random.seed(seed)
    n = indptr.shape[0] - 1
    gamma = np.zeros((n, k), np.int64)
    for v in range(n):
        for p in range(indptr[v], indptr[v + 1]):
            gamma[v, col[indices[p]]] += 1
    f = 0
    for v in range(n):
        f += gamma[v, col[v]]
    f //= 2
    tabu = np.zeros((n, k), np.int64)
    best_f = f
    best = col.copy()
    # conflicting-vertex set with O(1) add/remove
    pos = -np.ones(n, np.int64)
    conf = np.empty(n, np.int64)
    nc = 0
    for v in range(n):
        if gamma[v, col[v]] > 0:
            pos[v] = nc
            conf[nc] = v
            nc += 1
    it = 0
    while it < max_iter and f > 0:
        it += 1
        bd = 1 << 60
        bv = -1
        bc = -1
        cnt = 0
        for i in range(nc):
            v = conf[i]
            cv = col[v]
            gv = gamma[v, cv]
            for c in range(k):
                if c == cv:
                    continue
                d = gamma[v, c] - gv
                if tabu[v, c] > it and f + d >= best_f:
                    continue
                if d < bd:
                    bd = d
                    bv = v
                    bc = c
                    cnt = 1
                elif d == bd:
                    cnt += 1
                    if np.random.randint(cnt) == 0:
                        bv = v
                        bc = c
        if bv < 0:
            continue
        old = col[bv]
        col[bv] = bc
        f += bd
        tabu[bv, old] = it + int(0.6 * nc) + np.random.randint(10)
        for p in range(indptr[bv], indptr[bv + 1]):
            u = indices[p]
            gamma[u, old] -= 1
            gamma[u, bc] += 1
        for p in range(indptr[bv], indptr[bv + 1] + 1):
            u = bv if p == indptr[bv + 1] else indices[p]
            isc = gamma[u, col[u]] > 0
            if isc and pos[u] < 0:
                pos[u] = nc
                conf[nc] = u
                nc += 1
            elif (not isc) and pos[u] >= 0:
                j = pos[u]
                last = conf[nc - 1]
                conf[j] = last
                pos[last] = j
                pos[u] = -1
                nc -= 1
        if f < best_f:
            best_f = f
            best[:] = col
    return best_f, best


def tabucol_ub(g: Graph, time_limit: float = 60.0, seed: int = 0, start_colors=None,
               chunk: int = 20000):
    """Find successively smaller k-colourings within the time budget."""
    indptr, indices = g.csr()
    cur = np.asarray(start_colors if start_colors is not None else dsatur_lex(g), np.int64)
    best_k = int(cur.max()) + 1
    best_cols = cur.copy()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    k = best_k - 1
    while k >= 1 and time.time() - t0 < time_limit:
        col = best_cols.copy()
        # merge the highest colour class away (random reassignment), then repair
        mask = col >= k
        col[mask] = rng.integers(0, k, mask.sum())
        f = 1
        s = int(rng.integers(1 << 30))
        while time.time() - t0 < time_limit:
            f, col = _tabucol(indptr, indices, k, col, chunk, s)
            s += 1
            if f == 0:
                break
        if f == 0:
            assert verify_coloring(g, col.tolist())
            best_k, best_cols = k, col.copy()
            k -= 1
        else:
            break
    return best_k, best_cols
