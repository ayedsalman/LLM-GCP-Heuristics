"""Exact CPU emulator of the CUDA local-maxima colouring kernel (plan E-F, F1-F3, E7, C5).

Algorithm 2 of the revised manuscript. Each round:
  1. every uncoloured v computes score(v) from sat, deg, ucn, cc, twohop and,
     optionally, r(v) (all in float64, same operation order as the kernel);
  2. v is a local maximum if, for every uncoloured neighbour u,
     score(u) < score(v), or score(u) == score(v) and u > v (lowest id wins);
  3. all local maxima take the smallest colour unused by their already
     coloured neighbours (they are pairwise non-adjacent, so this is safe).
Rounds repeat until every vertex is coloured; the round count R is returned.

r(v) is the declared randomised term (C5): splitmix64(seed ^ v) mapped to
[0, 1), fixed for the whole run (the Jones-Plassmann convention), identical in
src/cuda/parallel_coloring.cu. The emulator therefore reproduces the kernel's
colourings exactly and lets the priority ablation of E-F run without a GPU.

PRIORITIES lists the ablation set of E-F: (a) random, (b) degree, (c) -ucn,
(d) sat only, (e) 1e6*sat + deg, (f) the four LLM CUDA formulas exactly as
reported in Table 6, (g) anti-priority controls -deg and -sat.
"""

from __future__ import annotations

import numpy as np

from .graph import Graph

MASK64 = (1 << 64) - 1


def splitmix_unit(seed: int, n: int) -> np.ndarray:
    """r(v) = splitmix64(seed ^ v) >> 11, times 2^-53, for v = 0..n-1 (exact uint64 math)."""
    v = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (np.uint64(seed & MASK64) ^ v) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


# score(sat, deg, ucn, cc, th, r) -- float64 arrays. Keep the operation order
# identical to device_score() in parallel_coloring.cu.
PRIORITIES = {
    "random":      lambda s, d, u, c, t, r: r,
    "degree":      lambda s, d, u, c, t, r: d,
    "neg_ucn":     lambda s, d, u, c, t, r: -u,
    "sat":         lambda s, d, u, c, t, r: s,
    "sat1e6_deg":  lambda s, d, u, c, t, r: 1e6 * s + d,
    "GPT4o":       lambda s, d, u, c, t, r: 1e6 * s + r * c + d,
    "GPT51":       lambda s, d, u, c, t, r: 1e6 * s + 1200.0 * u + 17.0 * d,
    "Claude":      lambda s, d, u, c, t, r: -u,
    "DeepSeek":    lambda s, d, u, c, t, r: d + r,
    "neg_deg":     lambda s, d, u, c, t, r: -d,
    "neg_sat":     lambda s, d, u, c, t, r: -s,
}
PRIORITY_IDS = {name: i for i, name in enumerate(PRIORITIES)}   # must match the .cu enum
LLM_PRIORITIES = ("GPT4o", "GPT51", "Claude", "DeepSeek")


def linear_priority(w, w_rand: float = 0.0):
    """Generic priority ((((w0*s + w1*d) + w2*u) + w3*c) + w4*t) + wr*r."""
    a, b, c_, d_, e = (float(x) for x in w)
    return lambda s, d, u, c, t, r: a * s + b * d + c_ * u + d_ * c + e * t + w_rand * r


def _smallest_free(vs: np.ndarray, cs: np.ndarray, targets: np.ndarray, n: int) -> np.ndarray:
    """For each target vertex, the smallest colour not in {cs[i] : vs[i] == target}."""
    out = np.zeros(n, dtype=np.int64)
    if len(vs):
        key = np.unique(vs.astype(np.int64) * (n + 1) + cs)
        v = key // (n + 1)
        c = key % (n + 1)
        start = np.r_[0, np.flatnonzero(np.diff(v)) + 1]
        grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(v)]))
        pos = np.arange(len(v)) - start[grp]
        gap = c != pos
        first_gap = np.full(len(start), -1, dtype=np.int64)
        gi = np.flatnonzero(gap)
        # first gap per group
        g_of_gap = grp[gi]
        uniq, idx = np.unique(g_of_gap, return_index=True)
        first_gap[uniq] = pos[gi[idx]]
        sizes = np.diff(np.r_[start, len(v)])
        val = np.where(first_gap >= 0, first_gap, sizes)
        out[v[start]] = val
    res = np.zeros(n, dtype=np.int64)
    res[targets] = out[targets]
    return res


def color_parallel(g: Graph, priority="sat1e6_deg", seed: int = 0, return_rounds: bool = True,
                   max_rounds: int | None = None):
    """Run the local-maxima kernel semantics. Returns (colors, rounds)."""
    fn = PRIORITIES[priority] if isinstance(priority, str) else priority
    n = g.n
    indptr, indices = g.csr()
    deg_i = np.diff(indptr)
    src = np.repeat(np.arange(n, dtype=np.int64), deg_i)
    dst = indices
    deg, cc, th = g.feature_arrays()
    r = splitmix_unit(seed, n)
    color = np.full(n, -1, dtype=np.int64)
    rounds = 0
    max_rounds = max_rounds or 10 * n + 10
    while True:
        unc = color < 0
        if not unc.any():
            break
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("no progress: exceeded round limit")
        e_src_unc = unc[src]
        s_, d_ = src[e_src_unc], dst[e_src_unc]
        cd = color[d_]
        col_e = cd >= 0
        # sat: distinct colours among coloured neighbours
        if col_e.any():
            pairs = np.unique(s_[col_e] * (n + 1) + cd[col_e])
            sat = np.bincount(pairs // (n + 1), minlength=n).astype(np.float64)
        else:
            sat = np.zeros(n)
        ucn = np.bincount(s_[~col_e], minlength=n).astype(np.float64)
        score = fn(sat, deg, ucn, cc, th, r)
        score = np.broadcast_to(np.asarray(score, dtype=np.float64), (n,))
        # local maxima among uncoloured vertices
        both = ~col_e
        a, b = s_[both], d_[both]
        lose = (score[b] > score[a]) | ((score[b] == score[a]) & (b < a))
        loser = np.zeros(n, dtype=bool)
        loser[a[lose]] = True
        local = unc & ~loser
        targets = np.flatnonzero(local)
        # smallest free colour w.r.t. neighbours coloured before this round
        sel = local[s_] & col_e
        newc = _smallest_free(s_[sel], cd[sel], targets, n)
        color[targets] = newc[targets]
    return (color.tolist(), rounds) if return_rounds else color.tolist()


def jones_plassmann(g: Graph, seed: int = 0):
    return color_parallel(g, "random", seed)


def luby_mis_coloring(g: Graph, seed: int = 0):
    """Colour class by class; each class is a maximal independent set grown with
    Luby's algorithm (fresh random values per iteration). Returns (colors, rounds)."""
    n = g.n
    indptr, indices = g.csr()
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    dst = indices
    color = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    c = 0
    rounds = 0
    while (color < 0).any():
        cand = color < 0
        while cand.any():
            rounds += 1
            r = rng.random(n)
            both = cand[src] & cand[dst]
            a, b = src[both], dst[both]
            lose = (r[b] > r[a]) | ((r[b] == r[a]) & (b < a))
            loser = np.zeros(n, dtype=bool)
            loser[a[lose]] = True
            win = cand & ~loser
            color[win] = c
            cand &= ~win
            nb = np.zeros(n, dtype=bool)
            ws = win[src]
            nb[dst[ws]] = True
            cand &= ~nb
        c += 1
    return color.tolist(), rounds
