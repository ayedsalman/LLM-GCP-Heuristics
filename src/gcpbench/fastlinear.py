"""Numba engine for linear-in-dynamic-features scores (controls, tuning, perturbation).

score(v) = w_sat*sat(v) + w_ucn*ucn(v) + static(v)

where static(v) is any precomputed per-vertex term (for the five-feature
family: w_deg*deg + w_cc*cc + w_twohop*twohop). Same incremental algorithm
and tie-breaking (lowest id) as gcpbench.greedy, compiled with numba so the
random/handcrafted/tuned control arms (E-B2, B3, G5) can afford thousands of
evaluations. Because the summation order differs from a formula's written
order, scores can differ from the exact engine in the last bit; the
agreement of colour counts with the exact engine is reported by
experiments/e0_equivalence.py.

Falls back to the pure-Python engine if numba is unavailable.
"""

from __future__ import annotations

import numpy as np

from . import FEATURES

try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


if HAVE_NUMBA:
    @njit(cache=True, inline="always")
    def _better(hk, hv, a, b):
        return hk[a] > hk[b] or (hk[a] == hk[b] and hv[a] < hv[b])

    @njit(cache=True)
    def _color_linear(indptr, indices, w_sat, w_ucn, static, out_colors):
        n = indptr.shape[0] - 1
        deg = np.empty(n, np.int64)
        maxdeg = 0
        for v in range(n):
            deg[v] = indptr[v + 1] - indptr[v]
            if deg[v] > maxdeg:
                maxdeg = deg[v]
        kmax = maxdeg + 2
        seen = np.zeros(n * kmax, np.bool_)       # seen[v*kmax + c]: colour c among nbrs of v
        sat = np.zeros(n, np.int64)
        ucn = deg.copy()
        color = -np.ones(n, np.int64)
        key = np.empty(n, np.float64)
        cap = n + indices.shape[0] + 1
        hk = np.empty(cap, np.float64)
        hv = np.empty(cap, np.int64)
        size = 0
        # build heap
        for v in range(n):
            key[v] = w_sat * 0.0 + w_ucn * ucn[v] + static[v]
            hk[size] = key[v]
            hv[size] = v
            size += 1
        for i in range(size // 2 - 1, -1, -1):
            j = i
            while True:
                l = 2 * j + 1
                if l >= size:
                    break
                b = l
                r = l + 1
                if r < size and _better(hk, hv, r, l):
                    b = r
                if _better(hk, hv, b, j):
                    hk[b], hk[j] = hk[j], hk[b]
                    hv[b], hv[j] = hv[j], hv[b]
                    j = b
                else:
                    break
        ncol = 0
        while size > 0:
            k0 = hk[0]
            v = hv[0]
            size -= 1
            hk[0] = hk[size]
            hv[0] = hv[size]
            j = 0
            while True:
                l = 2 * j + 1
                if l >= size:
                    break
                b = l
                r = l + 1
                if r < size and _better(hk, hv, r, l):
                    b = r
                if _better(hk, hv, b, j):
                    hk[b], hk[j] = hk[j], hk[b]
                    hv[b], hv[j] = hv[j], hv[b]
                    j = b
                else:
                    break
            if color[v] != -1 or k0 != key[v]:
                continue
            c = 0
            base = v * kmax
            while seen[base + c]:
                c += 1
            color[v] = c
            if c + 1 > ncol:
                ncol = c + 1
            for p in range(indptr[v], indptr[v + 1]):
                u = indices[p]
                if color[u] != -1:
                    continue
                ucn[u] -= 1
                if not seen[u * kmax + c]:
                    seen[u * kmax + c] = True
                    sat[u] += 1
                nk = w_sat * sat[u] + w_ucn * ucn[u] + static[u]
                if nk != key[u]:
                    key[u] = nk
                    hk[size] = nk
                    hv[size] = u
                    i = size
                    size += 1
                    while i > 0:
                        par = (i - 1) // 2
                        if _better(hk, hv, i, par):
                            hk[i], hk[par] = hk[par], hk[i]
                            hv[i], hv[par] = hv[par], hv[i]
                            i = par
                        else:
                            break
        for v in range(n):
            out_colors[v] = color[v]
        return ncol


class LinearEvaluator:
    """Pre-loads a set of graphs and evaluates coefficient vectors quickly."""

    def __init__(self, graphs):
        self.graphs = list(graphs)
        self._csr = [g.csr() for g in self.graphs]
        self._feat = [g.feature_arrays() for g in self.graphs]

    def colors_one(self, i: int, w) -> int:
        w = np.asarray(w, dtype=np.float64)
        w_sat, w_deg, w_ucn, w_cc, w_th = (float(x) for x in w)
        deg, cc, th = self._feat[i]
        static = w_deg * deg + w_cc * cc + w_th * th
        indptr, indices = self._csr[i]
        if not HAVE_NUMBA:  # pragma: no cover
            raise RuntimeError("numba is required for LinearEvaluator (pip install numba)")
        out = np.empty(len(indptr) - 1, np.int64)
        return int(_color_linear(indptr, indices, w_sat, w_ucn, static, out))

    def colors(self, w) -> np.ndarray:
        return np.array([self.colors_one(i, w) for i in range(len(self.graphs))], dtype=np.int64)

    def coloring(self, i: int, w) -> np.ndarray:
        w_sat, w_deg, w_ucn, w_cc, w_th = (float(x) for x in w)
        deg, cc, th = self._feat[i]
        static = w_deg * deg + w_cc * cc + w_th * th
        indptr, indices = self._csr[i]
        out = np.empty(len(indptr) - 1, np.int64)
        _color_linear(indptr, indices, w_sat, w_ucn, static, out)
        return out


FEATURE_ORDER = FEATURES  # (sat, deg, ucn, cc, twohop)
