"""Classical sequential baselines on the same Graph / tie-breaking (plan G6, E1).

All methods break remaining ties by the lowest vertex id.

DSATUR_paper   sat*10000 + deg (the baseline used in the submitted paper)
DSATUR_lex     lexicographic (sat, deg)
DSATUR_brelaz  lexicographic (sat, degree in the uncoloured subgraph) -- Brelaz (1979)
DSATUR_sewell  (sat, Sewell's tie-break: max sum over uncoloured neighbours u of
               |A(v) & A(u)|, A = available colours among those already in use)
IDO            incidence-degree ordering: (#coloured neighbours, deg)
WelshPowell    static largest-degree-first order, first-fit
SmallestLast   Matula-Beck degeneracy order, first-fit
FirstFit       vertex-index order
RLF            Leighton's Recursive Largest First (correct version)
RLF_legacy     the submitted paper's RLF; see note below

Note on RLF_legacy. The original ``rlf_coloring`` chooses the next vertex of
a colour class by ``len(N(v) & current_class)``. Every candidate is, by
construction, non-adjacent to the class, so this key is always 0 and the
choice falls back to Python's set-iteration order: the method is a
maximal-independent-set greedy with an arbitrary growth rule, not RLF.
Leighton's rule maximises the number of neighbours in the set W of
uncoloured vertices already excluded from the class (ties: fewest
neighbours among the remaining candidates). ``RLF`` implements that rule.
"""

from __future__ import annotations

import heapq

import numpy as np

from .graph import Graph
from .greedy import greedy_scored, greedy_in_order, _smallest_free


def first_fit(g: Graph):
    return greedy_in_order(g, range(g.n))


def welsh_powell(g: Graph):
    order = sorted(range(g.n), key=lambda v: (-len(g.adj[v]), v))
    return greedy_in_order(g, order)


def smallest_last_order(g: Graph):
    """Matula-Beck: repeatedly delete a minimum-degree vertex (lowest id on ties);
    colour in reverse deletion order. O(m log n) with a lazy heap."""
    d = [len(a) for a in g.adj]
    removed = [False] * g.n
    heap = [(d[v], v) for v in range(g.n)]
    heapq.heapify(heap)
    seq = []
    while heap:
        dv, v = heapq.heappop(heap)
        if removed[v] or dv != d[v]:
            continue
        removed[v] = True
        seq.append(v)
        for u in g.adj[v]:
            if not removed[u]:
                d[u] -= 1
                heapq.heappush(heap, (d[u], u))
    return seq[::-1]


def smallest_last(g: Graph):
    return greedy_in_order(g, smallest_last_order(g))


def dsatur_paper(g: Graph):
    return greedy_scored(g, lambda s, d, u, c, t: s * 10000 + d)


def dsatur_lex(g: Graph):
    return greedy_scored(g, lambda s, d, u, c, t: (s, d), tuple_key=True)


def dsatur_brelaz(g: Graph):
    return greedy_scored(g, lambda s, d, u, c, t: (s, u), tuple_key=True)


def ido(g: Graph):
    return greedy_scored(g, lambda s, d, u, c, t: (d - u, d), tuple_key=True)


def dsatur_sewell(g: Graph):
    """DSATUR with Sewell's (1996) tie-break among maximum-saturation vertices:
    maximise sum over uncoloured neighbours u of |A(v) & A(u)|, where A(x) is the
    set of colours already in use that x could still take; then lowest id."""
    n, adj = g.n, g.adj
    color = [-1] * n
    nbr = [set() for _ in range(n)]
    sat = np.zeros(n, dtype=np.int64)
    unc = np.ones(n, dtype=bool)
    k_used = 0
    for _ in range(n):
        smax = sat[unc].max()
        ties = np.flatnonzero(unc & (sat == smax))
        if len(ties) > 1 and k_used > 0:
            best, v = None, -1
            for x in ties:                       # ascending ids
                av = [c for c in range(k_used) if c not in nbr[x]]
                tot = 0
                for u in adj[x]:
                    if color[u] == -1:
                        nu = nbr[u]
                        tot += sum(1 for c in av if c not in nu)
                if best is None or tot > best:
                    best, v = tot, int(x)
        else:
            v = int(ties[0])
        c = _smallest_free(nbr[v])
        color[v] = c
        k_used = max(k_used, c + 1)
        unc[v] = False
        for u in adj[v]:
            if c not in nbr[u]:
                nbr[u].add(c)
                if color[u] == -1:
                    sat[u] += 1
    return color


def rlf(g: Graph):
    """Leighton's RLF. Seed: max degree in the uncoloured subgraph; grow by max
    |N(v) & W|, ties min |N(v) & U|, then lowest id. O(n^2) array scans."""
    n = g.n
    indptr, indices = g.csr()
    color = np.full(n, -1, dtype=np.int64)
    deg_unc = np.diff(indptr).astype(np.int64)
    c = 0
    big = n + 1
    while (color < 0).any():
        in_u = color < 0                        # candidates for this class
        in_w = np.zeros(n, dtype=bool)          # uncoloured, excluded from class
        d_u = deg_unc.copy()                    # neighbours inside U
        d_w = np.zeros(n, dtype=np.int64)       # neighbours inside W
        first = True
        while in_u.any():
            if first:
                key = np.where(in_u, d_u, -1)
                first = False
            else:
                key = np.where(in_u, d_w * big + (big - 1 - d_u), -1)
            v = int(np.argmax(key))             # argmax -> lowest id on ties
            color[v] = c
            in_u[v] = False
            nb = indices[indptr[v]:indptr[v + 1]]
            # v leaves U: its U-neighbours lose one U-neighbour
            d_u[nb] -= 1
            # U-neighbours of v move to W
            moving = nb[in_u[nb]]
            in_u[moving] = False
            in_w[moving] = True
            for x in moving:
                nx_ = indices[indptr[x]:indptr[x + 1]]
                d_u[nx_] -= 1
                d_w[nx_] += 1
        # remove the class from the uncoloured subgraph
        cls = np.flatnonzero(color == c)
        for v in cls:
            deg_unc[indices[indptr[v]:indptr[v + 1]]] -= 1
        c += 1
    return color.tolist()


def rlf_legacy(g: Graph):
    """Verbatim logic of the submitted paper's rlf_coloring (see module note)."""
    adjs = [set(a) for a in g.adj]
    coloring = [-1] * g.n
    remaining = set(range(g.n))
    color = 0
    while remaining:
        seed = max(remaining, key=lambda v: len(adjs[v] & remaining))
        current = {seed}
        remaining.discard(seed)
        candidates = remaining - adjs[seed]
        while candidates:
            best = max(candidates, key=lambda v: len(adjs[v] & current))
            current.add(best)
            remaining.discard(best)
            candidates.discard(best)
            candidates -= adjs[best]
        for v in current:
            coloring[v] = color
        color += 1
    return coloring


CLASSICAL = {
    "DSATUR": dsatur_paper,
    "DSATUR_lex": dsatur_lex,
    "DSATUR_brelaz": dsatur_brelaz,
    "DSATUR_sewell": dsatur_sewell,
    "IDO": ido,
    "WelshPowell": welsh_powell,
    "SmallestLast": smallest_last,
    "FirstFit": first_fit,
    "RLF": rlf,
    "RLF_legacy": rlf_legacy,
}
