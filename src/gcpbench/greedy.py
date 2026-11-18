"""Sequential greedy coloring with incremental score maintenance (plan S0.1, E1).

Algorithm 1 of the revised manuscript. At each step the uncoloured vertex
with the highest score is coloured with the smallest colour not used by
its neighbours. Ties are broken by the LOWEST vertex id (the rule used,
implicitly, by the original implementation: ``max(..., key=(score, -v))``).

Instead of rescoring every uncoloured vertex at every step (O(n(n+m))),
colouring v updates sat/ucn only for v's neighbours and pushes their new
score onto a max-heap with lazy deletion: O((n + m) log n) overall,
independent of which of the five features a formula uses (bilinear static
terms such as deg*cc do not change this).

Because the score is computed by the same compiled expression on the same
argument values, the coloring is bit-identical to the original naive loop;
``experiments/e0_equivalence.py`` checks this on every instance.
"""

from __future__ import annotations

import heapq

from .graph import Graph


def _smallest_free(used: set) -> int:
    c = 0
    while c in used:
        c += 1
    return c


def greedy_scored(g: Graph, score, *, tuple_key: bool = False, dynamic: bool = True,
                  return_order: bool = False, probe=None, probe_every: int = 0):
    """Generic incremental greedy.

    score(sat, deg, ucn, cc, twohop) -> float, or a tuple if ``tuple_key``.
    ``dynamic=False`` skips rescoring (valid only if score ignores sat and ucn).
    ``probe(color, sat, ucn)`` is called every ``probe_every`` colourings
    (analysis hook for plan G1; it must not modify its arguments).
    Returns colors (list) and optionally the colouring order.
    """
    n, adj = g.n, g.adj
    deg, cc, th = g.static_features()
    color = [-1] * n
    nbr_colors = [set() for _ in range(n)]
    sat = [0] * n
    ucn = list(deg)
    key = [score(0, deg[v], deg[v], cc[v], th[v]) for v in range(n)]
    if tuple_key:
        heap = [tuple(-k for k in key[v]) + (v,) for v in range(n)]
    else:
        heap = [(-key[v], v) for v in range(n)]
    heapq.heapify(heap)
    order = []
    pop, push = heapq.heappop, heapq.heappush

    while heap:
        item = pop(heap)
        v = item[-1]
        if color[v] != -1:
            continue
        if tuple_key:
            if tuple(-k for k in item[:-1]) != key[v]:
                continue
        elif -item[0] != key[v]:
            continue
        if probe is not None and probe_every and len(order) % probe_every == 0:
            probe(color, sat, ucn)
        c = _smallest_free(nbr_colors[v])
        color[v] = c
        order.append(v)
        for u in adj[v]:
            if color[u] != -1:
                continue
            ucn[u] -= 1
            s = nbr_colors[u]
            if c not in s:
                s.add(c)
                sat[u] += 1
            if dynamic:
                k = score(sat[u], deg[u], ucn[u], cc[u], th[u])
                if k != key[u]:
                    key[u] = k
                    push(heap, (tuple(-x for x in k) + (u,)) if tuple_key else (-k, u))
    return (color, order) if return_order else color


def greedy_formula(g: Graph, formula, return_order: bool = False):
    """Run a gcpbench.formulas.Formula through the incremental greedy."""
    return greedy_scored(g, formula.fn, dynamic=bool(formula.dynamic), return_order=return_order)


def greedy_in_order(g: Graph, order) -> list:
    """First-fit colouring along a fixed vertex order."""
    color = [-1] * g.n
    for v in order:
        used = {color[u] for u in g.adj[v] if color[u] != -1}
        color[v] = _smallest_free(used)
    return color


# ---------------------------------------------------------------------------
# Naive reference (same semantics as the original code, but on a Graph);
# used for the equal-footing runtime comparison (E-E) and for testing.
# ---------------------------------------------------------------------------

def greedy_naive(g: Graph, score, return_order: bool = False):
    n, adj = g.n, g.adj
    deg, cc, th = g.static_features()
    color = [-1] * n
    uncolored = set(range(n))
    order = []
    while uncolored:
        def k(v):
            cols = {color[u] for u in adj[v] if color[u] != -1}
            uc = sum(1 for u in adj[v] if color[u] == -1)
            return (score(len(cols), deg[v], uc, cc[v], th[v]), -v)
        v = max(uncolored, key=k)
        color[v] = _smallest_free({color[u] for u in adj[v] if color[u] != -1})
        uncolored.remove(v)
        order.append(v)
    return (color, order) if return_order else color
