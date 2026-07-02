"""
sequential_heuristics.py

Implements the vertex-ordering scoring formulas from the paper's
formulas table, and the greedy coloring loop that uses them
(Section 3.5, CPU Sequential Pipeline).

Each heuristic is a function score(G, v, coloring, static_features) -> float.
At each greedy step, the uncolored vertex with the highest score is
colored next, using the smallest available color.

Feature definitions (must match the paper's Table of vertex features):
    sat(v)      Distinct colors already assigned to neighbors of v   (dynamic)
    deg(v)      Total number of edges incident to v                  (static)
    ucn(v)      Neighbors of v not yet assigned a color               (dynamic)
    cc(v)       Local clustering coefficient of v, in [0, 1]          (static)
    twohop(v)   Number of vertices at graph distance exactly 2        (static)
"""

from __future__ import annotations

import networkx as nx


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def static_features(G: nx.Graph) -> dict:
    """Precompute degree, clustering coefficient, and two-hop count once."""
    deg = dict(G.degree())
    cc = nx.clustering(G)
    twohop = {}
    for v in G.nodes():
        neighbors = set(G.neighbors(v))
        two_hop_set = set()
        for u in neighbors:
            two_hop_set.update(G.neighbors(u))
        two_hop_set -= neighbors
        two_hop_set.discard(v)
        twohop[v] = len(two_hop_set)
    return {"deg": deg, "cc": cc, "twohop": twohop}


def saturation(G: nx.Graph, v, coloring: dict) -> int:
    """Number of distinct colors already assigned to neighbors of v."""
    return len({coloring[u] for u in G.neighbors(v) if u in coloring})


def uncolored_neighbors(G: nx.Graph, v, coloring: dict) -> int:
    return sum(1 for u in G.neighbors(v) if u not in coloring)


# ---------------------------------------------------------------------------
# Heuristic formulas (verified against the paper's formulas table)
# ---------------------------------------------------------------------------
# Each entry: name -> callable(G, v, coloring, feats) -> float score.
# Higher score = colored sooner.

HEURISTICS = {
    # --- DSATUR (classical baseline; saturation only, degree as tiebreak) ---
    "DSATUR": lambda G, v, c, f: (
        saturation(G, v, c) * 10_000 + f["deg"][v]
    ),

    # --- DeepSeek H1 (best Trial-1; non-linear bilinear term) ---
    "DeepSeek_H1": lambda G, v, c, f: (
        0.6 * saturation(G, v, c) + 0.4 * f["deg"][v] * f["cc"][v]
    ),
    "DeepSeek_H1A": lambda G, v, c, f: (
        0.7 * saturation(G, v, c) + 0.5 * uncolored_neighbors(G, v, c)
    ),
    "DeepSeek_H1B": lambda G, v, c, f: (
        0.8 * saturation(G, v, c)
        + 0.3 * f["twohop"][v]
        - 0.2 * f["deg"][v] * f["cc"][v]
    ),

    # --- GPT-4o H4 (best Trial-1) ---
    "GPT4o_H4": lambda G, v, c, f: (
        6 * saturation(G, v, c) + 2 * f["cc"][v] + uncolored_neighbors(G, v, c)
    ),
    "GPT4o_H4A": lambda G, v, c, f: (
        5 * saturation(G, v, c) + 3 * f["deg"][v] + uncolored_neighbors(G, v, c)
    ),

    # --- GPT-5.1 H1 (best Trial-1) ---
    "GPT51_H1": lambda G, v, c, f: (
        0.48 * saturation(G, v, c) + 0.32 * f["deg"][v]
        + 0.20 * uncolored_neighbors(G, v, c)
    ),
    "GPT51_H1B": lambda G, v, c, f: (
        0.52 * saturation(G, v, c) + 0.28 * f["deg"][v]
        + 0.20 * uncolored_neighbors(G, v, c)
    ),

    # --- Claude (best Trial-1, round-1 refined, round-2 refined) ---
    "Claude_H3": lambda G, v, c, f: (
        2 * saturation(G, v, c) + f["deg"][v] - 3 * f["cc"][v]
    ),
    "Claude_H3A": lambda G, v, c, f: (
        10 * saturation(G, v, c) + 2 * uncolored_neighbors(G, v, c) - 0.5 * f["cc"][v]
    ),
    "Claude_H3A2A": lambda G, v, c, f: (
        20 * saturation(G, v, c) + uncolored_neighbors(G, v, c) + 0.3 * f["twohop"][v]
    ),

    # --- LLaMA H3 (best Trial-1 -- the single best result in the study
    #     at small/medium scale -- and its refined variant) ---
    "LLaMA_H3": lambda G, v, c, f: (
        1.2 * saturation(G, v, c) - 0.6 * f["cc"][v] + 0.1 * uncolored_neighbors(G, v, c)
    ),
    "LLaMA_H3A": lambda G, v, c, f: (
        1.8 * saturation(G, v, c) - 0.2 * f["cc"][v] + 0.05 * uncolored_neighbors(G, v, c)
    ),

    # --- Classical baselines ---
    "RLF_seed_degree": lambda G, v, c, f: (
        # Simplified RLF-style seed selection proxy: prioritize by
        # uncolored-neighbor count (full RLF builds one independent
        # set at a time; see rlf_coloring() below for the real algorithm)
        uncolored_neighbors(G, v, c)
    ),
    "FirstFit": lambda G, v, c, f: (
        # First-Fit processes vertices in index order; implemented via
        # first_fit_coloring() below, not through the score interface.
        0
    ),
}


# ---------------------------------------------------------------------------
# Greedy coloring loop (used by all score-based heuristics, including DSATUR)
# ---------------------------------------------------------------------------

def greedy_color(G: nx.Graph, heuristic_name: str) -> dict:
    """
    Runs the standard vertex-ordering greedy coloring loop: at each
    step, score every uncolored vertex, color the highest-scoring one
    with the smallest available color, repeat until all vertices are
    colored. Ties broken by vertex id (deterministic).
    """
    score_fn = HEURISTICS[heuristic_name]
    feats = static_features(G)
    coloring: dict = {}
    uncolored = set(G.nodes())

    while uncolored:
        best_v = max(uncolored, key=lambda v: (score_fn(G, v, coloring, feats), -v))
        used_colors = {coloring[u] for u in G.neighbors(best_v) if u in coloring}
        color = 0
        while color in used_colors:
            color += 1
        coloring[best_v] = color
        uncolored.remove(best_v)

    return coloring


def first_fit_coloring(G: nx.Graph) -> dict:
    """First-Fit: vertices colored in natural index order."""
    coloring: dict = {}
    for v in sorted(G.nodes()):
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[v] = color
    return coloring


def rlf_coloring(G: nx.Graph) -> dict:
    """
    Recursive Largest First: builds one independent color class at a
    time. Seeds each class with the uncolored vertex with the most
    uncolored neighbors, then greedily grows the class.
    """
    coloring: dict = {}
    remaining = set(G.nodes())
    color = 0

    while remaining:
        # Seed: uncolored vertex with the most uncolored neighbors
        seed = max(remaining, key=lambda v: len(set(G.neighbors(v)) & remaining))
        current_class = {seed}
        remaining.discard(seed)
        candidates = remaining - set(G.neighbors(seed))

        while candidates:
            # Add the candidate with the most neighbors already in the class
            best = max(
                candidates,
                key=lambda v: len(set(G.neighbors(v)) & current_class),
            )
            current_class.add(best)
            remaining.discard(best)
            candidates.discard(best)
            candidates -= set(G.neighbors(best))

        for v in current_class:
            coloring[v] = color
        color += 1

    return coloring


def num_colors_used(coloring: dict) -> int:
    return len(set(coloring.values())) if coloring else 0


def verify_valid_coloring(G: nx.Graph, coloring: dict) -> bool:
    """Sanity check: no edge connects two same-colored vertices."""
    return all(coloring[u] != coloring[v] for u, v in G.edges())


if __name__ == "__main__":
    # Minimal smoke test
    import sys

    G = nx.erdos_renyi_graph(50, 0.15, seed=0)

    for name in ["DSATUR", "LLaMA_H3", "GPT4o_H4", "Claude_H3A2A"]:
        coloring = greedy_color(G, name)
        assert verify_valid_coloring(G, coloring), f"{name} produced an invalid coloring!"
        print(f"{name:15s} -> {num_colors_used(coloring)} colors")

    ff_coloring = first_fit_coloring(G)
    assert verify_valid_coloring(G, ff_coloring)
    print(f"{'FirstFit':15s} -> {num_colors_used(ff_coloring)} colors")

    rlf = rlf_coloring(G)
    assert verify_valid_coloring(G, rlf)
    print(f"{'RLF':15s} -> {num_colors_used(rlf)} colors")

    print("\nAll colorings verified valid.", file=sys.stderr)
