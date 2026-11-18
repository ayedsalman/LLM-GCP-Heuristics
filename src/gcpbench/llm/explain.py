"""Automated structural explanation of failure instances (plan B8).

Replaces the author-written explanations of the original refinement prompts
with a fixed, deterministic description: size, density, degree statistics,
clustering, family/scale, and colours used versus DSATUR. Identical code is
used for every model, seed and arm.
"""

from __future__ import annotations

import statistics

from ..graph import Graph


def describe_instance(g: Graph, meta: dict, colors_h: int, colors_dsatur: int) -> str:
    deg, cc, _ = g.static_features()
    n, m = g.n, g.m
    dens = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    mean_d = 2 * m / n if n else 0.0
    sd_d = statistics.pstdev(deg) if n > 1 else 0.0
    cv = sd_d / mean_d if mean_d else 0.0
    hub_share = sum(1 for d in deg if d >= 2 * mean_d) / n if n else 0.0
    mcc = sum(cc) / n if n else 0.0
    shape = ("heavy-tailed degree distribution (hubs)" if cv > 0.8 else
             "moderately uneven degrees" if cv > 0.3 else "near-uniform degrees")
    clus = ("high local clustering" if mcc > 0.3 else
            "moderate local clustering" if mcc > 0.1 else "little local clustering")
    return (f"- {meta.get('instance_id', g.name)} (family {meta.get('family', '?')}, scale "
            f"{meta.get('scale', '?')}): n={n}, m={m}, density={dens:.4f}, mean degree={mean_d:.2f}, "
            f"max degree={max(deg) if deg else 0}, degree SD={sd_d:.2f}, share of vertices with "
            f"degree >= 2x mean={hub_share:.3f}, mean clustering={mcc:.3f}; {shape}, {clus}. "
            f"Colors used: this heuristic {colors_h}, DSATUR {colors_dsatur}.")


def describe_successes(per_cell: dict) -> str:
    """per_cell: (family, scale) -> (mean colours heuristic, mean colours DSATUR)."""
    lines = []
    for (fam, sc), (h, d) in sorted(per_cell.items()):
        if h <= d:
            lines.append(f"- {fam} {sc}: {h:.2f} vs DSATUR {d:.2f}")
    return "\n".join(lines) if lines else "- none"
