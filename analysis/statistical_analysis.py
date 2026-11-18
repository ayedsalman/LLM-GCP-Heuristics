"""
statistical_analysis.py

Friedman test with Nemenyi post-hoc correction, as described in the
paper's Statistical Analysis section (Section 3.9) and applied to the
sequential and parallel results in Section 4.

Usage
-----
Input: a CSV with one row per (instance, algorithm) pair, or a wide
CSV with one row per instance and one column per algorithm. Both
formats are supported -- see `load_results()`.

    python statistical_analysis.py --input data/seq_small.csv --alpha 0.05

Or import the functions directly:

    from statistical_analysis import run_friedman_nemenyi
    report = run_friedman_nemenyi(df, alpha=0.05)
    print(report.summary())

Methodology (matches paper Section 3.9)
----------------------------------------
1. For each graph instance, every algorithm's color count is ranked
   (rank 1 = fewest colors = best).
2. The Friedman statistic chi^2_F is computed over the resulting
   N (instances) x k (algorithms) rank matrix. A significant Friedman
   test (p < alpha) indicates that at least one pair of algorithms
   differs significantly.
3. Where the Friedman test is significant, the Nemenyi post-hoc
   procedure computes a critical difference (CD) at the chosen alpha.
   Any two algorithms whose average ranks differ by more than CD are
   considered statistically distinguishable; algorithms within CD of
   one another are treated as practically equivalent.

All analyses in the paper use alpha = 0.05, scipy.stats for the
Friedman test, and scikit-posthocs for the Nemenyi post-hoc test.

Revision note: "within CD of the baseline" is NOT evidence of equivalence
(failure to reject a null). For claims that a heuristic "matches" DSATUR use
``--paired`` (TOST against a pre-registered margin), implemented in
src/gcpbench/stats.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

try:
    import scikit_posthocs as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-posthocs is required. Install with:\n"
        "    pip install scikit-posthocs"
    ) from exc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    """
    Load a results file and return a wide DataFrame:
    rows = graph instances, columns = algorithms, values = color count
    (or any other per-instance metric being compared, e.g. runtime).

    Accepts either:
      - Wide format already: columns are algorithm names, one row per
        instance. Returned as-is.
      - Long format: columns [instance_id, algorithm, value]. Pivoted
        to wide format automatically.
    """
    df = pd.read_csv(path)

    long_format_cols = {"instance_id", "algorithm", "value"}
    if long_format_cols.issubset(set(df.columns)):
        wide = df.pivot(index="instance_id", columns="algorithm", values="value")
        wide.columns.name = None
        return wide

    # Assume already wide: drop any non-numeric id column if present
    if "instance_id" in df.columns:
        df = df.set_index("instance_id")
    return df


# ---------------------------------------------------------------------------
# Critical difference (Nemenyi)
# ---------------------------------------------------------------------------

# Studentized range statistic q_alpha for the Nemenyi test (two-tailed),
# indexed by number of algorithms k. Standard table from Demsar (2006),
# "Statistical Comparisons of Classifiers over Multiple Data Sets."
_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544,
}
_Q_ALPHA_010 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
    8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077,
    14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291,
    20: 3.319,
}


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / (6N))

    Revision (plan S0.4 / Appendix B): q_alpha is now computed from the
    studentized range distribution (scipy.stats.studentized_range with
    infinite df, divided by sqrt(2)), which is the source of Demsar's
    (2006) table, so any k and alpha are supported and the source is
    recorded. The tabulated values above are kept only as a cross-check
    (they agree to 3 decimals for k = 2..20, alpha in {0.05, 0.10}).
    """
    from scipy.stats import studentized_range
    q_alpha = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    table = _Q_ALPHA_005 if alpha == 0.05 else _Q_ALPHA_010 if alpha == 0.10 else {}
    if k in table and abs(table[k] - q_alpha) > 2e-3:  # pragma: no cover
        raise AssertionError(f"q_alpha mismatch for k={k}: table {table[k]} vs computed {q_alpha:.4f}")
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

@dataclass
class FriedmanNemenyiReport:
    algorithms: list
    n_instances: int
    k_algorithms: int
    chi2: float
    df: int
    p_value: float
    avg_ranks: pd.Series          # sorted ascending (best first)
    nemenyi_pvalues: pd.DataFrame  # pairwise p-value matrix
    critical_difference: float
    alpha: float
    baseline: str | None = None
    indistinguishable_from_baseline: list = field(default_factory=list)
    significantly_different_from_baseline: list = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        lines.append(
            f"Friedman test over N={self.n_instances} instances and "
            f"k={self.k_algorithms} algorithms: "
            f"chi^2_F={self.chi2:.2f}, df={self.df}, "
            f"p={'<0.001' if self.p_value < 0.001 else f'{self.p_value:.3g}'}, "
            f"Nemenyi CD={self.critical_difference:.3f} (alpha={self.alpha})."
        )
        lines.append("Average ranks (best first):")
        for alg, rank in self.avg_ranks.items():
            lines.append(f"  {alg:20s} {rank:.2f}")
        if self.baseline:
            lines.append(f"\nRelative to baseline '{self.baseline}':")
            lines.append(
                f"  Statistically indistinguishable (within CD): "
                f"{', '.join(self.indistinguishable_from_baseline) or 'none'}"
            )
            lines.append(
                f"  Significantly different (exceeds CD): "
                f"{', '.join(self.significantly_different_from_baseline) or 'none'}"
            )
        return "\n".join(lines)


def run_friedman_nemenyi(
    df: pd.DataFrame,
    alpha: float = 0.05,
    baseline: str | None = None,
) -> FriedmanNemenyiReport:
    """
    df       : wide DataFrame, rows = instances, columns = algorithms,
               values = color count (lower is better) or any other
               per-instance metric being compared.
    alpha    : significance level (default 0.05, matching the paper).
    baseline : optional algorithm name (e.g. "DSATUR" or "LLM_cluster")
               to report which other algorithms are statistically
               indistinguishable from it vs. significantly different.

    Returns a FriedmanNemenyiReport with the same quantities reported
    in the paper: chi^2_F, df, p, average ranks, and Nemenyi CD.
    """
    df = df.dropna(axis=0, how="any")  # Friedman requires complete blocks
    n, k = df.shape
    algorithms = list(df.columns)

    # --- Friedman test ---
    chi2, p_value = stats.friedmanchisquare(*[df[c].values for c in algorithms])
    dof = k - 1

    # --- Average ranks (1 = best i.e. fewest colors) ---
    ranks = df.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
    ranks.columns = algorithms
    avg_ranks = ranks.mean(axis=0).sort_values()

    # --- Nemenyi post-hoc pairwise p-values ---
    # scikit-posthocs accepts the block-design matrix directly:
    # rows = blocks (instances), columns = groups (algorithms).
    nemenyi_p = sp.posthoc_nemenyi_friedman(df.values)
    nemenyi_p.index = algorithms
    nemenyi_p.columns = algorithms

    cd = critical_difference(k=k, n=n, alpha=alpha)

    report = FriedmanNemenyiReport(
        algorithms=algorithms,
        n_instances=n,
        k_algorithms=k,
        chi2=chi2,
        df=dof,
        p_value=p_value,
        avg_ranks=avg_ranks,
        nemenyi_pvalues=nemenyi_p,
        critical_difference=cd,
        alpha=alpha,
        baseline=baseline,
    )

    if baseline is not None:
        if baseline not in avg_ranks.index:
            raise ValueError(f"baseline '{baseline}' not found among algorithms")
        base_rank = avg_ranks[baseline]
        for alg in avg_ranks.index:
            if alg == baseline:
                continue
            rank_diff = abs(avg_ranks[alg] - base_rank)
            if rank_diff <= cd:
                report.indistinguishable_from_baseline.append(alg)
            else:
                report.significantly_different_from_baseline.append(alg)

    return report


# ---------------------------------------------------------------------------
# CD diagram (Demsar-style, as commonly used alongside this analysis)
# ---------------------------------------------------------------------------

def plot_cd_diagram(report: FriedmanNemenyiReport, out_path: str) -> None:
    """
    Draws a simple critical-difference diagram: algorithms placed on a
    number line by average rank, with a horizontal bar connecting any
    group of algorithms whose ranks are all within CD of each other.
    Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = report.avg_ranks.sort_values()
    names = ranks.index.tolist()
    values = ranks.values

    fig, ax = plt.subplots(figsize=(9, 3.2))
    x_pad = max(0.5, cd_pad := (max(values) - min(values)) * 0.15 + 0.3)
    ax.set_xlim(min(values) - x_pad, max(values) + x_pad)
    ax.axhline(0, color="black", linewidth=0.8)

    # Stagger label vertical offset based on proximity in x, not just index
    # parity, so labels for closely-ranked algorithms don't collide.
    order = np.argsort(values)
    offsets = [0] * len(values)
    lane = 0
    for pos, idx in enumerate(order):
        if pos > 0:
            prev_idx = order[pos - 1]
            if values[idx] - values[prev_idx] < x_pad * 0.6:
                lane = (lane + 1) % 4
            else:
                lane = 0
        offsets[idx] = lane

    for i, (name, val) in enumerate(zip(names, values)):
        ax.plot(val, 0, "o", color="steelblue", zorder=3)
        lane = offsets[i]
        y_off = 14 + lane * 13 if lane % 2 == 0 else -14 - lane * 13
        ax.annotate(
            f"{name}\n({val:.2f})",
            (val, 0),
            textcoords="offset points",
            xytext=(0, y_off),
            ha="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="gray") if abs(y_off) > 20 else None,
        )

    # Draw CD bars over cliques of statistically indistinguishable algorithms
    cd = report.critical_difference
    used = set()
    y = -0.15
    for i in range(len(values)):
        if names[i] in used:
            continue
        group = [i]
        for j in range(i + 1, len(values)):
            if values[j] - values[i] <= cd:
                group.append(j)
        if len(group) > 1:
            ax.plot(
                [values[group[0]], values[group[-1]]],
                [y, y],
                color="black",
                linewidth=2.5,
            )
            for idx in group:
                used.add(names[idx])
            y -= 0.12

    ax.set_yticks([])
    ax.set_ylim(y - 0.15, 0.28)
    ax.set_title(f"CD diagram (CD={cd:.3f}, alpha={report.alpha})", pad=12)
    ax.set_xlabel("Average rank (lower = better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of per-instance results")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--baseline", default=None, help="e.g. DSATUR or Jones-Plassmann")
    parser.add_argument("--cd-plot", default=None, help="optional path to save a CD diagram PNG")
    parser.add_argument("--paired", action="store_true",
                        help="also report the revision's primary analysis: paired differences vs "
                             "--baseline with bootstrap CI, Wilcoxon+Holm and TOST (plan Sec. 2.3)")
    parser.add_argument("--delta", type=float, default=0.10, help="TOST equivalence margin (colours)")
    args = parser.parse_args()

    df = load_results(args.input)
    report = run_friedman_nemenyi(df, alpha=args.alpha, baseline=args.baseline)
    print(report.summary())

    if args.paired:
        if not args.baseline:
            raise SystemExit("--paired requires --baseline")
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        from gcpbench.stats import paired_vs_baseline
        pd.set_option("display.width", 200)
        tab = paired_vs_baseline(df, args.baseline, delta=args.delta)
        print("\nPaired comparison vs", args.baseline)
        print(tab[["algorithm", "N", "mean_diff", "ci95_lo", "ci95_hi", "share_equal",
                   "p_holm", "p_tost", "verdict"]].to_string(index=False, float_format="%.4f"))

    if args.cd_plot:
        plot_cd_diagram(report, args.cd_plot)
        print(f"\nCD diagram saved to {args.cd_plot}")


if __name__ == "__main__":
    main()
