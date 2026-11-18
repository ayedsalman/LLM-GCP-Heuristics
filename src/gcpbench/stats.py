"""Statistical protocol of the revision (plan Section 2.3; S0.4, B6, D7).

Primary analysis  per-instance paired difference (heuristic - baseline) in
                  colour count: mean with 95% bootstrap CI (10,000 resamples),
                  Wilcoxon signed-rank (Pratt zero handling), Holm correction
                  across all heuristic-vs-baseline comparisons, and the share
                  of instances with an exactly equal colour count.
Equivalence       TOST on the paired differences against a pre-registered
                  margin (default delta = 0.10 colours). "Indistinguishable"
                  in the paper becomes "equivalent within +/-delta" only when
                  TOST rejects both one-sided nulls.
Continuity        Friedman + Nemenyi. The critical value q_alpha is computed
                  from scipy.stats.studentized_range (infinite df) divided by
                  sqrt(2) -- the source of Demsar's (2006) table -- so any k is
                  supported and the source is recorded in every report.
Across seeds      per-seed TEST means summarised by mean, SD, 95% bootstrap CI;
                  Kruskal-Wallis across models, Holm-corrected pairwise
                  Mann-Whitney with Cliff's delta as effect size.
Ties in ranks     average ranks (scipy.stats.rankdata default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Q_SOURCE = "scipy.stats.studentized_range.ppf(1-alpha, k, inf) / sqrt(2)"
DEFAULT_DELTA = 0.10
N_BOOT = 10_000


# ------------------------------------------------------------- basic tools

def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(x, stat=np.mean, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    if np.all(x == x[0]):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    if stat is np.mean:
        bs = x[idx].mean(axis=1)
    else:
        bs = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def wilcoxon_p(diff) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    if np.all(d == 0) or len(d) == 0:
        return (math.nan, 1.0)
    res = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


@dataclass
class TOST:
    delta: float
    mean: float
    ci90: tuple
    p_lower: float
    p_upper: float

    @property
    def p(self) -> float:
        return max(self.p_lower, self.p_upper)

    def equivalent(self, alpha: float = 0.05) -> bool:
        return self.p < alpha


def tost_paired(diff, delta: float = DEFAULT_DELTA) -> TOST:
    """Two one-sided paired t-tests: H0a mu <= -delta, H0b mu >= +delta."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        inside = abs(mu) < delta
        p = 0.0 if inside else 1.0
        return TOST(delta, mu, (mu, mu), p, p)
    se = sd / math.sqrt(n)
    t1 = (mu + delta) / se
    t2 = (mu - delta) / se
    p1 = float(stats.t.sf(t1, n - 1))
    p2 = float(stats.t.cdf(t2, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return TOST(delta, mu, (mu - h, mu + h), p1, p2)


def cliffs_delta(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


# -------------------------------------------- paired comparison vs baseline

def paired_vs_baseline(wide: pd.DataFrame, baseline: str, algorithms=None,
                       delta: float = DEFAULT_DELTA, n_boot: int = N_BOOT,
                       alpha: float = 0.05) -> pd.DataFrame:
    """wide: rows = instances, columns = algorithms, values = colour counts."""
    algorithms = [a for a in (algorithms or wide.columns) if a != baseline]
    rows = []
    base = wide[baseline].to_numpy(float)
    for a in algorithms:
        x = wide[a].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(base))
        d = x[ok] - base[ok]
        lo, hi = bootstrap_ci(d, n_boot=n_boot)
        w, p = wilcoxon_p(d)
        t = tost_paired(d, delta)
        rows.append({
            "algorithm": a, "baseline": baseline, "N": int(ok.sum()),
            "mean_alg": float(x[ok].mean()), "mean_base": float(base[ok].mean()),
            "mean_diff": float(d.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "share_equal": float((d == 0).mean()), "share_better": float((d < 0).mean()),
            "share_worse": float((d > 0).mean()),
            "wilcoxon_W": w, "p_wilcoxon": p,
            "tost_delta": delta, "tost_ci90_lo": t.ci90[0], "tost_ci90_hi": t.ci90[1],
            "p_tost": t.p, "equivalent": t.equivalent(alpha),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].fillna(1.0))
        out["significant_holm"] = out["p_holm"] < alpha
        out["verdict"] = np.where(out["equivalent"], f"equivalent (|d|<{delta})",
                                  np.where(out["significant_holm"],
                                           np.where(out["mean_diff"] < 0, "better", "worse"),
                                           "inconclusive"))
    return out


# ------------------------------------------------------ Friedman / Nemenyi

def q_alpha(k: int, alpha: float = 0.05) -> float:
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    return q_alpha(k, alpha) * math.sqrt(k * (k + 1) / (6.0 * n))


def friedman_nemenyi(wide: pd.DataFrame, alpha: float = 0.05) -> dict:
    df = wide.dropna(axis=0, how="any")
    n, k = df.shape
    chi2, p = stats.friedmanchisquare(*[df[c].to_numpy(float) for c in df.columns])
    ranks = df.apply(lambda r: stats.rankdata(r.to_numpy(float)), axis=1, result_type="expand")
    ranks.columns = df.columns
    avg = ranks.mean().sort_values()
    cd = critical_difference(k, n, alpha)
    try:
        import scikit_posthocs as sp
        pm = sp.posthoc_nemenyi_friedman(df.to_numpy(float))
        pm.index = pm.columns = list(df.columns)
    except Exception:  # pragma: no cover
        pm = None
    return {"N": n, "k": k, "chi2": float(chi2), "df": k - 1, "p": float(p),
            "CD": cd, "q_alpha": q_alpha(k, alpha), "q_source": Q_SOURCE, "alpha": alpha,
            "avg_ranks": avg, "nemenyi_p": pm}


def test_table_row(label: str, set_name: str, scale: str, res: dict) -> dict:
    """One row of the 'every statistical test' table requested by R1.7 (D7)."""
    return {"test": label, "set": set_name, "scale": scale, "N": res["N"], "k": res["k"],
            "statistic": round(res["chi2"], 3), "df": res["df"], "p": res["p"],
            "CD": round(res["CD"], 4), "q_alpha": round(res["q_alpha"], 4)}


# ------------------------------------------------------------- across seeds

def across_seeds(per_seed: dict, n_boot: int = N_BOOT, alpha: float = 0.05):
    """per_seed: model -> array of per-seed TEST mean colours.
    Returns (summary DataFrame, pairwise DataFrame, Kruskal-Wallis result)."""
    summ = []
    for m, x in per_seed.items():
        x = np.asarray(x, float)
        lo, hi = bootstrap_ci(x, n_boot=n_boot)
        summ.append({"model": m, "n_seeds": len(x), "mean": x.mean(),
                     "sd": x.std(ddof=1) if len(x) > 1 else 0.0, "ci95_lo": lo, "ci95_hi": hi,
                     "best": x.min(), "median": float(np.median(x)), "worst": x.max()})
    summ = pd.DataFrame(summ)
    groups = [np.asarray(v, float) for v in per_seed.values()]
    kw = stats.kruskal(*groups) if len(groups) > 1 and all(len(g) > 1 for g in groups) else None
    pairs = []
    names = list(per_seed)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.asarray(per_seed[names[i]], float), np.asarray(per_seed[names[j]], float)
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else math.nan
            pairs.append({"a": names[i], "b": names[j], "p_mwu": float(p),
                          "cliffs_delta": cliffs_delta(a, b), "mean_diff": float(a.mean() - b.mean())})
    pairs = pd.DataFrame(pairs)
    if len(pairs):
        pairs["p_holm"] = holm(pairs["p_mwu"].fillna(1.0))
    return summ, pairs, kw


def stratified_means(long: pd.DataFrame, value: str = "colors",
                     by=("family", "scale"), algo_col: str = "algorithm") -> pd.DataFrame:
    """Per-family x per-scale mean of ``value`` for each algorithm, plus pooled."""
    t = long.pivot_table(index=list(by), columns=algo_col, values=value, aggfunc="mean")
    pooled = long.groupby(algo_col)[value].mean().to_frame().T
    pooled.index = pd.MultiIndex.from_tuples([("pooled",) * len(by)], names=list(by))
    return pd.concat([t, pooled])
