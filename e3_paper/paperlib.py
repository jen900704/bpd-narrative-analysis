"""Pure, side-effect-free functions shared by the e3_paper rerun scripts.

Extracted so that the logic in r1/r2/r3/r4 can be unit tested. Nothing in this module
reads a file, writes a file, or touches participant data; every function takes arrays
or frames and returns values. Tests live in
discovery_analysis/tests/test_e3_paper_lib.py.
"""
import numpy as np
import pandas as pd

__all__ = ["build_years_ago", "split_primary_family", "heaping_counts",
           "cronbach", "alpha_if_deleted", "identity_metrics",
           "percentile_ci", "two_sided_bootstrap_p", "cluster_bootstrap_median"]


def build_years_ago(current_age_midpoint, onset_age):
    """Elapsed time since a reported onset, in years.

    current_age_midpoint - onset_age, with negative results set to missing (a reported
    onset later than the current age band cannot be interpreted). Both inputs are
    self-reported and current_age_midpoint is a band midpoint, so the result carries an
    error of roughly half a band width.
    """
    a = pd.Series(current_age_midpoint, dtype="float64").reset_index(drop=True)
    b = pd.Series(onset_age, dtype="float64").reset_index(drop=True)
    y = a - b
    return y.where(~(y < 0), np.nan)


def split_primary_family(frame, domain_col, proxy_domain):
    """Split a flagged event frame into the primary family and the named proxy domain.

    Returns (primary, proxy). The proxy domain must be present; a silent no-op split
    is the failure mode this guards against.
    """
    doms = set(frame[domain_col].dropna().unique().tolist())
    if proxy_domain not in doms:
        raise KeyError("proxy domain %r not present; domains are %r"
                       % (proxy_domain, sorted(doms)))
    is_proxy = frame[domain_col] == proxy_domain
    return frame.loc[~is_proxy].copy(), frame.loc[is_proxy].copy()


def heaping_counts(ages):
    """Round-number concentration in reported ages.

    Multiples of ten are a SUBSET of multiples of five, so they are reported nested,
    never as two independent percentages. `multiples_of_5_not_10` is the disjoint
    remainder and the three counts satisfy m5 = m10 + m5_not_10 by construction.
    """
    a = pd.Series(ages, dtype="float64").dropna()
    n = int(len(a))
    m5 = a % 5 == 0
    m10 = a % 10 == 0
    only5 = m5 & ~m10
    pct = (lambda s: round(float(s.mean() * 100), 2)) if n else (lambda s: 0.0)
    return {"n_events": n,
            "multiples_of_5_n": int(m5.sum()), "multiples_of_5_pct": pct(m5),
            "of_which_multiples_of_10_n": int(m10.sum()),
            "of_which_multiples_of_10_pct": pct(m10),
            "multiples_of_5_not_10_n": int(only5.sum()),
            "multiples_of_5_not_10_pct": pct(only5)}


def cronbach(X):
    """Cronbach's alpha on complete cases. NaN when k < 2 or n < 3."""
    X = pd.DataFrame(X).dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return float("nan")
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var / total_var))


def alpha_if_deleted(X):
    """{item: alpha of the scale with that item removed}, on complete cases."""
    X = pd.DataFrame(X).dropna()
    return {c: cronbach(X[[o for o in X.columns if o != c]]) for c in X.columns}


def identity_metrics(recomputed, archived):
    """Is a recomputed score the same number as the archived one, or only correlated?

    A perfect correlation is compatible with any affine rescaling, so r alone cannot
    establish identity. Returns MAE, max |diff| and the calibration line
    archived = slope * recomputed + intercept, whose residuals distinguish an exact
    affine rescaling from a merely strong linear relationship.
    """
    a = np.asarray(recomputed, dtype="float64")
    b = np.asarray(archived, dtype="float64")
    if a.shape != b.shape:
        raise ValueError("shape mismatch: %r vs %r" % (a.shape, b.shape))
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        raise ValueError("fewer than 3 comparable values")
    d = a - b
    slope, intercept = np.polyfit(a, b, 1)
    resid = b - (slope * a + intercept)
    return {"n_compared": int(len(a)),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "mean_signed_difference": float(d.mean()),
            "MAE": float(np.abs(d).mean()),
            "RMSE": float(np.sqrt((d ** 2).mean())),
            "max_abs_difference": float(np.abs(d).max()),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
            "calibration_max_abs_residual": float(np.abs(resid).max())}


def percentile_ci(values, level=95.0):
    lo = (100.0 - level) / 2.0
    v = np.asarray(values, dtype="float64")
    return (float(np.percentile(v, lo)), float(np.percentile(v, 100.0 - lo)))


def two_sided_bootstrap_p(diffs):
    """Two-sided bootstrap p for 'the difference is zero'.

    2 * the smaller tail mass, capped at 1. Ties count into both tails, so a
    degenerate all-zero distribution returns 1 rather than 0.
    """
    d = np.asarray(diffs, dtype="float64")
    if len(d) == 0:
        return float("nan")
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))


def cluster_bootstrap_median(values, cluster_ids, n_boot=2000, seed=0):
    """Bootstrap medians resampling CLUSTERS (participants), not rows.

    Rows from a resampled cluster enter as many times as the cluster was drawn, so a
    participant contributing several events moves together.
    """
    v = np.asarray(values, dtype="float64")
    c = pd.Series(cluster_ids).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    uniq = c.unique()
    out = []
    for _ in range(n_boot):
        draw = pd.Series(rng.choice(uniq, len(uniq), replace=True)).value_counts()
        keep = c.isin(draw.index).to_numpy()
        w = c[keep].map(draw).to_numpy()
        vals = np.repeat(v[keep], w)
        if len(vals):
            out.append(float(np.median(vals)))
    return np.asarray(out)
