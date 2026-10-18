"""E3 on CLSP — one mutually adjusted life-course + language model.

Self-contained: no imports outside numpy/pandas/scipy. Reads the two existing PRIVATE
tables by their actual column names, writes SAFE aggregate outputs only.

Inputs (PRIVATE, read-only, never printed):
  discovery_analysis/intermediate_PRIVATE/phase1_3/phase1_3_analysis_table_PRIVATE.csv
  discovery_analysis/intermediate_PRIVATE/window_table_augmented_PRIVATE.csv

Outputs (SAFE): coefficients, hierarchical blocks, equivalence curve, VIF,
reliability, MI sensitivity, E4 time-since-event, flow, summary.

Confirmatory family (H1-H3) = symptom_count, sbert_reframing_mean, earliest_onset,
diagnosis_lag_direct on the primary outcome, BH-FDR together.
SESOI = .15 (revised from .10 before any real-data run; see plan v0.2 section 0).

Usage:
  export BPD_ROOT=/path/to/project_root
  python e3_clsp.py --root "$BPD_ROOT" \
      --out discovery_analysis/output/e3 [--synthetic-check DIR]
"""
from __future__ import annotations

import argparse, json, hashlib, os, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260905
SESOI = 0.15
SESOI_CURVE = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
FDR_ALPHA = 0.05
MIN_DELTA_R2_LANG = 0.01
MI_M, MI_K = 20, 5

# --- confirmed column map (verified against the real headers on 2026-09-05) ---
PID = "ResponseId"
FOCAL = {
    "earliest_onset": "earliest_onset",
    "diagnosis_lag": "diagnosis_lag_direct",
    "symptom_count": "symptom_count",
    "mean_reframe": "sbert_reframing_mean",
    "mean_bw": "sbert_blackwhite_mean",
}
COVARS = ["current_age_midpoint", "gender_collapsed",
          "PAST_narrative_word_count_total", "sbert_n_windows"]
CATEGORICALS = ["gender_collapsed"]
OUTCOME = "overall_negative_primary"
SECONDARY = ["negative_behaviors_primary", "thoughts_feelings_primary",
             "positive_behaviors_primary"]
CONFIRMATORY = ["earliest_onset", "diagnosis_lag", "symptom_count", "mean_reframe"]
BLOCKS = {"covariates": COVARS,
          "life_course": ["earliest_onset", "diagnosis_lag", "symptom_count"],
          "language": ["mean_reframe", "mean_bw"]}

W_PID, W_Q = "ResponseId", "question_id"
W_REF, W_BW = "sbert_reframing_similarity", "sbert_blackwhite_similarity"
W_YEARS = "years_ago"
COPING_PROMPTS = ["Q64", "Q68", "Q53"]


# ------------------------------------------------------------------ stats core
def ols_hc3(y, X):
    Xv = np.asarray(X, float); yv = np.asarray(y, float)
    n, k = Xv.shape
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    h = np.clip(np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv), 0, 1 - 1e-10)
    meat = (Xv * ((resid / (1 - h)) ** 2)[:, None]).T @ Xv
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    dfr = n - k
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {"terms": list(X.columns), "beta": beta, "se": se, "cov": cov,
            "t": beta / se, "p": 2 * stats.t.sf(np.abs(beta / se), dfr),
            "n": n, "df_resid": dfr, "resid": resid,
            "r2": 1 - ss_res / ss_tot,
            "adj_r2": 1 - (1 - (1 - ss_res / ss_tot)) * 0 - (ss_res / dfr) / (ss_tot / (n - 1))}


def design(df, preds, cats):
    parts = []
    for v in preds:
        if v in cats:
            parts.append(pd.get_dummies(df[v].astype("category"), prefix=v,
                                        drop_first=True, dtype=float))
        else:
            s = df[v].astype(float); sd = s.std(ddof=1)
            parts.append(((s - s.mean()) / sd if sd > 0 else s * 0.0).rename(v))
    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)
    return X


def coef_frame(fit, conf=0.95):
    crit = stats.t.ppf(0.5 + conf / 2, fit["df_resid"])
    return pd.DataFrame({"term": fit["terms"], "beta": fit["beta"], "se": fit["se"],
                         "t": fit["t"], "p": fit["p"],
                         "ci_lo": fit["beta"] - crit * fit["se"],
                         "ci_hi": fit["beta"] + crit * fit["se"]})


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def tost(beta, se, dfr, sesoi, alpha=0.05):
    pl = stats.t.sf((beta + sesoi) / se, dfr)
    pu = stats.t.cdf((beta - sesoi) / se, dfr)
    crit = stats.t.ppf(1 - alpha, dfr)
    lo, hi = beta - crit * se, beta + crit * se
    return {"beta": beta, "se": se, "sesoi": sesoi, "ci90_lo": lo, "ci90_hi": hi,
            "p_tost": max(pl, pu), "equivalent": bool(lo > -sesoi and hi < sesoi)}


def vif_table(X):
    cols = [c for c in X.columns if c != "const"]; out = []
    for c in cols:
        others = [o for o in cols if o != c]
        Xo = np.column_stack([np.ones(len(X)), X[others].to_numpy(float)])
        y = X[c].to_numpy(float)
        b = np.linalg.pinv(Xo.T @ Xo) @ Xo.T @ y
        r = y - Xo @ b
        sst = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (r @ r) / sst if sst > 0 else 0.0
        out.append({"term": c, "vif": float(1 / max(1 - r2, 1e-12))})
    return pd.DataFrame(out)


def delta_r2(y, Xs, Xb):
    fs, fb = ols_hc3(y, Xs), ols_hc3(y, Xb)
    ss_s = float(fs["resid"] @ fs["resid"]); ss_b = float(fb["resid"] @ fb["resid"])
    q = Xb.shape[1] - Xs.shape[1]; dfb = fb["df_resid"]
    F = ((ss_s - ss_b) / q) / (ss_b / dfb) if q > 0 and ss_b > 0 else np.nan
    return {"r2_small": fs["r2"], "r2_big": fb["r2"], "delta_r2": fb["r2"] - fs["r2"],
            "df1": q, "df2": dfb, "F": F,
            "p": float(stats.f.sf(F, q, dfb)) if np.isfinite(F) else np.nan}


def alpha_omega(items: pd.DataFrame):
    x = items.dropna()
    if x.shape[1] < 2 or len(x) < 20:
        return np.nan, np.nan
    k = x.shape[1]
    a = k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))
    z = (x - x.mean()) / x.std(ddof=1)
    R = np.corrcoef(z.values, rowvar=False)
    w, v = np.linalg.eigh(R)
    lam = v[:, -1] * np.sqrt(max(w[-1], 0))
    if lam.sum() < 0:
        lam = -lam
    psi = np.clip(1 - lam ** 2, 1e-9, None)
    om = lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum())
    return float(a), float(om)


def pmm(df, targets, preds, m=MI_M, k=MI_K, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    for _ in range(m):
        d = df.copy()
        for t in targets:
            if t not in d or not d[t].isna().any():
                continue
            P = [c for c in preds if c != t and c in d.columns]
            base = d[P].astype(float); base = base.fillna(base.mean())
            obs = d[t].notna().to_numpy()
            if obs.sum() < max(10, k + 1):
                continue
            X = np.column_stack([np.ones(len(d)), base.to_numpy(float)])
            y = d[t].to_numpy(float)
            idx = rng.choice(np.flatnonzero(obs), obs.sum(), replace=True)
            b = np.linalg.pinv(X[idx].T @ X[idx]) @ X[idx].T @ y[idx]
            yh = X @ b; donors = np.flatnonzero(obs)
            for i in np.flatnonzero(~obs):
                near = donors[np.argsort(np.abs(yh[donors] - yh[i]))[:k]]
                d.iloc[i, d.columns.get_loc(t)] = y[rng.choice(near)]
        out.append(d)
    return out


def rubin(betas, ses, terms):
    B = np.vstack(betas); U = np.vstack(ses) ** 2; m = B.shape[0]
    qb = B.mean(0); ub = U.mean(0); bv = B.var(0, ddof=1) if m > 1 else np.zeros_like(qb)
    tot = ub + (1 + 1 / m) * bv; se = np.sqrt(tot)
    lam = np.clip(np.where(tot > 0, (1 + 1 / m) * bv / tot, 0.0), 1e-9, 1 - 1e-9)
    dfm = (m - 1) / lam ** 2
    crit = stats.t.ppf(0.975, dfm)
    return pd.DataFrame({"term": terms, "beta": qb, "se": se, "t": qb / se,
                         "p": 2 * stats.t.sf(np.abs(qb / se), dfm), "fmi": lam,
                         "ci_lo": qb - crit * se, "ci_hi": qb + crit * se})


# ------------------------------------------------------------------ pipeline
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def build(root, synthetic=None):
    if synthetic:
        p3 = Path(synthetic) / "phase1_3.csv"
        wt = Path(synthetic) / "windows.csv"
    else:
        root = Path(root)
        p3 = root / "discovery_analysis/intermediate_PRIVATE/phase1_3/phase1_3_analysis_table_PRIVATE.csv"
        wt = root / "discovery_analysis/intermediate_PRIVATE/window_table_augmented_PRIVATE.csv"
    part = pd.read_csv(p3, low_memory=False)
    # window table: never read the text columns
    cols = pd.read_csv(wt, nrows=0).columns
    use = [c for c in [W_PID, W_Q, W_REF, W_BW, W_YEARS] if c in cols]
    win = pd.read_csv(wt, usecols=use, low_memory=False)
    prov = {"phase1_3": {"path": str(p3), "sha256_16": sha16(p3), "rows": len(part)},
            "windows": {"path": str(wt), "sha256_16": sha16(wt), "rows": len(win),
                        "columns_read": use}}

    g = win.groupby(W_PID)
    agg = pd.DataFrame({"sbert_reframing_mean": g[W_REF].mean(),
                        "sbert_blackwhite_mean": g[W_BW].mean(),
                        "sbert_n_windows": g.size()})
    df = part.merge(agg, left_on=PID, right_index=True, how="left")
    df = df.rename(columns={v: k for k, v in FOCAL.items() if v != k})
    return df, win, prov


def run(root, out_dir, synthetic=None, verbose=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df, win, prov = build(root, synthetic)
    preds = list(FOCAL.keys()) + COVARS

    flow = {"rows_phase1_3": int(prov["phase1_3"]["rows"]),
            "rows_windows": int(prov["windows"]["rows"])}
    step = df.copy()
    for lbl, cols in [("outcome", [OUTCOME]), ("language", ["mean_reframe", "mean_bw"]),
                      ("onset_breadth", ["earliest_onset", "symptom_count"]),
                      ("diagnosis_lag", ["diagnosis_lag"]), ("covariates", COVARS)]:
        step = step.dropna(subset=[c for c in cols if c in step.columns])
        flow[f"after_{lbl}"] = int(len(step))
    ana = df.dropna(subset=[OUTCOME] + [p for p in preds if p in df.columns]).copy()
    flow["analytic_n"] = int(len(ana))

    # ---- primary model
    X = design(ana, preds, CATEGORICALS)
    y = ana[OUTCOME].astype(float)
    fit = ols_hc3(y, X)
    cf = coef_frame(fit)
    mask = cf["term"].isin(CONFIRMATORY)
    cf["q_bh"] = np.nan
    cf.loc[mask, "q_bh"] = bh_fdr(cf.loc[mask, "p"].to_numpy())
    cf["confirmatory"] = mask
    cf.round(6).to_csv(out / "e3_primary_coefficients_SAFE.csv", index=False)
    vif_table(X).round(3).to_csv(out / "e3_vif_SAFE.csv", index=False)

    # ---- blocks
    rows, cum = [], []
    prev = pd.DataFrame({"const": np.ones(len(ana))}, index=ana.index)
    for name, blk in BLOCKS.items():
        cum += [b for b in blk if b in ana.columns]
        Xb = design(ana, cum, CATEGORICALS)
        rows.append({"block": name, "predictors": ", ".join(blk), **delta_r2(y, prev, Xb)})
        prev = Xb
    blocks = pd.DataFrame(rows)
    blocks.round(6).to_csv(out / "e3_blocks_SAFE.csv", index=False)
    d_lang = float(blocks.loc[blocks["block"] == "language", "delta_r2"].iloc[0])

    # ---- equivalence
    eq, curve = [], []
    for t in ("earliest_onset", "diagnosis_lag"):
        r = cf.loc[cf["term"] == t]
        if not len(r):
            continue
        b, se = float(r["beta"].iloc[0]), float(r["se"].iloc[0])
        eq.append({"term": t, **tost(b, se, fit["df_resid"], SESOI)})
        for s in SESOI_CURVE:
            curve.append({"term": t, **tost(b, se, fit["df_resid"], s)})
    eq = pd.DataFrame(eq); curve = pd.DataFrame(curve)
    eq.round(6).to_csv(out / "e3_equivalence_SAFE.csv", index=False)
    curve.round(6).to_csv(out / "e3_equivalence_curve_SAFE.csv", index=False)

    # ---- reliability
    rel = []
    for nm, pat in [("Q92 thoughts/feelings", "Q92_%d_num"),
                    ("Q93 negative behaviours", "Q93_%d_num"),
                    ("Q109 positive behaviours", "Q109_%d_num")]:
        k = {"Q92 thoughts/feelings": 8, "Q93 negative behaviours": 4,
             "Q109 positive behaviours": 3}[nm]
        cols = [pat % i for i in range(1, k + 1)]
        if all(c in df.columns for c in cols):
            a, om = alpha_omega(df[cols])
            rel.append({"scale": nm, "k": k, "n": int(df[cols].dropna().shape[0]),
                        "alpha": a, "omega_total": om})
    comp = [f"Q92_{i}_num" for i in range(1, 9)] + [f"Q93_{i}_num" for i in range(1, 5)]
    if all(c in df.columns for c in comp):
        a, om = alpha_omega(df[comp])
        rel.append({"scale": "Q92+Q93 composite (primary)", "k": 12,
                    "n": int(df[comp].dropna().shape[0]), "alpha": a, "omega_total": om})
    pd.DataFrame(rel).round(4).to_csv(out / "e3_reliability_SAFE.csv", index=False)

    # ---- MI sensitivity
    mi = None
    tgt = [c for c in ("diagnosis_lag",) if c in df.columns and df[c].isna().any()]
    if tgt:
        base = df.dropna(subset=[c for c in [OUTCOME, "mean_reframe", "mean_bw",
                                             "earliest_onset", "symptom_count"] + COVARS
                                 if c in df.columns]).copy()
        ip = [c for c in ["earliest_onset", "symptom_count", "current_age_midpoint",
                          "PAST_narrative_word_count_total", "sbert_n_windows",
                          "mean_reframe", OUTCOME] if c in base.columns]
        bs, ss, terms = [], [], None
        for d in pmm(base, tgt, ip):
            Xi = design(d, preds, CATEGORICALS)
            f = ols_hc3(d[OUTCOME].astype(float), Xi)
            terms = terms or f["terms"]
            bs.append(f["beta"]); ss.append(f["se"])
        mi = rubin(bs, ss, terms)
        mi["n_per_imputation"] = len(base)
        mi.round(6).to_csv(out / "e3_mi_coefficients_SAFE.csv", index=False)

    # ---- sensitivity
    sens = []

    def rec(label, frame):
        for t in CONFIRMATORY:
            r = frame.loc[frame["term"] == t]
            if len(r):
                r = r.iloc[0]
                sens.append({"analysis": label, "term": t, "beta": r["beta"],
                             "se": r["se"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "p": r["p"]})
    rec("primary (complete case)", cf)

    wnc = win[~win[W_Q].isin(COPING_PROMPTS)]
    if 0 < len(wnc) < len(win):
        g = wnc.groupby(W_PID)
        alt = (df.drop(columns=["mean_reframe", "mean_bw"])
                 .merge(pd.DataFrame({"mean_reframe": g[W_REF].mean(),
                                      "mean_bw": g[W_BW].mean()}),
                        left_on=PID, right_index=True, how="left"))
        a2 = alt.dropna(subset=[OUTCOME] + [p for p in preds if p in alt.columns])
        if len(a2) > 60:
            rec("language excluding coping prompts", coef_frame(
                ols_hc3(a2[OUTCOME].astype(float), design(a2, preds, CATEGORICALS))))

    if "earliest_censored_low" in df.columns:
        a3 = ana[ana["earliest_censored_low"].fillna(0) == 0]
        if len(a3) > 60:
            rec("excluding left-censored onset", coef_frame(
                ols_hc3(a3[OUTCOME].astype(float), design(a3, preds, CATEGORICALS))))

    for so in SECONDARY:
        if so not in df.columns:
            continue
        a4 = df.dropna(subset=[so] + [p for p in preds if p in df.columns])
        if len(a4) > 60:
            rec(f"secondary outcome: {so}", coef_frame(
                ols_hc3(a4[so].astype(float), design(a4, preds, CATEGORICALS))))
    pd.DataFrame(sens).round(6).to_csv(out / "e3_sensitivity_SAFE.csv", index=False)

    # ---- E4 time since event
    e4 = []
    if W_YEARS in win.columns:
        for thr in [None, 10, 5]:
            sub = win if thr is None else win[win[W_YEARS] >= thr]
            if len(sub) < 200:
                continue
            g = sub.groupby(W_PID)
            d = (df.drop(columns=["mean_reframe", "mean_bw"])
                   .merge(pd.DataFrame({"mean_reframe": g[W_REF].mean(),
                                        "mean_bw": g[W_BW].mean()}),
                          left_on=PID, right_index=True, how="inner"))
            d = d.dropna(subset=[OUTCOME] + [p for p in preds if p in d.columns])
            if len(d) < 60:
                continue
            c2 = coef_frame(ols_hc3(d[OUTCOME].astype(float), design(d, preds, CATEGORICALS)))
            r = c2.loc[c2["term"] == "mean_reframe"]
            if len(r):
                r = r.iloc[0]
                e4.append({"threshold_years": "all" if thr is None else thr,
                           "n_participants": int(len(d)), "n_windows": int(len(sub)),
                           "beta_mean_reframe": r["beta"], "se": r["se"],
                           "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
    pd.DataFrame(e4).round(6).to_csv(out / "e4_time_since_event_SAFE.csv", index=False)

    # ---- summary
    def g_(t, col="beta"):
        s = cf.loc[cf["term"] == t, col]
        v = float(s.iloc[0]) if len(s) else None
        return None if v is not None and pd.isna(v) else v

    summary = {
        "script": "e3_clsp.py",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "synthetic_mode": bool(synthetic),
        "inputs": prov, "flow": flow,
        "outcome": OUTCOME, "analytic_n": int(len(ana)),
        "model_r2": fit["r2"],
        "coefficients": {t: {"beta": g_(t), "se": g_(t, "se"),
                             "ci": [g_(t, "ci_lo"), g_(t, "ci_hi")],
                             "p": g_(t, "p"), "q_bh": g_(t, "q_bh")}
                         for t in CONFIRMATORY},
        "delta_r2_language": d_lang,
        "equivalence_sesoi_primary": SESOI,
        "equivalence": eq.to_dict("records") if len(eq) else [],
        "smallest_equivalence_bound": (
            curve[curve["equivalent"]].groupby("term")["sesoi"].min().to_dict()
            if len(curve) else {}),
        "gates": {
            "H1_breadth": bool(g_("symptom_count") and g_("symptom_count") > 0
                               and (g_("symptom_count", "q_bh") or 1) < FDR_ALPHA),
            "H2_reframe": bool(g_("mean_reframe") and g_("mean_reframe") < 0
                               and (g_("mean_reframe", "q_bh") or 1) < FDR_ALPHA
                               and d_lang >= MIN_DELTA_R2_LANG),
            "H3_both_equivalent": bool(len(eq) and all(eq["equivalent"])),
        },
        "max_vif": float(vif_table(X)["vif"].max()),
    }
    (out / "e3_summary_SAFE.json").write_text(json.dumps(summary, indent=2, default=str))

    if verbose:
        print(f"\n{'SYNTHETIC CHECK' if synthetic else 'REAL DATA'} — outcome {OUTCOME}"
              f"   analytic n = {len(ana)}   R2 = {fit['r2']:.4f}")
        print(cf[cf["term"].isin(CONFIRMATORY)][
            ["term", "beta", "se", "ci_lo", "ci_hi", "p", "q_bh"]]
            .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        print("\nHierarchical blocks:")
        print(blocks[["block", "r2_big", "delta_r2", "F", "p"]]
              .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        if len(eq):
            print(f"\nEquivalence (TOST, SESOI={SESOI}):")
            print(eq[["term", "beta", "ci90_lo", "ci90_hi", "p_tost", "equivalent"]]
                  .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
            print("smallest bound where equivalent:",
                  summary["smallest_equivalence_bound"])
        if mi is not None:
            print("\nMI sensitivity:")
            print(mi[mi["term"].isin(CONFIRMATORY)][
                ["term", "beta", "se", "ci_lo", "ci_hi", "p", "fmi"]]
                .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        if e4:
            print("\nE4 reframing coefficient by minimum years since event:")
            print(pd.DataFrame(e4)[["threshold_years", "n_participants", "n_windows",
                                    "beta_mean_reframe", "ci_lo", "ci_hi", "p"]]
                  .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        print("\nGates:", summary["gates"])
        print("flow:", flow)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("BPD_ROOT"))
    ap.add_argument("--out", default="discovery_analysis/output/e3")
    ap.add_argument("--synthetic-check")
    a = ap.parse_args()
    run(a.root, a.out, a.synthetic_check)
