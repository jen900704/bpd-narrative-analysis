"""V7 - analyses requested by an external review of the manuscript.

A. Linguistic distancing (Nook et al.) as a competing predictor of the outcome.
B. Suppression check: absolutist coefficient, zero-order correlations, single-score models.
C. Re-parameterisation of the life-course block (age at diagnosis + years since diagnosis).
D. Column introspection so the remaining items can be scripted without guessing.

Convention: predictors standardised, outcome on its original 1-5 scale, matching
e3_clsp.run(). Aggregate output only: coefficients, correlations and column names.
No rows, no free text, no identifiers.
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import e3_clsp as E

from paths import root as bpd_root
ROOT = bpd_root()
PERSON = ROOT + "/discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
OUT = "output/e3/v7_review_response_SAFE.json"
DIST = ["liwc_focuspast", "liwc_focuspresent", "liwc_i"]

def fit(d, preds):
    d = d.dropna(subset=[E.OUTCOME] + preds).copy()
    X = E.design(d, preds, E.CATEGORICALS)
    y = d[E.OUTCOME].astype(float)
    f = E.ols_hc3(y, X)
    return d, E.coef_frame(f), f

def g(cf, t):
    r = cf.loc[cf["term"] == t]
    if not len(r): return None
    r = r.iloc[0]
    return {k: round(float(r[k]), 4) for k in ("beta", "se", "ci_lo", "ci_hi", "p")}

def allc(cf):
    return {str(r["term"]): {k: round(float(r[k]), 4)
            for k in ("beta", "se", "ci_lo", "ci_hi", "p")} for _, r in cf.iterrows()}


def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z", "seed": E.SEED,
           "outcome": E.OUTCOME,
           "convention": "predictors standardised; outcome on its original 1-5 scale"}
    df, win, prov = E.build(ROOT)
    COV = list(E.BLOCKS["covariates"])
    LC = list(E.BLOCKS["life_course"])
    LANG = list(E.BLOCKS["language"])
    preds = COV + LC + LANG
    res["blocks"] = {"covariates": COV, "life_course": LC, "language": LANG}

    # ---- D. introspection (names only) ----
    head = pd.read_csv(PERSON, nrows=0)
    res["introspect"] = {
        "person_df_columns": sorted(map(str, df.columns)),
        "window_df_columns": sorted(map(str, win.columns))[:60],
        "person_table_non_liwc_columns": sorted(
            c for c in map(str, head.columns) if not c.startswith("liwc_"))[:200],
        "distancing_cols_present": [c for c in DIST if c in head.columns]}

    # ---- B. suppression check ----
    d0, cf0, f0 = fit(df, preds)
    res["primary_full_coefficients"] = {"n": int(f0["n"]), "r2": round(float(f0["r2"]), 4),
                                        "terms": allc(cf0)}
    only_r = COV + LC + [c for c in LANG if c != "mean_bw"]
    only_b = COV + LC + [c for c in LANG if c != "mean_reframe"]
    dr, cfr, fr = fit(df.loc[d0.index], only_r)
    db, cfb, fb = fit(df.loc[d0.index], only_b)
    res["single_score_models"] = {
        "reflective_only": {"n": int(fr["n"]), "r2": round(float(fr["r2"]), 4),
                            "mean_reframe": g(cfr, "mean_reframe")},
        "absolutist_only": {"n": int(fb["n"]), "r2": round(float(fb["r2"]), 4),
                            "mean_bw": g(cfb, "mean_bw")},
        "both_primary": {"mean_reframe": g(cf0, "mean_reframe"),
                         "mean_bw": g(cf0, "mean_bw")}}

    # zero-order correlations among focal variables and the outcome
    zvars = [E.OUTCOME] + [c for c in preds if c != "gender_collapsed"]
    z = d0[zvars].apply(pd.to_numeric, errors="coerce")
    cm = z.corr()
    res["zero_order_correlations"] = {
        a: {b: round(float(cm.loc[a, b]), 3) for b in zvars} for a in zvars}

    # ---- A. linguistic distancing as a competing predictor ----
    have = [c for c in DIST if c in pd.read_csv(PERSON, nrows=0).columns]
    if not have:
        res["distancing"] = {"status": "COLUMNS_MISSING"}
    else:
        p = pd.read_csv(PERSON, usecols=[E.PID] + have, low_memory=False)
        m = df.merge(p.drop_duplicates(subset=[E.PID]), on=E.PID, how="left")
        s = m.dropna(subset=have + ["mean_reframe", "mean_bw", E.OUTCOME])
        zz = lambda v: (v - v.mean()) / v.std(ddof=1)
        # Nook-style composite: more past focus, less present focus, less first-person
        m["ling_distance"] = (zz(m["liwc_focuspast"]) - zz(m["liwc_focuspresent"])
                              - zz(m["liwc_i"]))
        res["distancing"] = {"columns": have, "n_with_liwc": int(len(s)),
            "correlations": {c: {"r_mean_reframe": round(float(s["mean_reframe"].corr(s[c])), 4),
                                 "r_mean_bw": round(float(s["mean_bw"].corr(s[c])), 4),
                                 "r_outcome": round(float(s[E.OUTCOME].corr(s[c])), 4)}
                             for c in have}}
        sc = m.dropna(subset=["ling_distance", "mean_reframe", E.OUTCOME])
        res["distancing"]["composite_correlations"] = {
            "r_mean_reframe": round(float(sc["ling_distance"].corr(sc["mean_reframe"])), 4),
            "r_mean_bw": round(float(sc["ling_distance"].corr(sc["mean_bw"])), 4),
            "r_outcome": round(float(sc["ling_distance"].corr(sc[E.OUTCOME])), 4)}

        dv, cfv, fv = fit(m, preds + have)
        dc, cfc, fc = fit(m, preds + ["ling_distance"])
        dn, cfn, fn = fit(m.loc[dv.index], preds)
        # can LIWC distancing stand in for the register score?
        swap = COV + LC + ["mean_bw", "ling_distance"]
        ds, cfs, fs = fit(m.loc[dv.index], swap)
        res["distancing"]["models"] = {
            "baseline_same_n": {"n": int(fn["n"]), "r2": round(float(fn["r2"]), 4),
                                "mean_reframe": g(cfn, "mean_reframe")},
            "plus_three_liwc": {"n": int(fv["n"]), "r2": round(float(fv["r2"]), 4),
                                "mean_reframe": g(cfv, "mean_reframe"),
                                "mean_bw": g(cfv, "mean_bw"),
                                "liwc_terms": {c: g(cfv, c) for c in have}},
            "plus_composite": {"n": int(fc["n"]), "r2": round(float(fc["r2"]), 4),
                               "mean_reframe": g(cfc, "mean_reframe"),
                               "ling_distance": g(cfc, "ling_distance")},
            "composite_instead_of_reflective": {
                "n": int(fs["n"]), "r2": round(float(fs["r2"]), 4),
                "ling_distance": g(cfs, "ling_distance")}}
        b0 = g(cfn, "mean_reframe")["beta"]
        b1 = g(cfv, "mean_reframe")["beta"]
        res["distancing"]["verdict_inputs"] = {
            "beta_without_liwc": b0, "beta_with_three_liwc": b1,
            "proportion_retained": round(abs(b1) / abs(b0), 4) if b0 else None,
            "delta_r2": round(float(fv["r2"] - fn["r2"]), 4)}

    # ---- C. re-parameterisation of the life-course block ----
    r = df.copy()
    r["age_at_diagnosis"] = r["earliest_onset"] + r["diagnosis_lag"]
    r["years_since_diagnosis"] = r["current_age_midpoint"] - r["age_at_diagnosis"]
    alt = [c for c in COV if c != "current_age_midpoint"] + \
          ["current_age_midpoint", "earliest_onset", "years_since_diagnosis",
           "symptom_count"] + LANG
    try:
        da, cfa, fa = fit(r, alt)
        res["reparameterised_life_course"] = {
            "note": ("current_age = earliest_onset + diagnosis_lag + years_since_diagnosis, "
                     "so with age and onset held fixed the diagnosis-lag coefficient is the "
                     "negative of the years-since-diagnosis coefficient"),
            "n": int(fa["n"]), "r2": round(float(fa["r2"]), 4),
            "years_since_diagnosis": g(cfa, "years_since_diagnosis"),
            "earliest_onset": g(cfa, "earliest_onset"),
            "mean_reframe": g(cfa, "mean_reframe"),
            "primary_diagnosis_lag_for_comparison": g(cf0, "diagnosis_lag")}
        dd = r.dropna(subset=["age_at_diagnosis", "years_since_diagnosis"])
        res["reparameterised_life_course"]["descriptives"] = {
            "age_at_diagnosis_mean": round(float(dd["age_at_diagnosis"].mean()), 2),
            "age_at_diagnosis_sd": round(float(dd["age_at_diagnosis"].std(ddof=1)), 2),
            "years_since_diagnosis_mean": round(float(dd["years_since_diagnosis"].mean()), 2),
            "years_since_diagnosis_sd": round(float(dd["years_since_diagnosis"].std(ddof=1)), 2),
            "n_negative_years_since_diagnosis": int((dd["years_since_diagnosis"] < 0).sum())}
    except Exception as ex:
        res["reparameterised_life_course"] = {"status": "FAILED", "error": str(ex)[:300]}

    res["status"] = "OK"
    out(res)


def out(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
