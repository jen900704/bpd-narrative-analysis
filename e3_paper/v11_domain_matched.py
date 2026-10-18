"""V11 - domain-matched test of the criterion-contamination explanation.

A reviewer objection: a participant who writes about a domain in the past tense may
simply be reporting that the symptom has resolved, and the outcome is the 30-day impact
of those same domains. On that account the register association is contamination, not
narration.

Test: build a person x domain long frame in which x is the participant's mean window
score for that domain's prompt and y is that participant's 30-day impact rating for the
matched domain. Decompose into within-person and between-person components.

  contamination  -> a strong WITHIN-person effect: within one person, the domains written
                    about in a more past-tense register are the domains bothering them less
  stable style   -> the association lives BETWEEN persons and the within effect is weak

A placebo using mismatched domains within the same person bounds how much of any within
effect is domain-specific rather than a general person-level tendency.

Aggregate output only. No rows, no free text, no identifiers.
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import e3_clsp as E

from paths import root as bpd_root
ROOT = bpd_root()
PERSON = ROOT + "/discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
OUT = "output/e3/v11_domain_matched_SAFE.json"
SEED = 20260920

# narrative prompt -> matched 30-day impact item (item order as printed in the manuscript)
MAP = {"Q7": "Q92_1_num", "Q25": "Q92_2_num", "Q28": "Q92_3_num", "Q34": "Q92_4_num",
       "Q37": "Q92_5_num", "q10": "Q92_6_num", "Q17": "Q92_7_num", "Q20": "Q92_8_num",
       "Q46": "Q93_1_num", "Q53": "Q93_2_num", "Q50": "Q93_2_num",
       "Q57": "Q93_3_num", "Q60": "Q93_4_num"}


def ols_cluster(y, X, grp):
    """OLS with cluster-robust (CR1) standard errors."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gval in np.unique(grp):
        m = grp == gval
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    G = len(np.unique(grp)); n, k = X.shape
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = XtXi @ meat @ XtXi * c
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    from scipy import stats
    t = b / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=G - 1)
    return b, se, p


def rep(b, se, p, i, z=1.96):
    return {"beta": round(float(b[i]), 4), "se": round(float(se[i]), 4),
            "ci_lo": round(float(b[i] - z * se[i]), 4),
            "ci_hi": round(float(b[i] + z * se[i]), 4),
            "p": round(float(p[i]), 4)}


def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z", "seed": SEED,
           "mapping": MAP}
    df, win, prov = E.build(ROOT)
    w = win.dropna(subset=["sbert_reframing_similarity", "question_id"]).copy()
    w = w[w["question_id"].isin(MAP)]
    dom = (w.groupby([E.PID, "question_id"], as_index=False)["sbert_reframing_similarity"]
             .mean().rename(columns={"sbert_reframing_similarity": "x"}))
    items = sorted(set(MAP.values()))
    p = pd.read_csv(PERSON, usecols=[E.PID] + items, low_memory=False)
    long = dom.merge(p, on=E.PID, how="left")
    long["y"] = [r[MAP[q]] for q, (_, r) in zip(long["question_id"], long.iterrows())]
    long = long.dropna(subset=["x", "y"])
    res["frame"] = {"rows": int(len(long)), "persons": int(long[E.PID].nunique()),
                    "domains": int(long["question_id"].nunique()),
                    "rows_per_person_mean": round(float(len(long) / long[E.PID].nunique()), 2)}

    # two-way demeaning: remove person means and domain means from both x and y
    long["xp"] = long.groupby(E.PID)["x"].transform("mean")
    long["yp"] = long.groupby(E.PID)["y"].transform("mean")
    long["xd"] = long.groupby("question_id")["x"].transform("mean")
    long["yd"] = long.groupby("question_id")["y"].transform("mean")
    gx, gy = long["x"].mean(), long["y"].mean()
    long["xw"] = long["x"] - long["xp"] - long["xd"] + gx
    long["yw"] = long["y"] - long["yp"] - long["yd"] + gy
    sx = long["xw"].std(ddof=1)
    Xw = np.column_stack([np.ones(len(long)), long["xw"] / sx])
    b, se, pv = ols_cluster(long["yw"], Xw, long[E.PID].values)
    res["within_person_domain_matched"] = rep(b, se, pv, 1)
    res["within_person_domain_matched"]["note"] = (
        "person and domain means removed from both sides; outcome in item scale points per "
        "SD of the within-person deviation in domain register score; CR1 SEs clustered on "
        "participant")

    # placebo: pair each person's domain score with a DIFFERENT domain's item, same person
    rng = np.random.default_rng(SEED)
    pl = []
    for pid, gdf in long.groupby(E.PID):
        if len(gdf) < 2:
            continue
        gg = gdf.copy()
        sh = gg["y"].to_numpy().copy()
        for _ in range(20):
            rng.shuffle(sh)
            if not np.all(sh == gg["y"].to_numpy()):
                break
        gg["y_placebo"] = sh
        pl.append(gg)
    pl = pd.concat(pl, ignore_index=True)
    pl["ypp"] = pl.groupby(E.PID)["y_placebo"].transform("mean")
    pl["ypd"] = pl.groupby("question_id")["y_placebo"].transform("mean")
    pl["ypw"] = pl["y_placebo"] - pl["ypp"] - pl["ypd"] + pl["y_placebo"].mean()
    Xp = np.column_stack([np.ones(len(pl)), pl["xw"] / sx])
    bp, sep, pp = ols_cluster(pl["ypw"], Xp, pl[E.PID].values)
    res["within_person_placebo_mismatched"] = rep(bp, sep, pp, 1)

    # between-person: person mean of the domain scores against the 12-item composite
    per = long.groupby(E.PID, as_index=False).agg(xbar=("x", "mean"), ybar=("y", "mean"))
    per = per.merge(df[[E.PID, E.OUTCOME]], on=E.PID, how="inner").dropna()
    zx = (per["xbar"] - per["xbar"].mean()) / per["xbar"].std(ddof=1)
    Xb = np.column_stack([np.ones(len(per)), zx])
    bb, seb, pb = ols_cluster(per[E.OUTCOME], Xb, per[E.PID].values)
    res["between_person_unadjusted"] = rep(bb, seb, pb, 1)
    res["between_person_unadjusted"]["n"] = int(len(per))

    # variance decomposition of the domain-level register score
    ssw = float(((long["x"] - long["xp"]) ** 2).sum())
    sst = float(((long["x"] - gx) ** 2).sum())
    res["register_score_variance"] = {
        "within_person_share": round(ssw / sst, 4),
        "between_person_share": round(1 - ssw / sst, 4)}

    # per-domain matched correlation, to show it is not one domain doing the work
    pd_rows = {}
    for q, gdf in long.groupby("question_id"):
        if len(gdf) < 30:
            continue
        pd_rows[str(q)] = {"n": int(len(gdf)),
                           "r_raw": round(float(np.corrcoef(gdf["x"], gdf["y"])[0, 1]), 3)}
    res["per_domain_raw_correlation"] = pd_rows

    res["status"] = "OK"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
