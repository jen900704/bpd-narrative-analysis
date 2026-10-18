"""V9 - analyses demanded by peer review.

A. Introspection needed to build the domain -> outcome-item mapping.
B. Censoring-floor subgroup: how do floor participants differ?
C. Reliability of the register score by word-count tertile.
D. Alternative onset operationalisations; diagnosis-lag censoring sensitivity.

Aggregate output only. No rows, no free text, no identifiers.
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import e3_clsp as E

from paths import root as bpd_root
ROOT = bpd_root()
PERSON = ROOT + "/discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
EVENTS = ROOT + "/discovery_analysis/intermediate_PRIVATE/event_table_PRIVATE.csv"
OUT = "output/e3/v9_review_analyses_SAFE.json"


def fit(d, preds):
    d = d.dropna(subset=[E.OUTCOME] + preds).copy()
    X = E.design(d, preds, E.CATEGORICALS)
    y = d[E.OUTCOME].astype(float)
    f = E.ols_hc3(y, X)
    return d, E.coef_frame(f), f


def g(cf, t):
    r = cf.loc[cf["term"] == t]
    if not len(r):
        return None
    r = r.iloc[0]
    return {k: round(float(r[k]), 4) for k in ("beta", "se", "ci_lo", "ci_hi", "p")}


def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z", "seed": E.SEED}
    df, win, prov = E.build(ROOT)
    COV = list(E.BLOCKS["covariates"]); LC = list(E.BLOCKS["life_course"])
    LANG = list(E.BLOCKS["language"]); preds = COV + LC + LANG

    # ---- A. introspection for the domain-matched test ----
    head = pd.read_csv(PERSON, nrows=0)
    res["introspect"] = {
        "window_question_ids": sorted(map(str, win["question_id"].dropna().unique())),
        "windows_per_question": {str(k): int(v) for k, v in
                                 win["question_id"].value_counts().items()},
        "person_Q_columns": sorted(c for c in map(str, head.columns)
                                   if c.startswith("Q") or "_30d" in c or "item" in c.lower())[:120],
        "event_table_exists": os.path.exists(EVENTS)}
    if os.path.exists(EVENTS):
        eh = pd.read_csv(EVENTS, nrows=0)
        res["introspect"]["event_columns"] = sorted(map(str, eh.columns))[:60]

    # ---- B. censoring-floor subgroup ----
    d0, cf0, f0 = fit(df, preds)
    d0 = d0.copy()
    d0["floor"] = (d0["earliest_onset"] <= 5).astype(int)
    cmp_vars = [E.OUTCOME, "earliest_onset", "diagnosis_lag", "symptom_count",
                "mean_reframe", "mean_bw", "current_age_midpoint",
                "PAST_narrative_word_count_total", "sbert_n_windows"]
    B = {}
    for v in cmp_vars:
        a = d0.loc[d0["floor"] == 1, v].astype(float)
        b = d0.loc[d0["floor"] == 0, v].astype(float)
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        B[v] = {"floor_mean": round(float(a.mean()), 3), "rest_mean": round(float(b.mean()), 3),
                "d": round(float((a.mean()-b.mean())/sp), 3) if sp else None}
    res["censoring_floor"] = {"n_floor": int(d0["floor"].sum()),
                              "n_rest": int((1-d0["floor"]).sum()), "comparisons": B}

    # ---- C. reliability of the register score by word-count tertile ----
    w = win.dropna(subset=["sbert_reframing_similarity", "question_id"]).copy()
    w = w.rename(columns={E.PID: "pid", "sbert_reframing_similarity": "val"})
    mod = w.groupby(["pid", "question_id"], as_index=False)["val"].mean()
    wc = df[[E.PID, "PAST_narrative_word_count_total"]].rename(columns={E.PID: "pid"})
    mod = mod.merge(wc, on="pid", how="left").dropna()
    mod["tert"] = pd.qcut(mod["PAST_narrative_word_count_total"], 3,
                          labels=["low", "mid", "high"])

    def icc1k(gf):
        grp = gf.groupby("pid")["val"]; k = grp.size(); n = len(k)
        if n < 10 or k.sum() == n:
            return None
        kbar = (k.sum() - (k ** 2).sum() / k.sum()) / (n - 1)
        gm = gf["val"].mean()
        msb = (k * (grp.mean() - gm) ** 2).sum() / (n - 1)
        msw = grp.apply(lambda v: ((v - v.mean()) ** 2).sum()).sum() / (k.sum() - n)
        i1 = (msb - msw) / (msb + (kbar - 1) * msw)
        return {"icc_single": round(float(i1), 4),
                "icc_1k": round(float(i1 * kbar / (1 + (kbar - 1) * i1)), 4),
                "mean_modules": round(float(kbar), 2), "n_persons": int(n)}
    res["reliability_by_word_count"] = {
        str(t): icc1k(mod.loc[mod["tert"] == t, ["pid", "val"]]) for t in ("low", "mid", "high")}
    res["reliability_by_word_count"]["tertile_word_count_bounds"] = [
        round(float(x), 1) for x in mod["PAST_narrative_word_count_total"].quantile([0, .333, .667, 1])]

    # ---- D. onset alternatives and lag censoring ----
    alt = {}
    for nm, col in [("primary_min_onset", "earliest_onset"),
                    ("latest_onset_direct", "latest_onset_direct"),
                    ("proxy_earliest_onset", "proxy_earliest_onset")]:
        if col not in df.columns:
            alt[nm] = "COLUMN_ABSENT"; continue
        p2 = [c for c in preds if c != "earliest_onset"] + [col]
        try:
            da, cfa, fa = fit(df, p2)
            alt[nm] = {"n": int(fa["n"]), "onset_term": g(cfa, col),
                       "mean_reframe": g(cfa, "mean_reframe"),
                       "symptom_count": g(cfa, "symptom_count")}
        except Exception as ex:
            alt[nm] = {"status": "FAILED", "error": str(ex)[:200]}
    res["onset_alternatives"] = alt

    # lag inherits the age-5 floor: refit excluding floor participants
    dfl = df.loc[df["earliest_onset"] > 5]
    try:
        dl, cfl, fl = fit(dfl, preds)
        res["lag_censoring_sensitivity"] = {
            "note": ("diagnosis lag = age at diagnosis minus earliest onset, so the 27.4% at the "
                     "age-5 floor have an over-estimated lag; this refits without them"),
            "n": int(fl["n"]), "diagnosis_lag": g(cfl, "diagnosis_lag"),
            "earliest_onset": g(cfl, "earliest_onset"),
            "mean_reframe": g(cfl, "mean_reframe"),
            "symptom_count": g(cfl, "symptom_count")}
    except Exception as ex:
        res["lag_censoring_sensitivity"] = {"status": "FAILED", "error": str(ex)[:200]}

    # ---- E. does the register score survive controlling how much was written, nonlinearly? ----
    d2 = df.copy()
    d2["log_words"] = np.log1p(d2["PAST_narrative_word_count_total"].astype(float))
    try:
        dw, cfw, fw = fit(d2, preds + ["log_words"])
        res["log_word_count_control"] = {"n": int(fw["n"]),
                                         "mean_reframe": g(cfw, "mean_reframe"),
                                         "log_words": g(cfw, "log_words")}
    except Exception as ex:
        res["log_word_count_control"] = {"status": "FAILED", "error": str(ex)[:200]}

    res["status"] = "OK"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
