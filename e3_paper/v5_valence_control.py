"""V5 - does the reflective-register association survive control for emotional valence?

Motivation: Gehrt et al. (2023, Memory) report that the narrative-identity / well-being
association disappears once the emotional valence of the narratives is controlled. The
primary model controls how much a participant wrote, not the tone they wrote in.

Convention: predictors standardised, outcome left on its original 1-5 scale, matching
e3_clsp.run(), so betas are directly comparable to the archived primary model.
Reads only the LIWC columns already authorised for V4. Aggregate output only.
"""
import os, sys, json, datetime
import pandas as pd
sys.path.insert(0, os.getcwd())
import e3_clsp as E

from paths import root as bpd_root
ROOT = bpd_root()
PERSON = ROOT + "/discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
OUT = "output/e3/v5_valence_control_SAFE.json"
SPECS = [("tone_pos_neg", ["liwc_tone_pos", "liwc_tone_neg"]),
         ("emo_pos_neg",  ["liwc_emo_pos",  "liwc_emo_neg"]),
         ("tone_composite", ["liwc_Tone"])]

def fit(d, preds, ystd=False):
    d = d.dropna(subset=[E.OUTCOME] + preds).copy()
    X = E.design(d, preds, E.CATEGORICALS)
    y = d[E.OUTCOME].astype(float)
    if ystd: y = (y - y.mean()) / y.std(ddof=1)
    f = E.ols_hc3(y, X)
    return d, E.coef_frame(f), f

def g(cf, t):
    r = cf.loc[cf["term"] == t]
    if not len(r): return None
    r = r.iloc[0]
    return {k: round(float(r[k]), 4) for k in ("beta", "se", "ci_lo", "ci_hi", "p")}

def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z",
           "seed": E.SEED, "outcome": E.OUTCOME, "person_table": PERSON,
           "convention": ("predictors standardised; outcome left on its original 1-5 "
                          "scale, matching e3_clsp.run(). Betas are outcome units per "
                          "predictor SD and are directly comparable to archived values. "
                          "SESOI 0.15 is on this same scale.")}
    if not os.path.exists(PERSON):
        res["status"] = "NO_PERSON_TABLE"
        return out(res)
    head = pd.read_csv(PERSON, nrows=0)
    have = [c for c in head.columns if c.startswith("liwc_")]
    res["liwc_columns_available"] = sorted(have)
    specs = [(nm, cs) for nm, cs in SPECS if all(c in have for c in cs)]
    if not specs:
        res["status"] = "NO_VALENCE_COLUMNS"
        return out(res)
    res["specs_available"] = [nm for nm, _ in specs]
    need = sorted({c for _, cs in specs for c in cs})
    p = pd.read_csv(PERSON, usecols=[E.PID] + need, low_memory=False)
    res["person_table_rows"] = int(len(p))

    df, win, prov = E.build(ROOT)
    preds = (list(E.BLOCKS["covariates"]) + list(E.BLOCKS["life_course"])
             + list(E.BLOCKS["language"]))
    res["primary_predictors"] = preds
    m = df.merge(p.drop_duplicates(subset=[E.PID]), on=E.PID, how="left")

    d0, cf0, f0 = fit(df, preds)
    res["archived_replication"] = {
        "n": int(f0["n"]), "r2": round(float(f0["r2"]), 4),
        "outcome_sd": round(float(d0[E.OUTCOME].std(ddof=1)), 4),
        "mean_reframe": g(cf0, "mean_reframe"), "mean_bw": g(cf0, "mean_bw"),
        "symptom_count": g(cf0, "symptom_count"),
        "earliest_onset": g(cf0, "earliest_onset"),
        "diagnosis_lag": g(cf0, "diagnosis_lag")}
    ds, cfs, fs = fit(df, preds, ystd=True)
    res["fully_standardised_for_reference"] = {
        "mean_reframe": g(cfs, "mean_reframe"), "mean_bw": g(cfs, "mean_bw")}

    res["valence_vs_language_scores"] = {}
    for c in need:
        s = m.dropna(subset=[c, "mean_reframe", "mean_bw"])
        res["valence_vs_language_scores"][c] = {
            "n": int(len(s)),
            "r_mean_reframe": round(float(s["mean_reframe"].corr(s[c])), 4),
            "r_mean_bw": round(float(s["mean_bw"].corr(s[c])), 4)}

    res["specs"] = {}
    for nm, cols in specs:
        dv, cfv, fv = fit(m, preds + cols)
        db, cfb, fb = fit(m.loc[dv.index], preds)
        b1, b0 = g(cfv, "mean_reframe"), g(cfb, "mean_reframe")
        att = {"beta_without": b0["beta"], "beta_with": b1["beta"],
               "change": round(b1["beta"] - b0["beta"], 4),
               "proportion_retained": (round(abs(b1["beta"]) / abs(b0["beta"]), 4)
                                       if b0["beta"] else None),
               "delta_r2_from_valence": round(float(fv["r2"] - fb["r2"]), 4)}
        verdict = ("not a valence artefact"
                   if abs(b1["beta"]) >= .15 and b1["p"] < .05
                   else "substantially valence"
                   if abs(b1["beta"]) < .10 or b1["p"] >= .05
                   else "partially attenuated; claim neither")
        res["specs"][nm] = {
            "valence_terms": cols,
            "with_valence": {"n": int(fv["n"]), "r2": round(float(fv["r2"]), 4),
                             "mean_reframe": b1, "mean_bw": g(cfv, "mean_bw"),
                             "symptom_count": g(cfv, "symptom_count")},
            "without_valence_same_n": {"n": int(fb["n"]),
                                       "r2": round(float(fb["r2"]), 4),
                                       "mean_reframe": b0,
                                       "mean_bw": g(cfb, "mean_bw")},
            "valence_coefficients": {c: g(cfv, c) for c in cols},
            "attenuation": att, "prespecified_verdict": verdict}
    res["headline_spec"] = specs[0][0]
    res["headline_verdict"] = res["specs"][specs[0][0]]["prespecified_verdict"]
    res["status"] = "OK"
    out(res)


def out(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
