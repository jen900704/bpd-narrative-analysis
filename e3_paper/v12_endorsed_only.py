"""V12 - is H1 partly true by construction?

Symptom breadth counts the domains a participant has ever endorsed. The outcome averages
30-day impact over twelve of those same domains, and a participant who never had a domain
will rate it at the floor. Part of the breadth-outcome association is therefore built in.

Test: recompute the outcome over only the domains that participant endorsed, and refit.
If breadth survives, the association is not purely constructional. If it collapses, it is.

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
OUT = "output/e3/v12_endorsed_only_outcome_SAFE.json"

MOD2ITEM = {"abandonment_worry": "Q92_1_num", "opinions_others": "Q92_2_num",
            "self_view": "Q92_3_num", "mood_swings": "Q92_4_num",
            "paranoia": "Q92_5_num", "intense_anger": "Q92_6_num",
            "emptiness": "Q92_7_num", "suicidal_feelings": "Q92_8_num",
            "abandonment_behavior": "Q93_1_num", "nssi": "Q93_2_num",
            "suicide_attempt": "Q93_2_num", "impulsivity": "Q93_3_num",
            "anger_consequences": "Q93_4_num"}
ITEMS = sorted(set(MOD2ITEM.values()))


def fit(d, preds, outcome):
    d = d.dropna(subset=[outcome] + preds).copy()
    X = E.design(d, preds, E.CATEGORICALS)
    f = E.ols_hc3(d[outcome].astype(float), X)
    return d, E.coef_frame(f), f


def g(cf, t):
    r = cf.loc[cf["term"] == t]
    if not len(r):
        return None
    r = r.iloc[0]
    return {k: round(float(r[k]), 4) for k in ("beta", "se", "ci_lo", "ci_hi", "p")}


def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z", "seed": E.SEED,
           "module_to_item": MOD2ITEM}
    df, win, prov = E.build(ROOT)
    preds = (list(E.BLOCKS["covariates"]) + list(E.BLOCKS["life_course"])
             + list(E.BLOCKS["language"]))

    ev = pd.read_csv(EVENTS, usecols=[E.PID, "module_id", "endorsement_observed",
                                      "endorsement_value_normalized"], low_memory=False)
    ev = ev[ev["module_id"].isin(MOD2ITEM)]
    res["endorsement_value_counts"] = {
        str(k): int(v) for k, v in
        ev["endorsement_value_normalized"].astype(str).value_counts().head(8).items()}
    yes = ev["endorsement_value_normalized"].astype(str).str.lower().isin(
        ["yes", "true", "1", "1.0", "endorsed"])
    ev = ev.assign(yes=yes)
    ev["item"] = ev["module_id"].map(MOD2ITEM)
    endorsed = (ev[ev["yes"]].groupby([E.PID, "item"]).size().rename("k").reset_index())

    p = pd.read_csv(PERSON, usecols=[E.PID] + ITEMS, low_memory=False)
    long = p.melt(id_vars=[E.PID], value_vars=ITEMS, var_name="item", value_name="v")
    m = long.merge(endorsed, on=[E.PID, "item"], how="left")
    m["kept"] = m["k"].notna()
    sel = m[m["kept"] & m["v"].notna()]
    eo = sel.groupby(E.PID, as_index=False).agg(endorsed_only_outcome=("v", "mean"),
                                                items_kept=("v", "size"))
    res["items_kept"] = {"mean": round(float(eo["items_kept"].mean()), 2),
                         "median": int(eo["items_kept"].median()),
                         "min": int(eo["items_kept"].min()),
                         "max": int(eo["items_kept"].max()),
                         "n_persons": int(len(eo))}
    d = df.merge(eo, on=E.PID, how="left")
    res["outcome_comparison"] = {
        "primary_mean": round(float(d[E.OUTCOME].mean()), 4),
        "endorsed_only_mean": round(float(d["endorsed_only_outcome"].mean()), 4),
        "correlation": round(float(d[[E.OUTCOME, "endorsed_only_outcome"]].dropna()
                                   .corr().iloc[0, 1]), 4)}


    base = fit(d, preds, E.OUTCOME)
    res["primary_for_reference"] = {
        "n": int(base[2]["n"]), "r2": round(float(base[2]["r2"]), 4),
        "symptom_count": g(base[1], "symptom_count"),
        "mean_reframe": g(base[1], "mean_reframe"),
        "earliest_onset": g(base[1], "earliest_onset"),
        "diagnosis_lag": g(base[1], "diagnosis_lag")}
    try:
        alt = fit(d, preds, "endorsed_only_outcome")
        res["endorsed_only_model"] = {
            "n": int(alt[2]["n"]), "r2": round(float(alt[2]["r2"]), 4),
            "symptom_count": g(alt[1], "symptom_count"),
            "mean_reframe": g(alt[1], "mean_reframe"),
            "mean_bw": g(alt[1], "mean_bw"),
            "earliest_onset": g(alt[1], "earliest_onset"),
            "diagnosis_lag": g(alt[1], "diagnosis_lag"),
            "outcome_sd": round(float(alt[0]["endorsed_only_outcome"].std(ddof=1)), 4)}
        b0 = res["primary_for_reference"]["symptom_count"]["beta"]
        b1 = res["endorsed_only_model"]["symptom_count"]["beta"]
        res["breadth_attenuation"] = {
            "beta_primary_outcome": b0, "beta_endorsed_only_outcome": b1,
            "proportion_retained": round(abs(b1) / abs(b0), 4) if b0 else None}
    except Exception as ex:
        res["endorsed_only_model"] = {"status": "FAILED", "error": str(ex)[:300]}

    res["status"] = "OK"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
