"""R9 - three specification fixes found while assembling the Methods section.

A. COPING-PROMPT SET IS WRONG IN THE FROZEN CODE.
   e3_clsp.py line 62 sets COPING_PROMPTS = ["Q64", "Q68", "Q53"], but the window join
   coverage table lists the LIFE_STRATEGY_NARRATIVE questions as Q64, Q68 and Q71. Q53
   is a PAST_NARRATIVE. So the "language excluding coping prompts" sensitivity dropped
   885 past-narrative windows and KEPT 617 life-strategy windows - the opposite of its
   stated intent. Rerun with the correct set and report both.

B. A TREATMENT-RELATED ITEM DOES EXIST.
   The audit said no treatment-exposure item exists. There is a life-strategy module
   "treatment follow-through" (416 endorsements, 411 with usable onset). It is not a
   treatment history, but it can enter as a sensitivity covariate rather than being
   listed as wholly unmeasured.

C. A SELF-REPORT ANALOGUE OF REFRAMING EXISTS AND HAS NEVER BEEN USED.
   The life-strategy module "helpful reinterpretation" (373 endorsements) is a
   self-reported version of the construct the SBERT score is meant to capture. This is
   a convergent check, NOT a substitute for E1: endorsing a coping strategy is not the
   same as using reframing language in a past narrative, and a null here would be
   ambiguous rather than damning. Reported as exploratory.

SAFE outputs only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, PID, FOCAL, COVARS, CATEGORICALS,
                     OUTCOME, CONFIRMATORY, W_PID, W_Q, W_REF, W_BW, build,
                     COPING_PROMPTS)

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
EVENT = Path(ROOT) / "discovery_analysis/intermediate_PRIVATE/event_table_PRIVATE.csv"

FROZEN_SET = list(COPING_PROMPTS)                 # ["Q64","Q68","Q53"]
LIFE_STRATEGY_QS = ["Q64", "Q68", "Q71"]          # per window_join_coverage_SAFE.csv
UNION_SET = sorted(set(FROZEN_SET) | set(LIFE_STRATEGY_QS))

df, win, prov = build(ROOT)
preds = list(FOCAL.keys()) + COVARS


def refit(frame, label, extra=None):
    ps = preds + (extra or [])
    d = frame.dropna(subset=[OUTCOME] + [c for c in ps if c in frame.columns])
    cf = coef_frame(ols_hc3(d[OUTCOME].astype(float), design(d, ps, CATEGORICALS)))
    rows = []
    for t in CONFIRMATORY + (extra or []):
        r = cf.loc[cf["term"] == t]
        if len(r):
            r = r.iloc[0]
            rows.append({"analysis": label, "n": int(len(d)), "term": t,
                         "beta": round(r["beta"], 4), "se": round(r["se"], 4),
                         "ci_lo": round(r["ci_lo"], 4), "ci_hi": round(r["ci_hi"], 4),
                         "p": round(r["p"], 6)})
    return rows


def relanguage(drop_qs, label):
    """Recompute the participant language means after dropping some question ids."""
    w = win[~win[W_Q].isin(drop_qs)]
    g = w.groupby(W_PID)
    alt = (df.drop(columns=["mean_reframe", "mean_bw"])
             .merge(pd.DataFrame({"mean_reframe": g[W_REF].mean(),
                                  "mean_bw": g[W_BW].mean()}),
                    left_on=PID, right_index=True, how="left"))
    return refit(alt, label), int(len(w))


qcounts = win[W_Q].value_counts()
rows, meta = [], {}
rows += refit(df, "primary (all windows)")
meta["windows_all"] = int(len(win))
for label, qs in [("frozen COPING_PROMPTS Q64/Q68/Q53 (as run)", FROZEN_SET),
                  ("corrected life-strategy Q64/Q68/Q71", LIFE_STRATEGY_QS),
                  ("union Q53/Q64/Q68/Q71", UNION_SET)]:
    r, nw = relanguage(qs, "excluding " + label)
    rows += r
    meta["windows_" + label.split()[0]] = nw
sens_a = pd.DataFrame(rows)
sens_a.to_csv(OUT / "r9a_coping_prompt_sensitivity_SAFE.csv", index=False)

# ---------------------------------------------------------------- B and C
ev = pd.read_csv(EVENT, usecols=["ResponseId", "symptom_domain", "event_family",
                                 "endorsement_value_normalized", "onset_age_numeric",
                                 "onset_usable"], low_memory=False)
ls = ev[ev["event_family"] == "life_strategy"]
flags = {}
for dom, col in [("treatment follow-through", "endorsed_treatment_followthrough"),
                 ("helpful reinterpretation", "endorsed_helpful_reinterpretation"),
                 ("positive choice", "endorsed_positive_choice")]:
    sub = ls[ls["symptom_domain"] == dom]
    yes = set(sub.loc[sub["endorsement_value_normalized"] == 1, "ResponseId"])
    flags[col] = yes

d2 = df.copy()
for col, yes in flags.items():
    d2[col] = d2[PID].isin(yes).astype(float)

rows_b = refit(d2, "primary (unchanged reference)")
rows_b += refit(d2, "+ treatment follow-through as covariate",
                extra=["endorsed_treatment_followthrough"])
rows_b += refit(d2, "+ all three life-strategy endorsements",
                extra=["endorsed_treatment_followthrough",
                       "endorsed_helpful_reinterpretation",
                       "endorsed_positive_choice"])
sens_b = pd.DataFrame(rows_b)
sens_b.to_csv(OUT / "r9b_life_strategy_covariate_SAFE.csv", index=False)

# convergent check: does the SBERT reframing score track the self-reported strategy?
conv = {}
a = d2.dropna(subset=["mean_reframe"])
g1 = a.loc[a["endorsed_helpful_reinterpretation"] == 1, "mean_reframe"].astype(float)
g0 = a.loc[a["endorsed_helpful_reinterpretation"] == 0, "mean_reframe"].astype(float)
sp = np.sqrt(((len(g1) - 1) * g1.var(ddof=1) + (len(g0) - 1) * g0.var(ddof=1)) /
             (len(g1) + len(g0) - 2))
dcoh = float((g1.mean() - g0.mean()) / sp)
rng = np.random.default_rng(20260906)
bd = []
for _ in range(2000):
    b1 = rng.choice(g1.to_numpy(), len(g1), replace=True)
    b0 = rng.choice(g0.to_numpy(), len(g0), replace=True)
    s = np.sqrt(((len(b1) - 1) * b1.var(ddof=1) + (len(b0) - 1) * b0.var(ddof=1)) /
                (len(b1) + len(b0) - 2))
    bd.append((b1.mean() - b0.mean()) / s)
bd = np.array(bd)
conv["endorsed_helpful_reinterpretation"] = {
    "n_endorsed": int(len(g1)), "n_not": int(len(g0)),
    "mean_reframe_endorsed": round(float(g1.mean()), 5),
    "mean_reframe_not": round(float(g0.mean()), 5),
    "cohens_d": round(dcoh, 4),
    "d_ci95": [round(float(np.percentile(bd, 2.5)), 4),
               round(float(np.percentile(bd, 97.5)), 4)],
    "bootstrap_p_two_sided": round(float(min(1.0, 2 * min((bd <= 0).mean(),
                                                          (bd >= 0).mean()))), 4),
}
# same for the black-and-white score, as a discriminant contrast
a2 = d2.dropna(subset=["mean_bw"])
h1 = a2.loc[a2["endorsed_helpful_reinterpretation"] == 1, "mean_bw"].astype(float)
h0 = a2.loc[a2["endorsed_helpful_reinterpretation"] == 0, "mean_bw"].astype(float)
spb = np.sqrt(((len(h1) - 1) * h1.var(ddof=1) + (len(h0) - 1) * h0.var(ddof=1)) /
              (len(h1) + len(h0) - 2))
conv["discriminant_black_and_white_score"] = {
    "cohens_d": round(float((h1.mean() - h0.mean()) / spb), 4),
    "note": "shown so the reframing d can be read against a contrast measure",
}
conv["interpretation"] = (
    "Convergent evidence only, and weak by construction: endorsing 'helpful "
    "reinterpretation' as a life strategy is a different act from using reframing "
    "language in a past-experience narrative. A null result here would be ambiguous, "
    "not disconfirming. This does not substitute for E1.")

report = {"A_coping_prompts": {
              "frozen_set": FROZEN_SET, "life_strategy_questions": LIFE_STRATEGY_QS,
              "union": UNION_SET,
              "windows_per_question": {q: int(qcounts.get(q, 0))
                                       for q in sorted(set(UNION_SET))},
              "defect": ("the frozen set drops Q53, a PAST_NARRATIVE, and retains Q71, "
                         "a LIFE_STRATEGY_NARRATIVE"),
              "window_counts": meta},
          "C_convergent": conv}
(OUT / "r9_report_SAFE.json").write_text(json.dumps(report, indent=2))

print("=== A. coping-prompt sensitivity ===")
print(json.dumps(report["A_coping_prompts"], indent=2))
print(sens_a[sens_a["term"] == "mean_reframe"].to_string(index=False))
print("\n=== B. life-strategy endorsements as covariates ===")
print(sens_b[sens_b["term"].isin(["mean_reframe", "symptom_count"])].to_string(index=False))
print("\n(added covariate coefficients)")
print(sens_b[sens_b["term"].str.startswith("endorsed")].to_string(index=False))
print("\n=== C. convergent check ===")
print(json.dumps(conv, indent=2))
print("\nR9 OK")
