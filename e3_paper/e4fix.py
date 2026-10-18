"""E4 (time since event) + collinearity sensitivity, run after e3_clsp.py.

Why this exists: `years_ago` is present in the window table but entirely null
(0 of 13,017), so e3_clsp.py could not compute the time-since-event contrast. It is
derivable as current_age_midpoint - onset_age_numeric, which is exactly how the column
was presumably meant to be filled. This script derives it, reports how much is usable,
and re-estimates. It also refits the primary model dropping sbert_n_windows, because
VIF showed PAST_narrative_word_count_total (7.06) and sbert_n_windows (7.20) are badly
collinear with each other.

SAFE outputs only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, vif_table, PID, FOCAL, COVARS,
                     CATEGORICALS, OUTCOME, CONFIRMATORY, W_PID, W_REF, W_BW, build)

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
WT = Path(ROOT) / "discovery_analysis/intermediate_PRIVATE/window_table_augmented_PRIVATE.csv"

df, _, prov = build(ROOT)
preds = list(FOCAL.keys()) + COVARS

cols = pd.read_csv(WT, nrows=0).columns
need = [c for c in [W_PID, "question_id", W_REF, W_BW, "onset_age_numeric",
                    "years_ago", "censored_low"] if c in cols]
win = pd.read_csv(WT, usecols=need, low_memory=False)

report = {"n_windows": int(len(win)),
          "years_ago_nonnull_original": int(win["years_ago"].notna().sum())
          if "years_ago" in win else None,
          "onset_age_numeric_nonnull": int(win["onset_age_numeric"].notna().sum())}

# derive time since event
w = win.merge(df[[PID, "current_age_midpoint"]], left_on=W_PID, right_on=PID, how="left")
w["tse"] = w["current_age_midpoint"] - w["onset_age_numeric"]
report["tse_derived_nonnull"] = int(w["tse"].notna().sum())
report["tse_negative"] = int((w["tse"] < 0).sum())
w.loc[w["tse"] < 0, "tse"] = np.nan
report["tse_describe"] = {k: (None if pd.isna(v) else round(float(v), 2))
                          for k, v in w["tse"].describe().items()}
for t in (5, 10, 15, 20):
    report[f"windows_ge{t}"] = int((w["tse"] >= t).sum())
    report[f"participants_ge{t}"] = int(w.loc[w["tse"] >= t, W_PID].nunique())

rows = []
for thr in [None, 5, 10, 15]:
    sub = w if thr is None else w[w["tse"] >= thr]
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
    cf = coef_frame(ols_hc3(d[OUTCOME].astype(float), design(d, preds, CATEGORICALS)))
    r = cf.loc[cf["term"] == "mean_reframe"].iloc[0]
    rows.append({"threshold_years": "all" if thr is None else thr,
                 "n_participants": int(len(d)), "n_windows": int(len(sub)),
                 "median_years_since_event": round(float(sub["tse"].median()), 2)
                 if sub["tse"].notna().any() else None,
                 "beta_mean_reframe": r["beta"], "se": r["se"],
                 "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"]})
e4 = pd.DataFrame(rows)
e4.round(6).to_csv(OUT / "e4_time_since_event_SAFE.csv", index=False)

# collinearity sensitivity: drop sbert_n_windows
alt = [p for p in preds if p != "sbert_n_windows"]
a = df.dropna(subset=[OUTCOME] + [p for p in alt if p in df.columns])
X2 = design(a, alt, CATEGORICALS)
cf2 = coef_frame(ols_hc3(a[OUTCOME].astype(float), X2))
cf2[cf2["term"].isin(CONFIRMATORY)].round(6).to_csv(
    OUT / "e3_sensitivity_drop_nwindows_SAFE.csv", index=False)
vif_table(X2).round(3).to_csv(OUT / "e3_vif_drop_nwindows_SAFE.csv", index=False)

(OUT / "e4_derivation_report_SAFE.json").write_text(json.dumps(report, indent=2))

print("=== time-since-event derivation ===")
for k, v in report.items():
    print(f"  {k}: {v}")
print("\n=== E4 ===")
print(e4.to_string(index=False, float_format=lambda v: f"{v: .4f}") if len(e4)
      else "  not estimable")
print("\n=== primary model without sbert_n_windows (collinearity check) ===")
print(cf2[cf2["term"].isin(CONFIRMATORY)][["term", "beta", "se", "ci_lo", "ci_hi", "p"]]
      .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
print(f"  max VIF now: {vif_table(X2)['vif'].max():.2f}   n = {len(a)}")
