"""E5 (moderation), E6 (onset figure + age heaping), and identification of the
candidate covariates Q4/Q80/Q112-Q117. SAFE outputs only; question text is survey
metadata, not participant data."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, PID, FOCAL, COVARS, CATEGORICALS,
                     OUTCOME, build)

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
EVENT = ROOT / "discovery_analysis/intermediate_PRIVATE/event_table_PRIVATE.csv"
PERSON = ROOT / "discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
from paths import raw_survey
RAW = raw_survey()

# ---------- 0. what are Q4 / Q80 / Q112-117 ----------
labels = {}
if RAW is not None and RAW.exists():
    hdr = pd.read_csv(RAW, nrows=1, low_memory=False)
    for q in ["Q3", "Q4", "Q80", "Q112", "Q113", "Q114", "Q115", "Q117"]:
        if q in hdr.columns:
            labels[q] = str(hdr[q].iloc[0])[:150].replace("\n", " ")
(OUT / "covariate_item_labels_SAFE.json").write_text(json.dumps(labels, indent=2))
print("=== candidate covariate items (survey question text) ===")
for k, v in labels.items():
    print(f"  {k}: {v}")

# ---------- 1. E5 moderation ----------
df, _, _ = build(str(ROOT))
preds = list(FOCAL.keys()) + COVARS
ana = df.dropna(subset=[OUTCOME] + [c for c in preds if c in df.columns]).copy()
X = design(ana, preds, CATEGORICALS)
z = lambda s: (s - s.mean()) / s.std(ddof=1)
zb, zr = z(ana["symptom_count"].astype(float)), z(ana["mean_reframe"].astype(float))
zlc = ana["earliest_censored_low"].fillna(0).astype(float)

X5 = X.copy(); X5["breadth_x_reframe"] = (zb * zr).to_numpy()
f5 = ols_hc3(ana[OUTCOME].astype(float), X5); c5 = coef_frame(f5)
r = c5.loc[c5["term"] == "breadth_x_reframe"].iloc[0]

cov, terms = f5["cov"], f5["terms"]
ib, ii = terms.index("symptom_count"), terms.index("breadth_x_reframe")
bb = dict(zip(terms, f5["beta"]))
slopes = []
for lvl, nm in [(-1, "-1 SD reframing"), (0, "mean reframing"), (1, "+1 SD reframing")]:
    est = bb["symptom_count"] + lvl * bb["breadth_x_reframe"]
    se = float(np.sqrt(max(cov[ib, ib] + lvl**2 * cov[ii, ii] + 2*lvl*cov[ib, ii], 0)))
    slopes.append({"reframing_level": nm, "breadth_slope": round(est, 4),
                   "se": round(se, 4), "ci_lo": round(est-1.96*se, 4),
                   "ci_hi": round(est+1.96*se, 4)})

# formal test of the left-censoring moderation seen earlier
X6 = X.copy()
X6["reframe_x_leftcens"] = (zr * (zlc - zlc.mean())).to_numpy()
X6["leftcens"] = (zlc - zlc.mean()).to_numpy()
f6 = ols_hc3(ana[OUTCOME].astype(float), X6); c6 = coef_frame(f6)
r6 = c6.loc[c6["term"] == "reframe_x_leftcens"].iloc[0]

e5 = pd.DataFrame([
    {"test": "breadth x reframing", "beta": round(r["beta"], 4), "se": round(r["se"], 4),
     "ci_lo": round(r["ci_lo"], 4), "ci_hi": round(r["ci_hi"], 4), "p": round(r["p"], 4),
     "status": "exploratory (preregistered as H4)"},
    {"test": "reframing x left-censored onset", "beta": round(r6["beta"], 4),
     "se": round(r6["se"], 4), "ci_lo": round(r6["ci_lo"], 4),
     "ci_hi": round(r6["ci_hi"], 4), "p": round(r6["p"], 4),
     "status": "POST HOC - not preregistered"}])
e5.to_csv(OUT / "e5_moderation_SAFE.csv", index=False)
pd.DataFrame(slopes).to_csv(OUT / "e5_simple_slopes_SAFE.csv", index=False)
print("\n=== E5 moderation ===")
print(e5.to_string(index=False))
print(pd.DataFrame(slopes).to_string(index=False))

# ---------- 2. E6 onset by domain + age heaping ----------
ev = pd.read_csv(EVENT, usecols=["ResponseId", "symptom_domain", "onset_age_numeric",
                                 "censored_low", "onset_usable",
                                 "include_in_primary_symptom_sequence"],
                 low_memory=False)
prim = ev[(ev["include_in_primary_symptom_sequence"] == True) &
          (ev["onset_usable"] == True) & ev["onset_age_numeric"].notna()].copy()
print(f"\n=== E6 ===\n  primary usable onset events: {len(prim)}"
      f"   participants: {prim['ResponseId'].nunique()}"
      f"   domains: {prim['symptom_domain'].nunique()}")

rng = np.random.default_rng(20260905)
pids = prim["ResponseId"].unique()
rows = []
for dom, g in prim.groupby("symptom_domain"):
    v = g["onset_age_numeric"].dropna()
    if len(v) < 30:
        continue
    boots = []
    for _ in range(2000):
        s = rng.choice(pids, len(pids), replace=True)
        cnt = pd.Series(s).value_counts()
        gg = g[g["ResponseId"].isin(cnt.index)]
        w = gg["ResponseId"].map(cnt).to_numpy()
        vals = np.repeat(gg["onset_age_numeric"].to_numpy(), w)
        if len(vals):
            boots.append(np.median(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    rows.append({"symptom_domain": dom, "n_events": int(len(v)),
                 "n_participants": int(g["ResponseId"].nunique()),
                 "median_onset": float(v.median()), "ci_lo": float(lo),
                 "ci_hi": float(hi), "q25": float(v.quantile(.25)),
                 "q75": float(v.quantile(.75)),
                 "pct_left_censored": round(float(g["censored_low"].fillna(False)
                                                  .astype(bool).mean()*100), 1)})
fig = pd.DataFrame(rows).sort_values("median_onset").reset_index(drop=True)
fig.round(3).to_csv(OUT / "e6_domain_onset_SAFE.csv", index=False)
print(fig[["symptom_domain", "n_events", "median_onset", "ci_lo", "ci_hi",
           "pct_left_censored"]].to_string(index=False))

a = prim["onset_age_numeric"].dropna()
heap = pd.DataFrame({"statistic": ["n events", "on multiple of 5", "on multiple of 10",
                                   "at 5", "at 10", "at 12", "at 13", "at 15", "at 18"],
                     "value": [len(a), float((a % 5 == 0).mean()),
                               float((a % 10 == 0).mean())] +
                              [float((a == k).mean()) for k in (5, 10, 12, 13, 15, 18)]})
heap.round(4).to_csv(OUT / "e6_age_heaping_SAFE.csv", index=False)
print("\n=== age heaping ===")
print(heap.round(4).to_string(index=False))

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = max(3.2, 0.42 * len(fig))
    f, ax = plt.subplots(figsize=(7.6, h))
    y = np.arange(len(fig))
    ax.hlines(y, fig["ci_lo"], fig["ci_hi"], color="#9aa8bb", lw=2.6)
    ax.plot(fig["median_onset"], y, "o", color="#1f3a5f", ms=6.5)
    ax.set_yticks(y); ax.set_yticklabels(fig["symptom_domain"], fontsize=9)
    ax.set_xlabel("Median retrospectively reported age at first experience (years)")
    ax.set_title("Reported onset by symptom domain\nparticipant-cluster bootstrap 95% CI",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color="#e6e9ee"); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(OUT / "e6_domain_onset_SAFE.png", dpi=200)
    print(f"\nfigure -> {OUT/'e6_domain_onset_SAFE.png'}")
except Exception as e:
    print(f"figure not rendered: {e}")
