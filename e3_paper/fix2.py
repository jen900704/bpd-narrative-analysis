"""Two gaps from the first real run:
1. e3_reliability_SAFE.csv was empty — the Q92/Q93/Q109 items live in the PERSON table,
   not the phase1_3 analytic table, so alpha/omega had nothing to read.
2. The 'excluding left-censored onset' sensitivity moved mean_reframe a long way
   (-.216 -> -.119). Need the n behind it before deciding whether that is fragility or
   just a smaller subsample.
SAFE outputs only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, alpha_omega, PID, FOCAL, COVARS,
                     CATEGORICALS, OUTCOME, CONFIRMATORY, build)

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
PERSON = Path(ROOT) / "discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"

# ---- 1. reliability from the person table
p = pd.read_csv(PERSON, low_memory=False)
rows = []
specs = [("Q92 thoughts/feelings", [f"Q92_{i}_num" for i in range(1, 9)]),
         ("Q93 negative behaviours", [f"Q93_{i}_num" for i in range(1, 5)]),
         ("Q109 positive behaviours", [f"Q109_{i}_num" for i in range(1, 4)]),
         ("Q92+Q93 composite (primary)",
          [f"Q92_{i}_num" for i in range(1, 9)] + [f"Q93_{i}_num" for i in range(1, 5)])]
for nm, cols in specs:
    have = [c for c in cols if c in p.columns]
    if len(have) != len(cols):
        rows.append({"scale": nm, "k": len(cols), "n": 0, "alpha": None,
                     "omega_total": None, "note": "items not found"})
        continue
    a, om = alpha_omega(p[have])
    s = p[have].mean(axis=1)
    rows.append({"scale": nm, "k": len(cols), "n": int(p[have].dropna().shape[0]),
                 "alpha": round(a, 3), "omega_total": round(om, 3),
                 "mean": round(float(s.mean()), 3), "sd": round(float(s.std(ddof=1)), 3)})
rel = pd.DataFrame(rows)
rel.to_csv(OUT / "e3_reliability_SAFE.csv", index=False)

# ---- 2. left-censoring sensitivity, with n
df, _, _ = build(ROOT)
preds = list(FOCAL.keys()) + COVARS
ana = df.dropna(subset=[OUTCOME] + [c for c in preds if c in df.columns])
lc = ana["earliest_censored_low"].fillna(0).astype(float)
out = {"analytic_n": int(len(ana)),
       "left_censored_n": int((lc == 1).sum()),
       "left_censored_pct": round(float((lc == 1).mean() * 100), 1)}

sub = []
for label, d in [("all (primary)", ana), ("excluding left-censored", ana[lc == 0]),
                 ("left-censored only", ana[lc == 1])]:
    if len(d) < 60:
        sub.append({"subset": label, "n": int(len(d)), "note": "too small to fit"})
        continue
    cf = coef_frame(ols_hc3(d[OUTCOME].astype(float), design(d, preds, CATEGORICALS)))
    for t in CONFIRMATORY:
        r = cf.loc[cf["term"] == t].iloc[0]
        sub.append({"subset": label, "n": int(len(d)), "term": t,
                    "beta": round(r["beta"], 4), "se": round(r["se"], 4),
                    "ci_lo": round(r["ci_lo"], 4), "ci_hi": round(r["ci_hi"], 4),
                    "p": round(r["p"], 4)})
s = pd.DataFrame(sub)
s.to_csv(OUT / "e3_left_censoring_sensitivity_SAFE.csv", index=False)
(OUT / "e3_left_censoring_report_SAFE.json").write_text(json.dumps(out, indent=2))

print("=== reliability ===")
print(rel.to_string(index=False))
print("\n=== left-censoring ===")
for k, v in out.items():
    print(f"  {k}: {v}")
print(s.to_string(index=False))
