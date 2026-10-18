"""Attrition check for the Results section: do the 61 participants without language
measures differ from the 478 in the analytic sample? SAFE aggregate output only."""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import PID, FOCAL, COVARS, CATEGORICALS, OUTCOME, build

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
df, _, _ = build(ROOT)
preds = list(FOCAL.keys()) + COVARS
ana = df.dropna(subset=[OUTCOME] + [c for c in preds if c in df.columns])
inc = df[PID].isin(ana[PID])
rows = []
for col, lab in [(OUTCOME, "30-day symptom impact"), ("symptom_count", "symptom breadth"),
                 ("earliest_onset", "earliest onset age"),
                 ("diagnosis_lag", "diagnosis lag"),
                 ("current_age_midpoint", "current age (band midpoint)"),
                 ("PAST_narrative_word_count_total", "past-narrative word count")]:
    if col not in df.columns:
        continue
    a = df.loc[inc, col].astype(float).dropna()
    b = df.loc[~inc, col].astype(float).dropna()
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                 max(len(a) + len(b) - 2, 1)) if len(b) > 1 else np.nan
    rows.append({"variable": lab, "n_analytic": int(len(a)), "n_excluded": int(len(b)),
                 "mean_analytic": round(float(a.mean()), 3),
                 "mean_excluded": round(float(b.mean()), 3) if len(b) else None,
                 "sd_analytic": round(float(a.std(ddof=1)), 3),
                 "sd_excluded": round(float(b.std(ddof=1)), 3) if len(b) > 1 else None,
                 "cohens_d": round(float((b.mean() - a.mean()) / sp), 3)
                 if len(b) > 1 and sp and sp > 0 else None})
t = pd.DataFrame(rows)
t.to_csv(OUT / "attrition_comparison_SAFE.csv", index=False)
g = df.loc[~inc, "gender_collapsed"].value_counts(dropna=False)
rec = {"n_excluded": int((~inc).sum()), "n_analytic": int(inc.sum()),
       "table": rows,
       "excluded_gender_counts_suppressed_below_5":
           {str(k): (int(v) if v >= 5 else "<5") for k, v in g.items()}}
(OUT / "attrition_comparison_SAFE.json").write_text(json.dumps(rec, indent=2))
print(t.to_string(index=False))
print("\nexcluded n =", int((~inc).sum()))
