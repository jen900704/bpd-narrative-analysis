"""Complete-responder sensitivity for the 30-day symptom-impact composite.

The composite is a row mean computed with missing values skipped and NO minimum-item
rule, inherited from the earlier pipeline. A participant answering one item receives a
score on the same 1-5 scale as one answering all twelve. This refits the primary model
under progressively stricter item-completeness requirements, and recomputes the
composite from the raw items so the requirement is actually enforced (the archived
composite cannot be un-skipna'd).

Reported: n at each threshold, the four confirmatory coefficients, and the correlation
between the archived composite and a strict complete-case recomputation.
SAFE aggregate outputs only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, PID, FOCAL, COVARS, CATEGORICALS,
                     OUTCOME, CONFIRMATORY, build)

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
PERSON = Path(ROOT) / "discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
ITEMS = [f"Q92_{i}_num" for i in range(1, 9)] + [f"Q93_{i}_num" for i in range(1, 5)]

df, _, _ = build(ROOT)
p = pd.read_csv(PERSON, usecols=[PID] + ITEMS, low_memory=False)
d = df.merge(p, on=PID, how="left")
d["_n_items"] = d[ITEMS].notna().sum(axis=1)
d["_recomputed"] = d[ITEMS].mean(axis=1)          # same skipna rule, for comparison
preds = list(FOCAL.keys()) + COVARS

# how much does the archived composite depend on partial responders?
arch = d[OUTCOME].astype(float)
full = d.loc[d["_n_items"] == 12, [OUTCOME, "_recomputed"]].dropna()
audit = {
    "n_total": int(len(d)),
    "items_answered_distribution": {int(k): int(v) for k, v in
                                    d["_n_items"].value_counts().sort_index().items()},
    "n_with_all_12": int((d["_n_items"] == 12).sum()),
    "n_with_fewer_than_12": int(((d["_n_items"] < 12) & (d["_n_items"] > 0)).sum()),
    "n_with_zero_items": int((d["_n_items"] == 0).sum()),
    "archived_vs_recomputed_max_abs_diff_among_complete":
        float((full[OUTCOME] - full["_recomputed"]).abs().max()) if len(full) else None,
    "archived_vs_recomputed_r_all":
        round(float(arch.corr(d["_recomputed"])), 6),
}

rows = []
for thr in (1, 6, 9, 11, 12):
    sub = d[d["_n_items"] >= thr].copy()
    sub[OUTCOME] = sub["_recomputed"]              # enforce the threshold in the score
    ana = sub.dropna(subset=[OUTCOME] + [c for c in preds if c in sub.columns])
    if len(ana) < 60:
        continue
    fit = ols_hc3(ana[OUTCOME].astype(float), design(ana, preds, CATEGORICALS))
    cf = coef_frame(fit)
    for t in CONFIRMATORY:
        r = cf.loc[cf["term"] == t]
        if len(r):
            r = r.iloc[0]
            rows.append({"min_items": thr, "n": int(len(ana)), "term": t,
                         "beta": round(float(r["beta"]), 4),
                         "se": round(float(r["se"]), 4),
                         "ci_lo": round(float(r["ci_lo"]), 4),
                         "ci_hi": round(float(r["ci_hi"]), 4),
                         "p": round(float(r["p"]), 6),
                         "r2": round(float(fit["r2"]), 4)})
t = pd.DataFrame(rows)
t.to_csv(OUT / "complete_responder_sensitivity_SAFE.csv", index=False)

focal = t[t["term"].isin(["symptom_count", "mean_reframe"])]
piv = t.pivot(index="min_items", columns="term", values="beta")
audit["max_shift_from_primary"] = {
    c: round(float((piv[c] - piv[c].iloc[0]).abs().max()), 4) for c in piv.columns}
audit["conclusion"] = (
    "the two supported coefficients are stable across item-completeness thresholds"
    if all(v < 0.05 for v in audit["max_shift_from_primary"].values())
    else "at least one coefficient moves by more than .05 across thresholds; report the curve")
(OUT / "complete_responder_sensitivity_SAFE.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
print()
print(focal.to_string(index=False))
print("\nOK")
