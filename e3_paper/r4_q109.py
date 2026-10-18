"""R4 - Q109 item-level statistics and leave-one-item-out reliability.

Decision 3 (5 Sep review): Q109 becomes an exploratory / secondary outcome; report
its reliability and item-level behaviour rather than treating it as a scale on the
same footing as the primary composite. Scale alpha was .598 in the person table.

SAFE outputs only: item means, SDs, item-total correlations, alpha if item deleted.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import alpha_omega

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
PERSON = ROOT / "discovery_analysis/intermediate_PRIVATE/person_table_PRIVATE.csv"
from paths import raw_survey
RAW = raw_survey()


def cronbach(X: pd.DataFrame) -> float:
    X = X.dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return float("nan")
    v = X.var(axis=0, ddof=1).sum()
    t = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - v / t))


p = pd.read_csv(PERSON, low_memory=False)
items = [c for c in p.columns if c.startswith("Q109_") and c.endswith("_num")]
items = sorted(items, key=lambda s: int(s.split("_")[1]))
assert items, "no Q109 items found in the person table"

labels = {}
if RAW is not None and RAW.exists():
    hdr = pd.read_csv(RAW, nrows=1, low_memory=False)
    for it in items:
        q = it[:-4]
        if q in hdr.columns:
            labels[it] = str(hdr[q].iloc[0])[:180].replace("\n", " ")

X = p[items]
Xc = X.dropna()
a_all, om_all = alpha_omega(X)

rows = []
for it in items:
    rest = [c for c in items if c != it]
    total_rest = Xc[rest].sum(axis=1)
    r_it = float(np.corrcoef(Xc[it], total_rest)[0, 1])
    rows.append({
        "item": it,
        "n": int(X[it].notna().sum()),
        "mean": round(float(X[it].mean()), 3),
        "sd": round(float(X[it].std(ddof=1)), 3),
        "min": float(X[it].min()),
        "max": float(X[it].max()),
        "corrected_item_total_r": round(r_it, 3),
        "alpha_if_deleted": round(cronbach(Xc[rest]), 3),
        "k_remaining": len(rest),
        "label": labels.get(it, ""),
    })
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "r4_q109_items_SAFE.csv", index=False)

corr = Xc.corr().round(3)
corr.to_csv(OUT / "r4_q109_interitem_SAFE.csv")

summary = {
    "k_items": len(items),
    "n_complete": int(len(Xc)),
    "alpha": round(float(a_all), 3),
    "omega_total": round(float(om_all), 3),
    "mean_interitem_r": round(float(corr.to_numpy()[np.triu_indices(len(items), 1)]
                                    .mean()), 3),
    "min_interitem_r": round(float(corr.to_numpy()[np.triu_indices(len(items), 1)]
                                   .min()), 3),
    "max_interitem_r": round(float(corr.to_numpy()[np.triu_indices(len(items), 1)]
                                   .max()), 3),
    "best_alpha_if_deleted": {
        "item": str(tab.loc[tab["alpha_if_deleted"].idxmax(), "item"]),
        "alpha": float(tab["alpha_if_deleted"].max()),
    },
    "reporting_status": ("exploratory / secondary outcome, reported with its own "
                         "reliability; not interchangeable with the Q92+Q93 primary "
                         "composite (alpha .885, omega .905)"),
    "note": ("With only %d items, alpha is bounded low by scale length as well as by "
             "inter-item correlation; leave-one-item-out on a %d-item scale leaves a "
             "%d-item scale, so 'alpha if deleted' should be read against that."
             % (len(items), len(items), len(items) - 1)),
}
(OUT / "r4_q109_report_SAFE.json").write_text(json.dumps(summary, indent=2))

print("=== Q109 item level ===")
print(tab.drop(columns=["label"]).to_string(index=False))
for _, r in tab.iterrows():
    if r["label"]:
        print("  %s: %s" % (r["item"], r["label"]))
print("\n=== inter-item correlations ===")
print(corr.to_string())
print("\n=== summary ===")
print(json.dumps(summary, indent=2))
print("\nR4 OK")
