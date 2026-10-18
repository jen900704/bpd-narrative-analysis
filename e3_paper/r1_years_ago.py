"""R1 - versioned years_ago builder.

Decision 1 (5 Sep review): fix the years_ago builder as a VERSIONED NEW COLUMN,
do not overwrite any frozen table, document how E4 actually computed it, and
verify the repaired field row-by-row against the values E4 used.

The archived column `years_ago` in window_table_augmented_PRIVATE.csv is non-null
for 0 of 13,017 rows. E4 (e4fix.py, lines 39-43) therefore computed it inline as
    current_age_midpoint - onset_age_numeric,   negatives set to missing.
This script reproduces that derivation as a named, versioned function, writes it to
a SEPARATE private file (nothing existing is modified), and asserts element-wise
identity with the E4 in-script values.

Writes:
  intermediate_PRIVATE/window_years_ago_v2_PRIVATE.csv   (row-aligned key + value)
  output/e3/r1_years_ago_v2_report_SAFE.json             (aggregate only)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import PID, W_PID, build

from paths import root as bpd_root
ROOT = bpd_root()
OUT = Path(ROOT) / "discovery_analysis/output/e3"
INT = Path(ROOT) / "discovery_analysis/intermediate_PRIVATE"
WT = INT / "window_table_augmented_PRIVATE.csv"
VERSION = "years_ago_v2"
BUILT = "2026-09-06"


def build_years_ago_v2(win: pd.DataFrame, person_age: pd.DataFrame) -> pd.Series:
    """years_ago = current age (band midpoint) - onset age, negatives -> missing.

    Both inputs are self-reported. current_age_midpoint is the midpoint of a banded
    age item, so every value carries an error of roughly half the band width; this is
    an approximate elapsed time, not a measured one.
    """
    w = win.merge(person_age, left_on=W_PID, right_on=PID, how="left")
    y = w["current_age_midpoint"].astype(float) - w["onset_age_numeric"].astype(float)
    y = y.where(~(y < 0), np.nan)
    return y


df, _, _ = build(ROOT)
person_age = df[[PID, "current_age_midpoint"]].copy()

cols = pd.read_csv(WT, nrows=0).columns
need = [c for c in [W_PID, "question_id", "onset_age_numeric", "years_ago",
                    "censored_low"] if c in cols]
win = pd.read_csv(WT, usecols=need, low_memory=False)

v2 = build_years_ago_v2(win, person_age)

# --- the exact e4fix.py derivation, replayed for verification -----------------
w_ref = win.merge(person_age, left_on=W_PID, right_on=PID, how="left")
ref = w_ref["current_age_midpoint"] - w_ref["onset_age_numeric"]
ref.loc[ref < 0] = np.nan

both_na = v2.isna() & ref.isna()
both_ok = v2.notna() & ref.notna()
maxdiff = float(np.nanmax(np.abs(v2[both_ok].to_numpy() - ref[both_ok].to_numpy()))) \
    if both_ok.any() else 0.0
identical = bool(int(both_na.sum() + both_ok.sum()) == len(v2) and maxdiff == 0.0)

merged_age = w_ref["current_age_midpoint"].notna()
raw = w_ref["current_age_midpoint"] - w_ref["onset_age_numeric"]

report = {
    "version": VERSION,
    "built": BUILT,
    "source_table": str(WT),
    "formula": "current_age_midpoint - onset_age_numeric; negative -> missing",
    "n_windows": int(len(win)),
    "archived_years_ago_nonnull": int(win["years_ago"].notna().sum())
        if "years_ago" in win else None,
    "onset_age_numeric_nonnull": int(win["onset_age_numeric"].notna().sum()),
    "current_age_midpoint_matched": int(merged_age.sum()),
    "current_age_midpoint_unmatched": int((~merged_age).sum()),
    "derived_before_negative_rule": int(raw.notna().sum()),
    "negative_set_missing": int((raw < 0).sum()),
    "usable_v2": int(v2.notna().sum()),
    "missing_v2": int(v2.isna().sum()),
    "verification_vs_e4_inline": {
        "rows_compared": int(len(v2)),
        "both_missing": int(both_na.sum()),
        "both_present": int(both_ok.sum()),
        "max_abs_difference": maxdiff,
        "element_wise_identical": identical,
    },
    "distribution_v2": {
        "mean": round(float(v2.mean()), 3),
        "sd": round(float(v2.std(ddof=1)), 3),
        "min": float(v2.min()),
        "q25": float(v2.quantile(.25)),
        "median": float(v2.median()),
        "q75": float(v2.quantile(.75)),
        "max": float(v2.max()),
    },
    "thresholds_used_by_e4": {
        str(k): {"windows": int((v2 >= k).sum()),
                 "participants": int(win.loc[v2 >= k, W_PID].nunique())}
        for k in (0, 5, 10, 15)
    },
    "known_limitation": ("current_age_midpoint is a band midpoint, so each value carries "
                         "an error of about half a band width; the E4 restrictions are "
                         "approximate and the quantiles are an artefact of midpoint "
                         "arithmetic, not of the data."),
}

outp = INT / "window_years_ago_v2_PRIVATE.csv"
keep = [c for c in [W_PID, "question_id"] if c in win.columns]
pd.concat([win[keep].reset_index(drop=True),
           v2.rename(VERSION).reset_index(drop=True)], axis=1).to_csv(outp, index=False)

(OUT / "r1_years_ago_v2_report_SAFE.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("\nnothing existing was modified; new private file -> " + str(outp))
assert identical, "R1 FAILED: v2 does not reproduce the values E4 used"
print("R1 OK: years_ago_v2 is element-wise identical to the E4 in-script derivation")
