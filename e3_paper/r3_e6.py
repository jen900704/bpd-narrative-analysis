"""R3 - E6 restricted to the frozen primary symptom family (14 domains).

Decision 2 (5 Sep review): the primary onset analysis and the main figure must be
restricted to the 14 directly-endorsed domains (6,255 events). Suicidal feelings is a
separately labelled proxy and moves to a supplementary panel.

Resolution recorded in the 6 Sep audit: filtering the event table on
include_in_primary_symptom_sequence gives 6,681 events across 15 domains; suicidal
feelings contributes 426; 6,681 - 426 = 6,255 across 14 domains, exactly the frozen
Phase 1.5A figure. The event-table flag is broader than the frozen primary family.

Also rewrites the age-heaping table so that the subset relationship is explicit:
multiples of 10 are a SUBSET of multiples of 5, and the 12/13 peaks are not round
numbers and are not heaping.

Supersedes e6_domain_onset_SAFE.csv, e6_age_heaping_SAFE.csv and
e6_domain_onset_SAFE.png. Those files are left in place; the new ones carry _r3_.
SAFE outputs only.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
EVENT = ROOT / "discovery_analysis/intermediate_PRIVATE/event_table_PRIVATE.csv"
SEED = 20260906
NBOOT = 2000
EXPECT_15, EXPECT_14, EXPECT_PROXY = 6681, 6255, 426

ev = pd.read_csv(EVENT, usecols=["ResponseId", "symptom_domain", "onset_age_numeric",
                                 "censored_low", "onset_usable",
                                 "include_in_primary_symptom_sequence"],
                 low_memory=False)
flagged = ev[(ev["include_in_primary_symptom_sequence"] == True) &
             (ev["onset_usable"] == True) & ev["onset_age_numeric"].notna()].copy()

doms = sorted(flagged["symptom_domain"].dropna().unique().tolist())
print("=== domains carrying include_in_primary_symptom_sequence ===")
for d in doms:
    print("  %-38s n=%d" % (d, int((flagged["symptom_domain"] == d).sum())))
cand = [d for d in doms if str(d).strip().lower() == "suicidal feelings"]
if not cand:
    cand = [d for d in doms
            if re.search(r"suicid", str(d), re.I)
            and not re.search(r"non[-\s]?suicid|self[-\s]?injur|attempt", str(d), re.I)]
assert len(cand) == 1, "could not identify a single suicidal-feelings domain: %r" % cand
PROXY = cand[0]

primary = flagged[flagged["symptom_domain"] != PROXY].copy()
proxy = flagged[flagged["symptom_domain"] == PROXY].copy()

resolution = {
    "flagged_events": int(len(flagged)),
    "flagged_domains": int(flagged["symptom_domain"].nunique()),
    "proxy_domain": str(PROXY),
    "proxy_events": int(len(proxy)),
    "primary_events": int(len(primary)),
    "primary_domains": int(primary["symptom_domain"].nunique()),
    "primary_participants": int(primary["ResponseId"].nunique()),
    "matches_frozen_phase_1_5A": bool(len(primary) == EXPECT_14 and
                                      primary["symptom_domain"].nunique() == 14),
    "arithmetic": "%d flagged - %d proxy = %d primary" % (len(flagged), len(proxy),
                                                          len(primary)),
}
print(json.dumps(resolution, indent=2))
assert len(flagged) == EXPECT_15, "flagged n changed: %d" % len(flagged)
assert len(proxy) == EXPECT_PROXY, "proxy n changed: %d" % len(proxy)
assert len(primary) == EXPECT_14, "primary n changed: %d" % len(primary)
assert primary["symptom_domain"].nunique() == 14


def onset_table(frame, label):
    rng = np.random.default_rng(SEED)
    pids = frame["ResponseId"].unique()
    rows = []
    for dom, g in frame.groupby("symptom_domain"):
        v = g["onset_age_numeric"].dropna()
        if len(v) < 30:
            continue
        boots = []
        for _ in range(NBOOT):
            s = rng.choice(pids, len(pids), replace=True)
            cnt = pd.Series(s).value_counts()
            gg = g[g["ResponseId"].isin(cnt.index)]
            w = gg["ResponseId"].map(cnt).to_numpy()
            vals = np.repeat(gg["onset_age_numeric"].to_numpy(), w)
            if len(vals):
                boots.append(np.median(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append({"set": label, "symptom_domain": dom, "n_events": int(len(v)),
                     "n_participants": int(g["ResponseId"].nunique()),
                     "median_onset": float(v.median()), "ci_lo": float(lo),
                     "ci_hi": float(hi), "q25": float(v.quantile(.25)),
                     "q75": float(v.quantile(.75)),
                     "pct_left_censored": round(float(g["censored_low"].fillna(False)
                                                      .astype(bool).mean() * 100), 1)})
    return pd.DataFrame(rows).sort_values("median_onset").reset_index(drop=True)


fig14 = onset_table(primary, "primary_14")
figpx = onset_table(proxy, "proxy_suicidal_feelings")
pd.concat([fig14, figpx], ignore_index=True).round(3).to_csv(
    OUT / "r3_e6_domain_onset_SAFE.csv", index=False)
print("\n=== E6 primary family (14 domains, %d events) ===" % len(primary))
print(fig14[["symptom_domain", "n_events", "median_onset", "ci_lo", "ci_hi",
             "pct_left_censored"]].to_string(index=False))
print("\n=== supplementary: %s (separately labelled proxy) ===" % PROXY)
print(figpx[["symptom_domain", "n_events", "median_onset", "ci_lo", "ci_hi",
             "pct_left_censored"]].to_string(index=False))


def heaping(frame, label):
    a = frame["onset_age_numeric"].dropna().astype(float)
    n = len(a)
    m5 = (a % 5 == 0)
    m10 = (a % 10 == 0)
    only5 = m5 & ~m10
    return {"set": label, "n_events": n,
            "multiples_of_5_n": int(m5.sum()),
            "multiples_of_5_pct": round(float(m5.mean() * 100), 2),
            "of_which_multiples_of_10_n": int(m10.sum()),
            "of_which_multiples_of_10_pct": round(float(m10.mean() * 100), 2),
            "multiples_of_5_not_10_n": int(only5.sum()),
            "multiples_of_5_not_10_pct": round(float(only5.mean() * 100), 2)}


heap = pd.DataFrame([heaping(flagged, "15_domain_flagged_superseded"),
                     heaping(primary, "14_domain_primary")])
heap.to_csv(OUT / "r3_e6_age_heaping_SAFE.csv", index=False)

a = primary["onset_age_numeric"].dropna().astype(float)
cens = primary.assign(_a=a)
peaks = []
for k in sorted(a.value_counts().head(10).index.tolist()):
    sub = primary[primary["onset_age_numeric"] == k]
    peaks.append({"age": float(k), "n": int((a == k).sum()),
                  "pct_of_primary": round(float((a == k).mean() * 100), 2),
                  "multiple_of_5": bool(k % 5 == 0),
                  "multiple_of_10": bool(k % 10 == 0),
                  "is_round_number": bool(k % 5 == 0),
                  "pct_left_censored_at_this_age":
                      round(float(sub["censored_low"].fillna(False).astype(bool)
                                  .mean() * 100), 1)})
pk = pd.DataFrame(peaks).sort_values("n", ascending=False).reset_index(drop=True)
pk.to_csv(OUT / "r3_e6_onset_peaks_SAFE.csv", index=False)

print("\n=== age heaping, corrected ===")
print(heap.to_string(index=False))
print("\n=== ten most frequent reported ages (primary family) ===")
print(pk.to_string(index=False))

notes = {
    "subset_relationship": ("Multiples of ten are a SUBSET of multiples of five. The two "
                            "percentages must not be presented as separate facts; report "
                            "multiples of 5 with 'of which multiples of 10' nested inside."),
    "non_round_peaks": ("The concentrations at 12 and 13 are not round numbers and are not "
                        "age heaping; they more plausibly reflect adolescent concentration "
                        "or a recall anchor."),
    "floor_at_5": ("Reports at 5 are confounded with the left-censoring floor ('5 or "
                   "younger') and should not be read as heaping either; see the "
                   "pct_left_censored_at_this_age column."),
    "wording": ("Reported onset ages showed round-number concentration consistent with "
                "possible age heaping: %.1f%% of events fell on multiples of five, "
                "including %.1f%% that were multiples of ten."
                % (heap.loc[1, "multiples_of_5_pct"],
                   heap.loc[1, "of_which_multiples_of_10_pct"])),
}
report = {"resolution": resolution, "heaping_notes": notes,
          "seed": SEED, "n_bootstrap": NBOOT,
          "supersedes": ["e6_domain_onset_SAFE.csv", "e6_age_heaping_SAFE.csv",
                         "e6_domain_onset_SAFE.png"]}
(OUT / "r3_e6_report_SAFE.json").write_text(json.dumps(report, indent=2))

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n14 = len(fig14)
    H = 1.30 + 0.30 * n14 + 0.86
    W = 8.6
    f = plt.figure(figsize=(W, H))
    ax = f.add_axes([0.335, 0.86 / H, 0.635, (0.30 * n14) / H])
    y = np.arange(n14)[::-1]
    ax.hlines(y, fig14["ci_lo"], fig14["ci_hi"], color="#9aa8bb", lw=2.6)
    ax.plot(fig14["median_onset"], y, "o", color="#1f3a5f", ms=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(fig14["symptom_domain"], fontsize=8.6)
    ax.set_ylim(-0.7, n14 - 0.3)
    ax.set_xlabel("Median retrospectively reported age at first experience (years)",
                  fontsize=9)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.grid(axis="x", color="#e6e9ee")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for i, r in fig14.iterrows():
        ax.text(r["ci_hi"] + 0.28, y[i], "%.0f" % r["median_onset"],
                va="center", fontsize=7.8, color="#54617a")
    f.text(0.335, 1 - 0.34 / H, "Retrospectively reported median onset ages",
           fontsize=12.5, weight="bold", color="#1f3a5f", va="top")
    f.text(0.335, 1 - 0.66 / H,
           "%d directly endorsed symptom domains  ·  %d events  ·  %d participants  ·  "
           "participant-cluster bootstrap 95%% CI" % (n14, len(primary),
                                                      primary["ResponseId"].nunique()),
           fontsize=8.4, color="#54617a", va="top")
    f.text(0.335, 0.30 / H,
           "Group-level medians. They describe tendencies across people and do not "
           "establish an individual sequence.\nSuicidal feelings is a separately "
           "labelled proxy and is reported in the supplement, not here.",
           fontsize=7.6, color="#7a8599", va="bottom")
    f.savefig(OUT / "r3_e6_domain_onset_primary14_SAFE.png", dpi=200)
    print("\nfigure -> %s" % (OUT / "r3_e6_domain_onset_primary14_SAFE.png"))
except Exception as e:
    print("figure not rendered: %r" % (e,))

print("\nR3 OK")
