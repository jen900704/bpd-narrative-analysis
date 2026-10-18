"""R3 figure, redrawn from the saved table (no re-estimation).

Layout is set in inches and every text block is given its own reserved band, because
the first two renders collided: the subtitle ran past the right edge, the x-axis label
overlapped the footnote, and the CI labels were clipped. Nothing here is recomputed;
it reads r3_e6_domain_onset_SAFE.csv and r3_e6_report_SAFE.json.
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# reads and writes SAFE files only: set BPD_SAFE_OUTDIR=outputs/e3 to redraw from this repo
if os.environ.get("BPD_SAFE_OUTDIR"):
    OUT = Path(os.environ["BPD_SAFE_OUTDIR"])
else:
    from paths import root as bpd_root
    OUT = Path(bpd_root()) / "discovery_analysis/output/e3"
tab = pd.read_csv(OUT / "r3_e6_domain_onset_SAFE.csv")
rep = json.loads((OUT / "r3_e6_report_SAFE.json").read_text())["resolution"]
fig14 = tab[tab["set"] == "primary_14"].sort_values("median_onset").reset_index(drop=True)
proxy = tab[tab["set"] != "primary_14"]
n = len(fig14)

# --- layout, in inches -------------------------------------------------------
W, TOP, ROW, BOT = 9.4, 1.25, 0.30, 1.35
H = TOP + ROW * n + BOT
LEFT, AXW = 0.285, 0.545

f = plt.figure(figsize=(W, H))
ax = f.add_axes([LEFT, BOT / H, AXW, (ROW * n) / H])
y = np.arange(n)[::-1]
ax.hlines(y, fig14["ci_lo"], fig14["ci_hi"], color="#9aa8bb", lw=2.8, zorder=2)
ax.plot(fig14["median_onset"], y, "o", color="#1f3a5f", ms=6.8, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(fig14["symptom_domain"], fontsize=8.8)
ax.set_ylim(-0.65, n - 0.35)
ax.set_xlim(11.4, 18.6)
ax.set_xticks(range(12, 19))
ax.tick_params(axis="x", labelsize=8.8)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Median retrospectively reported age at first experience (years)",
              fontsize=9, labelpad=8, color="#33405a")
ax.grid(axis="x", color="#e9ecf1", lw=0.9)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3cad6")

# value + CI in fixed columns just outside the axes, so spacing is uniform
tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(1.04, n - 0.30, "median", transform=tr, va="bottom", ha="left",
        fontsize=7.4, color="#8792a6")
ax.text(1.13, n - 0.30, "95% CI", transform=tr, va="bottom", ha="left",
        fontsize=7.4, color="#8792a6")
for i, r in fig14.iterrows():
    ax.text(1.04, y[i], "%.0f" % r["median_onset"], transform=tr, va="center",
            ha="left", fontsize=8.8, color="#1f3a5f", weight="bold")
    ax.text(1.13, y[i], "%.0f–%.0f" % (r["ci_lo"], r["ci_hi"]), transform=tr,
            va="center", ha="left", fontsize=7.8, color="#8792a6")

f.text(LEFT, 1 - 0.36 / H, "Retrospectively reported median onset ages",
       fontsize=13.5, weight="bold", color="#1f3a5f", va="top")
f.text(LEFT, 1 - 0.66 / H,
       "%d directly endorsed symptom domains · %s events · %d participants"
       % (n, format(rep["primary_events"], ","), rep["primary_participants"]),
       fontsize=8.8, color="#54617a", va="top")
f.text(LEFT, 1 - 0.86 / H,
       "Point = median; bar = participant-cluster bootstrap 95% CI, 2,000 resamples",
       fontsize=8.8, color="#54617a", va="top")
f.text(LEFT, 0.26 / H,
       "Group-level medians describing tendencies across people. They do not establish "
       "an individual sequence and are not\nevidence of developmental staging. Suicidal "
       "feelings is a separately labelled proxy (median %.0f, n = %d) and is\nreported "
       "in the supplement, not here."
       % (float(proxy["median_onset"].iloc[0]), int(proxy["n_events"].iloc[0])),
       fontsize=7.8, color="#7a8599", va="bottom", linespacing=1.55)

p = OUT / "r3_e6_domain_onset_primary14_SAFE.png"
f.savefig(p, dpi=200, facecolor="white")
print("figure -> %s" % p)
print("rows %d   events %d   participants %d"
      % (n, rep["primary_events"], rep["primary_participants"]))
