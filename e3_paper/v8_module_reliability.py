"""V8 - reliability recomputed at MODULE level.

The published ICC(1,k) and split-half were computed over overlapping text windows.
Adjacent windows share text, so those estimates can be inflated. Here each participant
is first averaged within question_id (module), and reliability is recomputed over
modules, which do not share text. Aggregate output only.
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import e3_clsp as E

from paths import root as bpd_root
ROOT = bpd_root()
OUT = "output/e3/v8_module_reliability_SAFE.json"
SCORES = [("mean_reframe", "sbert_reframing_similarity"),
          ("mean_bw", "sbert_blackwhite_similarity")]


def icc1k(g):
    """One-way random effects ICC from a long frame with columns pid, val."""
    grp = g.groupby("pid")["val"]
    k = grp.size()
    n = len(k)
    kbar = (k.sum() - (k ** 2).sum() / k.sum()) / (n - 1)
    gm = g["val"].mean()
    msb = (k * (grp.mean() - gm) ** 2).sum() / (n - 1)
    msw = grp.apply(lambda v: ((v - v.mean()) ** 2).sum()).sum() / (k.sum() - n)
    icc1 = (msb - msw) / (msb + (kbar - 1) * msw)
    return float(icc1), float(icc1 * kbar / (1 + (kbar - 1) * icc1)), float(kbar), int(n)


def main():
    res = {"generated": datetime.datetime.utcnow().isoformat() + "Z",
           "note": ("window level reproduces the published estimate; module level is the "
                    "same computation after averaging within question_id, so the units "
                    "being aggregated no longer share text")}
    df, win, prov = E.build(ROOT)
    res["window_rows"] = int(len(win))
    for label, col in SCORES:
        w = win.dropna(subset=[col, "question_id"]).copy()
        w = w.rename(columns={E.PID: "pid", col: "val"})[["pid", "question_id", "val"]]
        i1, ik, kbar, n = icc1k(w[["pid", "val"]])
        mod = w.groupby(["pid", "question_id"], as_index=False)["val"].mean()
        m1, mk, mkbar, mn = icc1k(mod[["pid", "val"]].rename(columns={"val": "val"}))
        # odd/even module split-half, Spearman-Brown corrected
        mod = mod.sort_values(["pid", "question_id"]).copy()
        mod["rank"] = mod.groupby("pid").cumcount()
        piv = mod.assign(half=np.where(mod["rank"] % 2 == 0, "a", "b")) \
                 .groupby(["pid", "half"])["val"].mean().unstack()
        piv = piv.dropna()
        r = float(piv["a"].corr(piv["b"]))
        res[label] = {
            "window_level": {"icc_single": round(i1, 4), "icc_1k": round(ik, 4),
                             "mean_units_per_person": round(kbar, 2), "n_persons": n},
            "module_level": {"icc_single": round(m1, 4), "icc_1k": round(mk, 4),
                             "mean_modules_per_person": round(mkbar, 2), "n_persons": mn},
            "module_split_half": {"r": round(r, 4),
                                  "spearman_brown": round(2 * r / (1 + r), 4),
                                  "n_persons": int(len(piv))},
            "attenuation_factor_module_icc1k": round(float(np.sqrt(mk)), 4)}
    res["status"] = "OK"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
