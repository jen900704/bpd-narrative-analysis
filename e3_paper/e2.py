"""E2 - robustness of the reframing measure.

The July per-window scoring convention (mean cosine to the prototype set vs max) was
not recorded, so this script does not assume it. It re-encodes with the ORIGINAL
encoder (MiniLM), computes both candidate conventions, and correlates each against the
stored sbert_reframing_similarity. Whichever reproduces the stored score is used for
every comparison. If neither reaches r >= .95 the script stops and says so, rather than
silently comparing incomparable numbers.

E2a  encoder robustness: rescore with all-mpnet-base-v2, refit the E3 model.
E2b  prototype robustness: leave-one-prototype-out over the 8 reframing sentences.

Fully offline. Reads window text (PRIVATE); prints and writes only aggregates.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, PID, FOCAL, COVARS, CATEGORICALS,
                     OUTCOME, build, W_PID)

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
WT = ROOT / "discovery_analysis/intermediate_PRIVATE/window_table_augmented_PRIVATE.csv"

REFRAME = [
 "At first I assumed the worst, but later I realized there were other explanations.",
 "I reconsidered my first interpretation and saw the situation in a more balanced way.",
 "After reflecting, I understood that the other person may have had reasons unrelated to me.",
 "I noticed that my initial reaction was extreme and changed how I understood the event.",
 "I was able to see both sides and recognize that the situation was more complicated.",
 "Instead of assuming rejection, I considered a less threatening explanation.",
 "My perspective changed after I calmed down and thought about the context.",
 "I recognized an exception to my original belief and responded differently."]
BW = [
 "They are always terrible and will never change.",
 "If someone disappoints me, they are completely bad.",
 "Everything was ruined and nothing could make it better.",
 "I was either perfect or worthless, with nothing in between.",
 "They either loved me completely or hated me.",
 "One mistake proved that the whole relationship was a failure.",
 "I knew for certain that everyone would abandon me.",
 "There were only two possibilities, and both were extreme."]

from sentence_transformers import SentenceTransformer
import torch
from paths import MINILM_PATH, MPNET_PATH
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device: " + DEV, flush=True)

win = pd.read_csv(WT, usecols=[W_PID, "window_text", "sbert_reframing_similarity",
                               "sbert_blackwhite_similarity"], low_memory=False)
win = win[win["window_text"].notna()].reset_index(drop=True)
print("windows with text: %d" % len(win), flush=True)
texts = win["window_text"].astype(str).tolist()

def enc(model, xs, bs=128):
    return model.encode(xs, batch_size=bs, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)

report = {"n_windows": int(len(win)), "device": DEV}

mini = SentenceTransformer(MINILM_PATH, local_files_only=True, device=DEV)
E = enc(mini, texts)
Pr = enc(mini, REFRAME)
Pb = enc(mini, BW)
sim_r = E @ Pr.T
sim_b = E @ Pb.T
cands = {"mean_over_prototypes": (sim_r.mean(1), sim_b.mean(1)),
         "max_over_prototypes": (sim_r.max(1), sim_b.max(1))}
stored_r = win["sbert_reframing_similarity"].to_numpy(float)
best = None
bestr = -1.0
for nm in cands:
    c = float(np.corrcoef(cands[nm][0], stored_r)[0, 1])
    report["reproduce_r_" + nm] = round(c, 4)
    print("  convention %s: r with stored score = %.4f" % (nm, c), flush=True)
    if c > bestr:
        best = nm
        bestr = c
report["convention_used"] = best
report["convention_r"] = round(bestr, 4)
if bestr < 0.95:
    report["STOP"] = "no candidate convention reproduces the stored score"
    (OUT / "e2_report_SAFE.json").write_text(json.dumps(report, indent=2))
    print("STOPPING: cannot reproduce the July convention (best r = %.3f)" % bestr)
    sys.exit(0)

def agg(rvals, bvals):
    d = pd.DataFrame({W_PID: win[W_PID], "r": rvals, "b": bvals})
    g = d.groupby(W_PID)
    return pd.DataFrame({"mean_reframe": g["r"].mean(), "mean_bw": g["b"].mean()})

BASE, _u1, _u2 = build(str(ROOT))
PREDS = list(FOCAL.keys()) + COVARS

def refit(agg_df, label):
    d = (BASE.drop(columns=["mean_reframe", "mean_bw"])
             .merge(agg_df, left_on=PID, right_index=True, how="inner"))
    d = d.dropna(subset=[OUTCOME] + [c for c in PREDS if c in d.columns])
    cf = coef_frame(ols_hc3(d[OUTCOME].astype(float), design(d, PREDS, CATEGORICALS)))
    r = cf.loc[cf["term"] == "mean_reframe"].iloc[0]
    return {"analysis": label, "n": int(len(d)), "beta": round(r["beta"], 4),
            "se": round(r["se"], 4), "ci_lo": round(r["ci_lo"], 4),
            "ci_hi": round(r["ci_hi"], 4), "p": round(r["p"], 6)}

rows = [refit(agg(cands[best][0], cands[best][1]), "MiniLM re-scored (" + best + ")")]

mp = SentenceTransformer(MPNET_PATH, local_files_only=True, device=DEV)
Em = enc(mp, texts)
Prm = enc(mp, REFRAME)
Pbm = enc(mp, BW)
sr = Em @ Prm.T
sb = Em @ Pbm.T
if best == "mean_over_prototypes":
    mr = sr.mean(1); mb = sb.mean(1)
else:
    mr = sr.max(1); mb = sb.max(1)
report["minilm_mpnet_window_corr"] = round(float(np.corrcoef(cands[best][0], mr)[0, 1]), 4)
agg_mp = agg(mr, mb)
agg_mi = agg(cands[best][0], cands[best][1])
report["minilm_mpnet_participant_corr"] = round(
    float(agg_mi["mean_reframe"].corr(agg_mp["mean_reframe"])), 4)
rows.append(refit(agg_mp, "MPNet (E2a encoder robustness)"))

loo = []
for k in range(len(REFRAME)):
    keep = [j for j in range(len(REFRAME)) if j != k]
    if best == "mean_over_prototypes":
        rv = sim_r[:, keep].mean(1)
    else:
        rv = sim_r[:, keep].max(1)
    res = refit(agg(rv, cands[best][1]), "drop prototype %d" % (k + 1))
    res["dropped_prototype_index"] = k + 1
    loo.append(res)
loo_df = pd.DataFrame(loo)
loo_df.to_csv(OUT / "e2b_leave_one_prototype_out_SAFE.csv", index=False)
report["loo_beta_min"] = float(loo_df["beta"].min())
report["loo_beta_max"] = float(loo_df["beta"].max())
report["loo_all_negative_and_p05"] = bool((loo_df["beta"] < 0).all() and (loo_df["p"] < .05).all())

res = pd.DataFrame(rows)
res.to_csv(OUT / "e2_encoder_robustness_SAFE.csv", index=False)
(OUT / "e2_report_SAFE.json").write_text(json.dumps(report, indent=2))

print("")
print("=== E2 report ===")
for k in report:
    print("  %s: %s" % (k, report[k]))
print("")
print("=== E2a encoder ===")
print(res.to_string(index=False))
print("")
print("=== E2b leave-one-prototype-out ===")
print(loo_df[["dropped_prototype_index", "n", "beta", "ci_lo", "ci_hi", "p"]].to_string(index=False))
