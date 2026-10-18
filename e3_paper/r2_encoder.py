"""R2 - encoder identity check (R2a) and paired bootstrap of the encoder difference (R2b).

Decision from the 5 Sep review:
  R2a  r = 1.0000 to four decimals establishes a near-perfect LINEAR relationship, not
       element-wise identity: an affine rescaling would also give r = 1. Verify the row
       count and the join, then report MAE, max |diff| and the calibration slope and
       intercept against the archived sbert_reframing_similarity.
  R2b  The claim that the MPNet estimate (-.265) differs from the MiniLM estimate
       (-.216) is not permitted without a test. Participant-cluster PAIRED bootstrap:
       both encoders are refit on the SAME resampled participants, and the statistic is
       the difference between the two focal coefficients.

Run under Slurm (R7). Fully offline. Reads window text (PRIVATE); prints aggregates only.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import (design, ols_hc3, coef_frame, PID, FOCAL, COVARS, CATEGORICALS,
                     OUTCOME, build, W_PID)
# Prototypes and model paths are copied verbatim from e2.py rather than imported,
# because importing e2 would execute the whole E2 analysis and overwrite its outputs.
# r2_prototype_check_SAFE.json below records that they still match e2.py byte for byte.
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
from paths import MINILM_PATH, MPNET_PATH

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
INT = ROOT / "discovery_analysis/intermediate_PRIVATE"
WT = INT / "window_table_augmented_PRIVATE.csv"
SEED = 20260906
NBOOT = 2000
CONVENTION = "mean_over_prototypes"   # established by E2, re-verified below

from sentence_transformers import SentenceTransformer
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % DEV, flush=True)


def enc(model, xs, bs=128):
    return model.encode(xs, batch_size=bs, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------- prototypes still match e2.py
_e2src = (Path(__file__).resolve().parent / "e2.py").read_text()
_missing = [s for s in REFRAME + BW if s not in _e2src]
proto_check = {"n_reframe": len(REFRAME), "n_blackwhite": len(BW),
               "sentences_not_found_in_e2_py": _missing,
               "identical_to_e2_py": not _missing}
(OUT / "r2_prototype_check_SAFE.json").write_text(json.dumps(proto_check, indent=2))
assert not _missing, "prototype sentences have drifted from e2.py: %r" % _missing
print("prototypes verified against e2.py: 8 reframing + 8 black-and-white", flush=True)

# ------------------------------------------------------------------ load + join audit
raw = pd.read_csv(WT, usecols=[W_PID, "window_text", "sbert_reframing_similarity",
                               "sbert_blackwhite_similarity"], low_memory=False)
join = {
    "rows_in_window_table": int(len(raw)),
    "window_text_nonnull": int(raw["window_text"].notna().sum()),
    "stored_reframing_nonnull": int(raw["sbert_reframing_similarity"].notna().sum()),
    "text_nonnull_and_score_nonnull": int((raw["window_text"].notna() &
                                           raw["sbert_reframing_similarity"].notna()).sum()),
    "text_nonnull_but_score_null": int((raw["window_text"].notna() &
                                        raw["sbert_reframing_similarity"].isna()).sum()),
    "text_null_but_score_nonnull": int((raw["window_text"].isna() &
                                        raw["sbert_reframing_similarity"].notna()).sum()),
}
win = raw[raw["window_text"].notna()].reset_index(drop=True)
join["rows_compared"] = int(len(win))
join["alignment"] = ("positional within the single source CSV after dropping null "
                     "window_text; no key-based merge is involved, so row order is the join")
print(json.dumps(join, indent=2), flush=True)

texts = win["window_text"].astype(str).tolist()
stored = win["sbert_reframing_similarity"].to_numpy(float)

# ------------------------------------------------------------------ encode
t0 = time.time()
mini = SentenceTransformer(MINILM_PATH, local_files_only=True, device=DEV)
E = enc(mini, texts)
sim_r = E @ enc(mini, REFRAME).T
sim_b = E @ enc(mini, BW).T
print("MiniLM encoded in %.1fs" % (time.time() - t0), flush=True)

mi_r = sim_r.mean(1) if CONVENTION == "mean_over_prototypes" else sim_r.max(1)
mi_b = sim_b.mean(1) if CONVENTION == "mean_over_prototypes" else sim_b.max(1)

t0 = time.time()
mp = SentenceTransformer(MPNET_PATH, local_files_only=True, device=DEV)
Em = enc(mp, texts)
sr = Em @ enc(mp, REFRAME).T
sb = Em @ enc(mp, BW).T
print("MPNet encoded in %.1fs" % (time.time() - t0), flush=True)
mp_r = sr.mean(1) if CONVENTION == "mean_over_prototypes" else sr.max(1)
mp_b = sb.mean(1) if CONVENTION == "mean_over_prototypes" else sb.max(1)

# ------------------------------------------------------------------ R2a identity check
ok = ~np.isnan(stored)
d = mi_r[ok] - stored[ok]
sl, ic = np.polyfit(mi_r[ok], stored[ok], 1)
resid = stored[ok] - (sl * mi_r[ok] + ic)
r2a = {
    "convention": CONVENTION,
    "n_compared": int(ok.sum()),
    "pearson_r": float(np.corrcoef(mi_r[ok], stored[ok])[0, 1]),
    "pearson_r_8dp": round(float(np.corrcoef(mi_r[ok], stored[ok])[0, 1]), 8),
    "mean_recomputed": float(mi_r[ok].mean()),
    "mean_archived": float(stored[ok].mean()),
    "sd_recomputed": float(mi_r[ok].std(ddof=1)),
    "sd_archived": float(stored[ok].std(ddof=1)),
    "mean_signed_difference": float(d.mean()),
    "MAE": float(np.abs(d).mean()),
    "RMSE": float(np.sqrt((d ** 2).mean())),
    "max_abs_difference": float(np.abs(d).max()),
    "p99_abs_difference": float(np.percentile(np.abs(d), 99)),
    "n_exact_to_1e_12": int((np.abs(d) < 1e-12).sum()),
    "n_exact_to_1e_6": int((np.abs(d) < 1e-6).sum()),
    "calibration_slope": float(sl),
    "calibration_intercept": float(ic),
    "calibration_residual_sd": float(resid.std(ddof=1)),
    "calibration_max_abs_residual": float(np.abs(resid).max()),
}
r2a["verdict"] = (
    "element-wise identical" if r2a["max_abs_difference"] < 1e-6 else
    ("affine rescaling of the archived score (slope %.4f, intercept %.4f), not identity"
     % (sl, ic)) if r2a["calibration_max_abs_residual"] < 1e-4 else
    "perfectly correlated but neither identical nor a clean affine rescaling")
print("\n=== R2a identity check ===")
print(json.dumps(r2a, indent=2), flush=True)

# ------------------------------------------------------------------ paired analytic frame
def agg(rv, bv, suffix):
    t = pd.DataFrame({W_PID: win[W_PID], "r": rv, "b": bv}).groupby(W_PID)
    return pd.DataFrame({"mean_reframe" + suffix: t["r"].mean(),
                         "mean_bw" + suffix: t["b"].mean()})


BASE, _u1, _u2 = build(str(ROOT))
PREDS = list(FOCAL.keys()) + COVARS
A = (BASE.drop(columns=["mean_reframe", "mean_bw"])
         .merge(agg(mi_r, mi_b, "_mi"), left_on=PID, right_index=True, how="inner")
         .merge(agg(mp_r, mp_b, "_mp"), left_on=PID, right_index=True, how="inner"))
other = [c for c in PREDS if c not in ("mean_reframe", "mean_bw")]
A = A.dropna(subset=[OUTCOME] + other + ["mean_reframe_mi", "mean_bw_mi",
                                         "mean_reframe_mp", "mean_bw_mp"])
A = A.reset_index(drop=True)
print("\npaired analytic n = %d" % len(A), flush=True)

(INT / "r2_participant_encoder_scores_PRIVATE.csv").write_text(
    A[[PID, "mean_reframe_mi", "mean_bw_mi", "mean_reframe_mp",
       "mean_bw_mp"]].to_csv(index=False))


def beta_for(frame, suffix):
    f = frame.rename(columns={"mean_reframe" + suffix: "mean_reframe",
                              "mean_bw" + suffix: "mean_bw"})
    X = design(f, PREDS, CATEGORICALS)
    fit = ols_hc3(f[OUTCOME].astype(float), X)
    cf = coef_frame(fit)
    return cf.loc[cf["term"] == "mean_reframe"].iloc[0]


obs_mi = beta_for(A, "_mi")
obs_mp = beta_for(A, "_mp")
obs_diff = float(obs_mp["beta"] - obs_mi["beta"])
print("observed MiniLM beta %.4f   MPNet beta %.4f   difference %.4f"
      % (obs_mi["beta"], obs_mp["beta"], obs_diff), flush=True)

# ------------------------------------------------------------------ R2b paired bootstrap
rng = np.random.default_rng(SEED)
n = len(A)
bmi, bmp, bdf = [], [], []
t0 = time.time()
for i in range(NBOOT):
    idx = rng.integers(0, n, n)
    s = A.iloc[idx].reset_index(drop=True)
    try:
        a = beta_for(s, "_mi")["beta"]
        b = beta_for(s, "_mp")["beta"]
    except Exception:
        continue
    bmi.append(float(a)); bmp.append(float(b)); bdf.append(float(b - a))
    if (i + 1) % 500 == 0:
        print("  bootstrap %d/%d  (%.0fs)" % (i + 1, NBOOT, time.time() - t0), flush=True)

bmi = np.array(bmi); bmp = np.array(bmp); bdf = np.array(bdf)
p_boot = 2 * min((bdf <= 0).mean(), (bdf >= 0).mean())
r2b = {
    "n_participants": int(n),
    "n_bootstrap_completed": int(len(bdf)),
    "seed": SEED,
    "resampling_unit": "participant (paired: both encoders refit on the same resample)",
    "beta_minilm_observed": round(float(obs_mi["beta"]), 4),
    "beta_minilm_hc3_se": round(float(obs_mi["se"]), 4),
    "beta_mpnet_observed": round(float(obs_mp["beta"]), 4),
    "beta_mpnet_hc3_se": round(float(obs_mp["se"]), 4),
    "observed_difference_mpnet_minus_minilm": round(obs_diff, 4),
    "bootstrap_mean_difference": round(float(bdf.mean()), 4),
    "bootstrap_se_difference": round(float(bdf.std(ddof=1)), 4),
    "difference_ci95": [round(float(np.percentile(bdf, 2.5)), 4),
                        round(float(np.percentile(bdf, 97.5)), 4)],
    "difference_ci99": [round(float(np.percentile(bdf, 0.5)), 4),
                        round(float(np.percentile(bdf, 99.5)), 4)],
    "bootstrap_p_two_sided": round(float(p_boot), 4),
    "pct_bootstrap_draws_mpnet_more_negative": round(float((bdf < 0).mean() * 100), 1),
    "corr_of_bootstrap_betas": round(float(np.corrcoef(bmi, bmp)[0, 1]), 4),
    "beta_minilm_ci95": [round(float(np.percentile(bmi, 2.5)), 4),
                         round(float(np.percentile(bmi, 97.5)), 4)],
    "beta_mpnet_ci95": [round(float(np.percentile(bmp, 2.5)), 4),
                        round(float(np.percentile(bmp, 97.5)), 4)],
}
r2b["conclusion"] = ("the two encoder estimates DIFFER (95% CI on the difference excludes 0)"
                     if not (r2b["difference_ci95"][0] <= 0 <= r2b["difference_ci95"][1])
                     else "the two encoder estimates are NOT distinguishable; report the "
                          "MPNet estimate as descriptively larger only")

pd.DataFrame({"boot_beta_minilm": bmi, "boot_beta_mpnet": bmp,
              "boot_difference": bdf}).describe().to_csv(
    OUT / "r2b_bootstrap_describe_SAFE.csv")
pd.DataFrame([r2a]).to_csv(OUT / "r2a_identity_check_SAFE.csv", index=False)
pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v)
               for k, v in r2b.items()}]).to_csv(
    OUT / "r2b_paired_bootstrap_SAFE.csv", index=False)
(OUT / "r2_report_SAFE.json").write_text(json.dumps(
    {"join_audit": join, "r2a_identity": r2a, "r2b_paired_bootstrap": r2b}, indent=2))

print("\n=== R2b paired bootstrap ===")
print(json.dumps(r2b, indent=2))
print("\nR2 OK")
