"""R2c - identify the exact archived scoring convention.

R2a found that the archived sbert_reframing_similarity is NOT the mean of the eight
per-prototype cosines. It is that quantity multiplied by a constant 1.4482, with an
intercept of -3.4e-08 and residuals at floating-point scale (max 3.8e-07). A pure
multiplicative constant with no intercept has an exact explanation available:

    cos(w, Pbar/||Pbar||) = w . Pbar / ||Pbar|| = mean_cosine / ||Pbar||

where Pbar is the arithmetic mean of the eight unit-norm prototype embeddings. That is,
scoring each window against the NORMALISED CENTROID of the prototype set gives exactly
the mean-of-cosines divided by ||Pbar||. If 1/1.4482 = 0.6905 equals ||Pbar||, the
archived convention is the centroid one, and the two differ only by a constant scale -
which is why every standardised regression coefficient is unaffected.

This encodes 16 sentences, not 13,017 windows, so it does not need Slurm.
"""
import json, os
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from paths import root as bpd_root, MINILM_PATH
OUT = Path(bpd_root()) / "discovery_analysis/output/e3"
OBSERVED_SLOPE = 1.4482053131538417   # from r2a_identity_check_SAFE.csv

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
m = SentenceTransformer(MINILM_PATH, local_files_only=True, device="cpu")


def stats(name, sents, observed_slope=None):
    P = m.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    centroid = P.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    G = P @ P.T
    off = G[~np.eye(len(P), dtype=bool)]
    d = {"set": name, "n_prototypes": len(sents),
         "centroid_norm": norm,
         "implied_scale_factor_1_over_norm": 1.0 / norm,
         "mean_pairwise_cosine": float(off.mean()),
         "min_pairwise_cosine": float(off.min()),
         "max_pairwise_cosine": float(off.max())}
    if observed_slope is not None:
        d["observed_calibration_slope_from_R2a"] = observed_slope
        d["difference"] = abs(1.0 / norm - observed_slope)
        d["matches_centroid_convention"] = bool(d["difference"] < 1e-5)
    return d


res = {"reframing": stats("reframing", REFRAME, OBSERVED_SLOPE),
       "black_and_white": stats("black_and_white", BW)}
res["conclusion"] = (
    "The archived score is cosine similarity to the NORMALISED CENTROID of the "
    "prototype set. It equals the mean of the per-prototype cosines divided by the "
    "centroid norm, so the two are proportional: correlation 1 by construction, and "
    "identical standardised regression coefficients. Methods should describe the "
    "centroid convention, not 'mean cosine over the eight prototypes'."
    if res["reframing"].get("matches_centroid_convention") else
    "The centroid convention does NOT explain the observed scale factor; the archived "
    "convention remains unidentified and must not be described as either candidate.")
(OUT / "r2c_convention_SAFE.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
