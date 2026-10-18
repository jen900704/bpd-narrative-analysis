"""R8 - how reliable is the participant-level reframing score?

NOT in the original R1-R7 list. It was forced by writing the unit tests: the synthetic
fixture plants a latent reframing effect of -.200, but the pipeline recovers about -.11,
because the pipeline never sees the latent value - it sees a participant MEAN of noisy
per-window cosines. A standardised coefficient on a fallible predictor is attenuated by
the correlation between the true and the measured score, i.e. by the SQUARE ROOT of that
measure's reliability.

That is a property of the real analysis too, and it is measurable from the existing
window table without re-encoding anything. This script estimates it two ways:

  1. one-way random-effects ICC of window-level scores within participant, and the
     Spearman-Brown reliability of the mean of k windows;
  2. an empirical split-half: split each participant's windows at random into two
     halves, correlate the two half-means across participants, Spearman-Brown up.

WHAT THIS DOES AND DOES NOT LICENCE. It quantifies how much of the between-participant
variance in the score is stable rather than window-level noise. It says nothing about
whether the stable part measures reframing as a person would recognise it - that is what
E1 (blinded human coding) is for. The disattenuated coefficient below is printed as an
UPPER REFERENCE only. The reported estimate stays the observed one; the honest sentence
is that the observed association is attenuated by measurement error and is therefore a
lower bound on the association with the stable component of the measure.

Reads only the two numeric similarity columns. No text. SAFE outputs only.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_clsp import W_PID, W_REF, W_BW

from paths import root as bpd_root
ROOT = Path(bpd_root())
OUT = ROOT / "discovery_analysis/output/e3"
WT = ROOT / "discovery_analysis/intermediate_PRIVATE/window_table_augmented_PRIVATE.csv"
SEED = 20260906
NBOOT = 1000

win = pd.read_csv(WT, usecols=[W_PID, W_REF, W_BW], low_memory=False)
win = win.dropna(subset=[W_REF])


def suff_stats(frame, col):
    """Per-participant (n_i, sum, sum of squares). ICC needs nothing else, and
    resampling participants is then just resampling these rows."""
    g = frame.groupby(W_PID)[col]
    return np.column_stack([g.size().to_numpy(float),
                            g.sum().to_numpy(float),
                            g.apply(lambda s: float((s.astype(float) ** 2).sum()))
                             .to_numpy(float)])


def icc1_from_stats(S):
    """One-way random-effects ICC(1,1) from sufficient statistics."""
    n_i, s_i, q_i = S[:, 0], S[:, 1], S[:, 2]
    k = len(n_i)
    N = n_i.sum()
    ssw = float((q_i - s_i ** 2 / n_i).sum())
    ssb = float((s_i ** 2 / n_i).sum() - s_i.sum() ** 2 / N)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    k0 = (N - (n_i ** 2).sum() / N) / (k - 1)      # standard unbalanced k0
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return float(icc), float(k0), float(msb), float(msw), int(k), int(N)


def icc1(frame, col):
    return icc1_from_stats(suff_stats(frame, col))


def spearman_brown(icc, k):
    return k * icc / (1 + (k - 1) * icc)


def split_half(frame, col, seed):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _, g in frame.groupby(W_PID):
        if len(g) < 2:
            continue
        idx = rng.permutation(len(g))
        h = len(g) // 2
        a.append(float(g[col].to_numpy()[idx[:h]].mean()))
        b.append(float(g[col].to_numpy()[idx[h:2 * h]].mean()))
    r = float(np.corrcoef(a, b)[0, 1])
    return r, spearman_brown(r, 2), len(a)


res = {}
for label, col in [("reframing", W_REF), ("black_and_white", W_BW)]:
    sub = win.dropna(subset=[col])
    icc, k0, msb, msw, kpar, N = icc1(sub, col)
    rel = spearman_brown(icc, k0)
    r_half, r_sb, n_half = split_half(sub, col, SEED)
    res[label] = {
        "n_participants": kpar, "n_windows": N,
        "mean_windows_per_participant": round(N / kpar, 2),
        "k0_unbalanced": round(k0, 3),
        "MS_between": msb, "MS_within": msw,
        "ICC_1_1_single_window": round(icc, 4),
        "reliability_of_participant_mean_ICC_1_k": round(rel, 4),
        "attenuation_factor_sqrt_reliability": round(float(np.sqrt(max(rel, 0))), 4),
        "split_half_r_raw": round(r_half, 4),
        "split_half_spearman_brown_2_halves": round(r_sb, 4),
        "split_half_n_participants": n_half,
    }

# bootstrap the reliability of the reframing mean over participants
rng = np.random.default_rng(SEED)
S = suff_stats(win.dropna(subset=[W_REF]), W_REF)
boots = []
for _ in range(NBOOT):
    idx = rng.integers(0, len(S), len(S))     # resample participants, keep them distinct
    icc, k0, *_ = icc1_from_stats(S[idx])
    boots.append(spearman_brown(icc, k0))
boots = np.array(boots)
res["reframing"]["reliability_ci95"] = [round(float(np.percentile(boots, 2.5)), 4),
                                        round(float(np.percentile(boots, 97.5)), 4)]

obs_beta = -0.216
att = res["reframing"]["attenuation_factor_sqrt_reliability"]
res["interpretation"] = {
    "observed_reframing_beta_E3": obs_beta,
    "attenuation_factor_used": att,
    "upper_reference_disattenuated_beta": round(obs_beta / att, 4) if att > 0 else None,
    "reported_estimate": ("the OBSERVED coefficient. The disattenuated value is an upper "
                          "reference under classical measurement error and is not the "
                          "reported result."),
    "what_this_establishes": ("how much of the between-participant variance in the "
                              "cosine score is stable across a participant's windows"),
    "what_this_does_not_establish": ("that the stable component measures reframing as a "
                                     "human coder would recognise it; that is E1"),
    "wording": ("The participant-level reframing score is a mean of noisy per-window "
                "cosines with an estimated reliability of %.2f, so the reported "
                "coefficient is attenuated and should be read as a lower bound on the "
                "association with the stable component of the measure."
                % res["reframing"]["reliability_of_participant_mean_ICC_1_k"]),
}

pd.DataFrame([{"measure": k, **{kk: vv for kk, vv in v.items()
                                if not isinstance(vv, (list, dict))}}
              for k, v in res.items() if k != "interpretation"]).to_csv(
    OUT / "r8_measure_reliability_SAFE.csv", index=False)
(OUT / "r8_measure_reliability_SAFE.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print("\nR8 OK")
