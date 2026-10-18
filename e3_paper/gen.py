"""Build a synthetic pair with the real CLSP column names and known planted effects."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(7)
n = 540
pid = [f"R{i:05d}" for i in range(n)]
b = np.clip(rng.binomial(14, .62, n), 1, 14)
on = np.clip(rng.normal(12.5 - .25 * (b - b.mean()), 4.5, n), 5, 45).round()
am = np.clip(rng.normal(31, 9, n), 18, 70).round()
dx = np.clip(on + rng.gamma(3, 3.2, n), on, 70).round()
dx[rng.random(n) < .01] = np.nan
lag = dx - on
rf = rng.normal(0, 1, n); bw = rng.normal(0, 1, n) - .25 * rf
wc = np.clip(rng.lognormal(5.85, .62, n), 20, 6000).round()
z = lambda a: (np.asarray(a, float) - np.nanmean(a)) / np.nanstd(a, ddof=1)
lin = .255 * z(b) - .200 * rf + .05 * bw + .08 * z(am) + .05 * z(np.log(wc))
lat = lin + rng.normal(0, np.sqrt(max(1 - lin.var(), .05)), n)
d = {"ResponseId": pid, "PAST_narrative_word_count_total": wc.astype(int),
     "symptom_count": b, "earliest_onset": on,
     "earliest_censored_low": (on <= 5).astype(int),
     "diagnosis_lag_direct": lag, "current_age_midpoint": am,
     "gender_collapsed": rng.choice(["w", "m", "nb"], n, p=[.74, .12, .14])}
def items(pre, k, load, off, miss=0.0):
    for j in range(1, k + 1):
        eta = load * lat + rng.normal(0, .75, n) + off
        v = (np.digitize(eta, np.quantile(eta, [.18, .4, .64, .85])) + 1).astype(float)
        if miss: v[rng.random(n) < miss] = np.nan
        d[f"{pre}_{j}_num"] = v
items("Q92", 8, .95, .35); items("Q93", 4, .90, -.30); items("Q109", 3, -.55, .10, .155)
p = pd.DataFrame(d)
p["thoughts_feelings_primary"] = p[[f"Q92_{j}_num" for j in range(1, 9)]].mean(axis=1)
p["negative_behaviors_primary"] = p[[f"Q93_{j}_num" for j in range(1, 5)]].mean(axis=1)
p["positive_behaviors_primary"] = p[[f"Q109_{j}_num" for j in range(1, 4)]].mean(axis=1)
p["overall_negative_primary"] = p[[f"Q92_{j}_num" for j in range(1, 9)] +
                                  [f"Q93_{j}_num" for j in range(1, 5)]].mean(axis=1)
QS = [f"Q{q}" for q in [7, 10, 17, 20, 34, 37, 46, 53, 58, 64, 68, 108]]
rows = []; hs = np.zeros(n, bool); hs[rng.choice(n, 479, replace=False)] = True
for i in range(n):
    if not hs[i]: continue
    for q in rng.choice(QS, 9, replace=False):
        ev = float(np.clip(rng.normal(on[i] + 6, 3), 5, am[i]))
        qe = .35 if q in ("Q64", "Q68", "Q53") else 0.
        for _ in range(2):
            rows.append((pid[i], q, .22 * rf[i] + qe + rng.normal(0, 1),
                         .22 * bw[i] - .2 * qe + rng.normal(0, 1), am[i] - ev))
w = pd.DataFrame(rows, columns=["ResponseId", "question_id",
                                "sbert_reframing_similarity",
                                "sbert_blackwhite_similarity", "years_ago"])
for c in ("sbert_reframing_similarity", "sbert_blackwhite_similarity"):
    v = w[c].to_numpy(); w[c] = .22 + .055 * (v - v.mean()) / v.std(ddof=1)
w["sentence_text"] = "SYNTHETIC"; w["window_text"] = "SYNTHETIC"
os.makedirs("synth", exist_ok=True)
p.to_csv("synth/phase1_3.csv", index=False); w.to_csv("synth/windows.csv", index=False)
print(f"synthetic built: {len(p)} participants, {len(w)} windows")
print("PLANTED symptom_count=.255 mean_reframe=-.200 earliest_onset=0 diagnosis_lag=0")
