"""Unit tests for the estimation core in e3_paper/e3_clsp.py.

Each statistical function is checked against an independent implementation or a closed
form, not against its own output. The final classes run the whole pipeline on the
synthetic fixture with known planted coefficients, and assert the privacy contract on
the SAFE outputs.

Run with:  python -m unittest discovery_analysis.tests.test_e3_paper_core
"""
import importlib.util
import json
import os
import subprocess
import sys
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "e3_paper"
# SAFE outputs to audit: the published copies by default, or a real run via BPD_SAFE_OUTDIR
OUTDIR = Path(os.environ.get("BPD_SAFE_OUTDIR", REPO / "outputs" / "e3"))
# the synthetic fixture is not committed; it is rebuilt from gen.py (fixed seed) on demand
SYNTH = PAPER / "synth"

_spec = importlib.util.spec_from_file_location("e3_clsp", PAPER / "e3_clsp.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

RESPONSE_ID = re.compile(r"\bR_[A-Za-z0-9]{8,}\b")


def _fake(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.4 * x1 + rng.normal(size=n)
    g = rng.choice(["w", "m", "nb"], n, p=[.7, .2, .1])
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "g": g, "y": y})


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.df = _fake()
        self.X = E.design(self.df, ["x1", "x2", "g"], ["g"])

    def test_constant_is_first_column(self):
        self.assertEqual(self.X.columns[0], "const")
        self.assertTrue((self.X["const"] == 1.0).all())

    def test_continuous_predictors_are_standardised(self):
        for c in ("x1", "x2"):
            self.assertAlmostEqual(float(self.X[c].mean()), 0.0, places=10)
            self.assertAlmostEqual(float(self.X[c].std(ddof=1)), 1.0, places=10)

    def test_categorical_is_k_minus_one_dummies(self):
        d = [c for c in self.X.columns if c.startswith("g_")]
        self.assertEqual(len(d), self.df["g"].nunique() - 1)

    def test_reference_level_is_the_first_category(self):
        d = [c for c in self.X.columns if c.startswith("g_")]
        self.assertNotIn("g_m", [])  # sanity
        self.assertNotIn("g_" + sorted(self.df["g"].unique())[0], d)

    def test_zero_variance_predictor_does_not_produce_nan(self):
        df = self.df.copy()
        df["flat"] = 3.0
        X = E.design(df, ["flat"], [])
        self.assertTrue(np.isfinite(X["flat"]).all())


class OLSTests(unittest.TestCase):
    def setUp(self):
        self.df = _fake(300, 1)
        self.X = E.design(self.df, ["x1", "x2"], [])
        self.y = self.df["y"].astype(float)
        self.fit = E.ols_hc3(self.y, self.X)

    def test_coefficients_match_least_squares(self):
        Xm = self.X.to_numpy(float)
        b = np.linalg.lstsq(Xm, self.y.to_numpy(float), rcond=None)[0]
        np.testing.assert_allclose(self.fit["beta"], b, rtol=1e-9, atol=1e-9)

    def test_hc3_standard_errors_match_the_textbook_formula(self):
        Xm = self.X.to_numpy(float)
        yv = self.y.to_numpy(float)
        XtXi = np.linalg.pinv(Xm.T @ Xm)
        b = XtXi @ Xm.T @ yv
        e = yv - Xm @ b
        h = np.einsum("ij,jk,ik->i", Xm, XtXi, Xm)
        omega = (e / (1 - h)) ** 2
        V = XtXi @ (Xm.T * omega) @ Xm @ XtXi
        np.testing.assert_allclose(self.fit["se"], np.sqrt(np.diag(V)),
                                   rtol=1e-8, atol=1e-10)

    def test_hc3_differs_from_homoskedastic_se_under_heteroskedasticity(self):
        rng = np.random.default_rng(3)
        n = 400
        x = rng.normal(size=n)
        y = x + rng.normal(0, np.abs(x) + .1, n)
        d = pd.DataFrame({"x": x, "y": y})
        X = E.design(d, ["x"], [])
        f = E.ols_hc3(d["y"].astype(float), X)
        Xm = X.to_numpy(float)
        b = np.linalg.lstsq(Xm, y, rcond=None)[0]
        e = y - Xm @ b
        s2 = (e @ e) / (n - Xm.shape[1])
        ols_se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xm.T @ Xm)))
        self.assertGreater(abs(f["se"][1] - ols_se[1]) / ols_se[1], 0.05)

    def test_residual_df_and_r2(self):
        n, p = self.X.shape
        self.assertEqual(self.fit["df_resid"], n - p)
        yv = self.y.to_numpy(float)
        r = self.fit["resid"]
        r2 = 1 - (r @ r) / ((yv - yv.mean()) ** 2).sum()
        self.assertAlmostEqual(self.fit["r2"], float(r2), places=10)

    def test_recovers_a_known_slope(self):
        rng = np.random.default_rng(8)
        n = 4000
        x = rng.normal(size=n)
        d = pd.DataFrame({"x": x, "y": 0.45 * x + rng.normal(0, 1, n)})
        f = E.ols_hc3(d["y"].astype(float), E.design(d, ["x"], []))
        self.assertAlmostEqual(float(f["beta"][1]), 0.45, delta=0.05)


class CoefFrameTests(unittest.TestCase):
    def test_confidence_interval_matches_t_critical(self):
        df = _fake(150, 2)
        fit = E.ols_hc3(df["y"].astype(float), E.design(df, ["x1"], []))
        cf = E.coef_frame(fit)
        crit = stats.t.ppf(0.975, fit["df_resid"])
        np.testing.assert_allclose(cf["ci_hi"] - cf["beta"], crit * cf["se"],
                                   rtol=1e-10)

    def test_p_values_are_two_sided(self):
        df = _fake(150, 2)
        fit = E.ols_hc3(df["y"].astype(float), E.design(df, ["x1"], []))
        cf = E.coef_frame(fit)
        np.testing.assert_allclose(
            cf["p"], 2 * stats.t.sf(np.abs(cf["t"]), fit["df_resid"]), rtol=1e-8)


class MultiplicityAndEquivalenceTests(unittest.TestCase):
    def test_bh_matches_a_hand_computed_example(self):
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
        q = E.bh_fdr(p)
        n = len(p)
        raw = p * n / (np.arange(n) + 1)
        expected = np.minimum.accumulate(raw[::-1])[::-1]
        np.testing.assert_allclose(q, expected, rtol=1e-12)

    def test_bh_is_monotone_and_bounded(self):
        rng = np.random.default_rng(4)
        p = np.sort(rng.random(40))
        q = E.bh_fdr(p)
        self.assertTrue(np.all(np.diff(q) >= -1e-12))
        self.assertTrue(np.all((q >= 0) & (q <= 1)))

    def test_bh_never_lowers_a_p_value_below_itself(self):
        p = np.array([0.02, 0.03, 0.04])
        self.assertTrue(np.all(E.bh_fdr(p) >= p - 1e-12))

    def test_tost_declares_equivalence_for_a_tight_null(self):
        r = E.tost(beta=0.0, se=0.02, dfr=500, sesoi=0.15)
        self.assertTrue(r["equivalent"])
        self.assertLess(r["p_tost"], 0.05)

    def test_tost_refuses_equivalence_for_a_wide_interval(self):
        r = E.tost(beta=0.0, se=0.20, dfr=500, sesoi=0.15)
        self.assertFalse(r["equivalent"])

    def test_tost_refuses_equivalence_for_a_large_effect(self):
        r = E.tost(beta=0.30, se=0.02, dfr=500, sesoi=0.15)
        self.assertFalse(r["equivalent"])

    def test_a_null_that_is_equivalent_at_15_may_not_be_at_10(self):
        """This is why SESOI was moved from .10 to .15 before any real-data run."""
        r15 = E.tost(beta=0.06, se=0.045, dfr=470, sesoi=0.15)
        r10 = E.tost(beta=0.06, se=0.045, dfr=470, sesoi=0.10)
        self.assertTrue(r15["equivalent"])
        self.assertFalse(r10["equivalent"])


class CollinearityTests(unittest.TestCase):
    def test_orthogonal_predictors_have_vif_near_one(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame({"const": 1.0, "a": rng.normal(size=2000),
                          "b": rng.normal(size=2000)})
        v = E.vif_table(X).set_index("term")["vif"]
        self.assertAlmostEqual(float(v["a"]), 1.0, delta=0.08)

    def test_near_duplicate_predictors_have_large_vif(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=500)
        X = pd.DataFrame({"const": 1.0, "a": a, "b": a + rng.normal(0, .01, 500)})
        self.assertGreater(E.vif_table(X).set_index("term")["vif"]["a"], 50)

    def test_vif_equals_one_over_one_minus_r2(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=400)
        b = .6 * a + rng.normal(0, .8, 400)
        X = pd.DataFrame({"const": 1.0, "a": a, "b": b})
        r = np.corrcoef(a, b)[0, 1]
        self.assertAlmostEqual(float(E.vif_table(X).set_index("term")["vif"]["a"]),
                               1 / (1 - r ** 2), places=6)

    def test_const_is_not_given_a_vif(self):
        X = pd.DataFrame({"const": 1.0, "a": np.arange(50.0)})
        self.assertNotIn("const", list(E.vif_table(X)["term"]))


class DeltaR2Tests(unittest.TestCase):
    def test_pure_noise_block_adds_almost_nothing(self):
        rng = np.random.default_rng(9)
        n = 800
        d = pd.DataFrame({"x": rng.normal(size=n), "junk": rng.normal(size=n)})
        d["y"] = d["x"] + rng.normal(0, 1, n)
        r = E.delta_r2(d["y"].astype(float), E.design(d, ["x"], []),
                       E.design(d, ["x", "junk"], []))
        self.assertLess(r["delta_r2"], 0.02)
        self.assertGreater(r["p"], 0.01)

    def test_true_predictor_block_is_detected(self):
        rng = np.random.default_rng(10)
        n = 800
        d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        d["y"] = d["x"] + 0.8 * d["z"] + rng.normal(0, 1, n)
        r = E.delta_r2(d["y"].astype(float), E.design(d, ["x"], []),
                       E.design(d, ["x", "z"], []))
        self.assertGreater(r["delta_r2"], 0.15)
        self.assertLess(r["p"], 1e-6)

    def test_f_statistic_matches_its_definition(self):
        rng = np.random.default_rng(11)
        n = 300
        d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        d["y"] = d["x"] + .5 * d["z"] + rng.normal(0, 1, n)
        Xs, Xb = E.design(d, ["x"], []), E.design(d, ["x", "z"], [])
        r = E.delta_r2(d["y"].astype(float), Xs, Xb)
        self.assertAlmostEqual(r["p"], float(stats.f.sf(r["F"], r["df1"], r["df2"])),
                               places=12)


class ReliabilityTests(unittest.TestCase):
    def test_alpha_matches_closed_form(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame(rng.normal(size=(300, 5)))
        k = 5
        exp = k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
        a, _ = E.alpha_omega(X)
        self.assertAlmostEqual(a, float(exp), places=10)

    def test_omega_exceeds_alpha_when_loadings_are_unequal(self):
        rng = np.random.default_rng(13)
        lat = rng.normal(size=800)
        X = pd.DataFrame({f"i{j}": l * lat + rng.normal(0, np.sqrt(1 - l ** 2), 800)
                          for j, l in enumerate([.9, .8, .5, .35])})
        a, om = E.alpha_omega(X)
        self.assertGreater(om, a)

    def test_too_few_rows_returns_nan_rather_than_a_number(self):
        a, om = E.alpha_omega(pd.DataFrame(np.random.default_rng(1).normal(size=(10, 3))))
        self.assertTrue(np.isnan(a) and np.isnan(om))


class ImputationTests(unittest.TestCase):
    def test_pmm_only_ever_fills_with_observed_donor_values(self):
        rng = np.random.default_rng(14)
        n = 300
        d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        d["t"] = 2 * d["a"] + rng.normal(0, 1, n)
        observed = set(np.round(d["t"].to_numpy(), 12))
        d.loc[rng.choice(n, 60, replace=False), "t"] = np.nan
        obs_vals = set(np.round(d["t"].dropna().to_numpy(), 12))
        for imp in E.pmm(d, ["t"], ["a", "b"], m=3, k=5, seed=1):
            self.assertFalse(imp["t"].isna().any())
            filled = set(np.round(imp["t"].to_numpy(), 12))
            self.assertTrue(filled.issubset(obs_vals | observed))

    def test_pmm_returns_the_requested_number_of_imputations(self):
        d = pd.DataFrame({"a": np.arange(60.0), "t": np.arange(60.0)})
        d.loc[0:9, "t"] = np.nan
        self.assertEqual(len(E.pmm(d, ["t"], ["a"], m=4, k=5, seed=2)), 4)

    def test_rubin_with_identical_imputations_reduces_to_the_within_variance(self):
        b = np.array([0.5, -0.2])
        s = np.array([0.1, 0.05])
        out = E.rubin([b, b, b, b, b], [s, s, s, s, s], ["x", "z"])
        np.testing.assert_allclose(out["se"].to_numpy(), s, rtol=1e-6)
        np.testing.assert_allclose(out["beta"].to_numpy(), b, rtol=1e-12)

    def test_rubin_inflates_se_when_imputations_disagree(self):
        s = np.array([0.10])
        spread = [np.array([v]) for v in (0.3, 0.5, 0.7, 0.9, 1.1)]
        out = E.rubin(spread, [s] * 5, ["x"])
        self.assertGreater(float(out["se"].iloc[0]), 0.10)
        self.assertGreater(float(out["fmi"].iloc[0]), 0.0)


class SyntheticPipelineTests(unittest.TestCase):
    """The fixture plants breadth +.255 and reframing -.200; the pipeline must find them."""

    @classmethod
    def setUpClass(cls):
        if not (SYNTH / "phase1_3.csv").exists():
            subprocess.run([sys.executable, str(PAPER / "gen.py")],
                           cwd=str(SYNTH.parent), check=True)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.summary = E.run(root=None, out_dir=cls.tmp.name, synthetic=str(SYNTH),
                            verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_renames_focal_columns_and_counts_windows(self):
        df, win, _ = E.build(None, synthetic=str(SYNTH))
        for k in E.FOCAL:
            self.assertIn(k, df.columns)
        counts = win.groupby(E.W_PID).size()
        merged = df.set_index(E.PID)["sbert_n_windows"].dropna()
        common = merged.index.intersection(counts.index)
        self.assertGreater(len(common), 100)
        np.testing.assert_array_equal(merged.loc[common].to_numpy(),
                                      counts.loc[common].to_numpy())

    def test_planted_breadth_coefficient_is_recovered(self):
        b = self.summary["coefficients"]["symptom_count"]["beta"]
        self.assertAlmostEqual(b, 0.255, delta=0.09)

    def test_planted_reframing_coefficient_is_recovered_with_the_right_sign(self):
        """The fixture plants -.200 on the LATENT reframing value, but the pipeline only
        ever sees a participant mean of noisy window scores, so the recovered
        coefficient is attenuated by the reliability of that mean. gen.py builds each
        window as .22*rf + noise(0,1) with 18 windows per participant, giving a
        participant-mean reliability near .22^2 / (.22^2 + 1/18) = .47. A STANDARDISED
        coefficient is attenuated by the square root of that, about .68, so the expected
        observed value is near -.14 before covariate adjustment. Asserting -.200 here
        would be asserting a quantity the estimator cannot see; AttenuationTests below
        verifies the mechanism.
        """
        b = self.summary["coefficients"]["mean_reframe"]["beta"]
        self.assertLess(b, 0)
        self.assertGreater(abs(b), 0.05)
        self.assertLess(abs(b), 0.17)

    def test_reframing_is_detected_in_the_fixture(self):
        self.assertLess(self.summary["coefficients"]["mean_reframe"]["p"], 0.05)

    def test_null_predictors_stay_near_zero(self):
        for t in ("earliest_onset", "diagnosis_lag"):
            self.assertLess(abs(self.summary["coefficients"][t]["beta"]), 0.15)

    def test_flow_counts_are_monotone_non_increasing(self):
        flow = self.summary["flow"]
        seq = [flow[k] for k in ("after_outcome", "after_language",
                                 "after_onset_breadth", "after_diagnosis_lag",
                                 "after_covariates")]
        self.assertEqual(seq, sorted(seq, reverse=True))
        self.assertEqual(flow["analytic_n"], seq[-1])

    def test_language_block_delta_r2_is_positive(self):
        self.assertGreater(self.summary["delta_r2_language"], 0.0)

    def test_sesoi_is_the_frozen_value(self):
        self.assertEqual(self.summary["equivalence_sesoi_primary"], 0.15)

    def test_no_output_file_is_named_private(self):
        for p in Path(self.tmp.name).iterdir():
            self.assertNotIn("PRIVATE", p.name)
            self.assertTrue("SAFE" in p.name or p.suffix == ".png")


class AttenuationTests(unittest.TestCase):
    """Why the fixture's -.200 comes back smaller, and what that implies for the paper.

    A participant's reframing score is the mean of noisy per-window cosines, so it is a
    fallible measure of whatever stable tendency it is meant to capture. Regression on a
    fallible predictor attenuates its STANDARDISED coefficient towards zero by the
    correlation between the true and measured score, which is the square root of the
    measure's reliability. The real-data estimate is therefore a LOWER bound on the
    association with the underlying tendency, not an unbiased estimate of it, and that is
    a limitation to state rather than a correction to apply.
    """

    @staticmethod
    def _simulate(n=6000, n_windows=18, signal_sd=0.22, beta=-0.20, seed=0):
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=n)
        y = beta * latent + rng.normal(0, np.sqrt(1 - beta ** 2), n)
        windows = (signal_sd * latent[:, None] +
                   rng.normal(0, 1.0, size=(n, n_windows)))
        measured = windows.mean(axis=1)
        d = pd.DataFrame({"m": measured, "y": y})
        fit = E.ols_hc3(d["y"].astype(float), E.design(d, ["m"], []))
        rel = signal_sd ** 2 / (signal_sd ** 2 + 1.0 / n_windows)
        return float(fit["beta"][1]), rel

    def test_observed_beta_is_attenuated_by_sqrt_reliability_of_the_mean(self):
        obs, rel = self._simulate()
        self.assertAlmostEqual(obs, -0.20 * np.sqrt(rel), delta=0.02)

    def test_the_attenuation_factor_is_not_the_reliability_itself(self):
        """Guards the docstring: using reliability rather than its square root would
        under-predict the observed coefficient by about a third."""
        obs, rel = self._simulate()
        self.assertGreater(abs(obs), abs(-0.20 * rel) + 0.02)

    def test_attenuation_shrinks_as_windows_per_participant_increase(self):
        few, _ = self._simulate(n_windows=4, seed=1)
        many, _ = self._simulate(n_windows=200, seed=1)
        self.assertLess(abs(few), abs(many))
        self.assertAlmostEqual(many, -0.20, delta=0.03)

    def test_attenuation_never_flips_the_sign(self):
        for w in (2, 5, 20, 100):
            obs, _ = self._simulate(n_windows=w, seed=2)
            self.assertLess(obs, 0)


class SafeOutputPrivacyTests(unittest.TestCase):
    """Contract on the real SAFE outputs: aggregates only, no identifiers, no text."""

    @classmethod
    def setUpClass(cls):
        if not OUTDIR.exists():
            raise unittest.SkipTest("no real output directory on this machine")
        cls.files = sorted([p for p in OUTDIR.iterdir()
                            if p.suffix in (".csv", ".json")])
        if not cls.files:
            raise unittest.SkipTest("no SAFE outputs present")

    def test_every_output_is_labelled_safe(self):
        for p in self.files:
            self.assertIn("SAFE", p.name, "unlabelled output: %s" % p.name)

    def test_no_response_id_appears_in_any_safe_output(self):
        for p in self.files:
            hit = RESPONSE_ID.search(p.read_text(errors="ignore"))
            self.assertIsNone(hit, "identifier-like token in %s" % p.name)

    def test_no_free_text_columns_are_carried_through(self):
        banned = {"window_text", "sentence_text", "narrative", "response_text",
                  "Q108", "text"}
        for p in self.files:
            if p.suffix != ".csv":
                continue
            cols = set(pd.read_csv(p, nrows=0).columns)
            self.assertFalse(cols & banned, "%s carries %s" % (p.name, cols & banned))

    def test_safe_csvs_are_aggregates_not_participant_rows(self):
        for p in self.files:
            if p.suffix != ".csv":
                continue
            n = sum(1 for _ in p.open()) - 1
            self.assertLess(n, 100, "%s has %d rows; SAFE files are aggregates" %
                            (p.name, n))

    def test_json_outputs_parse_and_carry_no_row_level_arrays(self):
        for p in self.files:
            if p.suffix != ".json":
                continue
            obj = json.loads(p.read_text())

            def walk(o):
                if isinstance(o, list):
                    self.assertLess(len(o), 100, "long array in %s" % p.name)
                    for i in o:
                        walk(i)
                elif isinstance(o, dict):
                    for v in o.values():
                        walk(v)
            walk(obj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
