"""Unit tests for e3_paper/paperlib.py - the shared logic behind R1-R4.

Every case here has an answer that can be checked by hand or against a closed form,
so a passing test means the function computes what it claims, not merely that it runs.
Run with:  python -m unittest discovery_analysis.tests.test_e3_paper_lib
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parents[1] / "e3_paper" / "paperlib.py"
_spec = importlib.util.spec_from_file_location("paperlib", LIB)
paperlib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paperlib)


class YearsAgoTests(unittest.TestCase):
    def test_simple_difference(self):
        y = paperlib.build_years_ago([30.0, 25.0, 40.0], [12.0, 5.0, 18.0])
        self.assertEqual(list(y), [18.0, 20.0, 22.0])

    def test_negative_becomes_missing_not_zero(self):
        y = paperlib.build_years_ago([20.0], [25.0])
        self.assertTrue(np.isnan(y.iloc[0]))

    def test_exact_zero_is_kept(self):
        y = paperlib.build_years_ago([20.0], [20.0])
        self.assertEqual(y.iloc[0], 0.0)

    def test_missing_onset_propagates(self):
        y = paperlib.build_years_ago([30.0, 30.0], [np.nan, 10.0])
        self.assertTrue(np.isnan(y.iloc[0]))
        self.assertEqual(y.iloc[1], 20.0)

    def test_missing_current_age_propagates(self):
        y = paperlib.build_years_ago([np.nan], [10.0])
        self.assertTrue(np.isnan(y.iloc[0]))

    def test_ignores_index_alignment(self):
        a = pd.Series([30.0, 40.0], index=[7, 9])
        b = pd.Series([10.0, 15.0], index=[0, 1])
        self.assertEqual(list(paperlib.build_years_ago(a, b)), [20.0, 25.0])


class PrimaryFamilyTests(unittest.TestCase):
    def setUp(self):
        self.f = pd.DataFrame({"symptom_domain": ["a", "b", "suicidal feelings",
                                                  "a", "suicidal feelings"],
                               "v": [1, 2, 3, 4, 5]})

    def test_counts_add_back_to_the_whole(self):
        prim, prox = paperlib.split_primary_family(self.f, "symptom_domain",
                                                   "suicidal feelings")
        self.assertEqual(len(prim), 3)
        self.assertEqual(len(prox), 2)
        self.assertEqual(len(prim) + len(prox), len(self.f))

    def test_proxy_domain_absent_from_primary(self):
        prim, _ = paperlib.split_primary_family(self.f, "symptom_domain",
                                                "suicidal feelings")
        self.assertNotIn("suicidal feelings", set(prim["symptom_domain"]))
        self.assertEqual(prim["symptom_domain"].nunique(), 2)

    def test_nonsuicidal_self_injury_is_not_treated_as_the_proxy(self):
        f = pd.DataFrame({"symptom_domain": ["nonsuicidal self-injury",
                                             "suicidal feelings"]})
        prim, prox = paperlib.split_primary_family(f, "symptom_domain",
                                                   "suicidal feelings")
        self.assertEqual(list(prim["symptom_domain"]), ["nonsuicidal self-injury"])

    def test_missing_proxy_raises_rather_than_silently_passing_through(self):
        with self.assertRaises(KeyError):
            paperlib.split_primary_family(self.f, "symptom_domain", "not a domain")


class HeapingTests(unittest.TestCase):
    def test_tens_are_nested_inside_fives(self):
        h = paperlib.heaping_counts([10, 15, 20, 12, 13])
        self.assertEqual(h["multiples_of_5_n"], 3)
        self.assertEqual(h["of_which_multiples_of_10_n"], 2)
        self.assertEqual(h["multiples_of_5_not_10_n"], 1)

    def test_partition_identity_holds(self):
        rng = np.random.default_rng(3)
        h = paperlib.heaping_counts(rng.integers(5, 46, 500))
        self.assertEqual(h["multiples_of_5_n"],
                         h["of_which_multiples_of_10_n"] + h["multiples_of_5_not_10_n"])

    def test_twelve_and_thirteen_are_not_counted_as_round(self):
        h = paperlib.heaping_counts([12, 13, 12, 13])
        self.assertEqual(h["multiples_of_5_n"], 0)

    def test_percentages_are_of_n(self):
        h = paperlib.heaping_counts([5, 5, 1, 1])
        self.assertEqual(h["multiples_of_5_pct"], 50.0)

    def test_missing_values_are_dropped_from_n(self):
        h = paperlib.heaping_counts([5, np.nan, 1])
        self.assertEqual(h["n_events"], 2)


class ReliabilityTests(unittest.TestCase):
    def test_alpha_matches_closed_form(self):
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        k = 4
        expected = k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
        self.assertAlmostEqual(paperlib.cronbach(X), float(expected), places=10)

    def test_identical_items_give_alpha_one(self):
        v = np.arange(50, dtype=float)
        X = pd.DataFrame({"a": v, "b": v, "c": v})
        self.assertAlmostEqual(paperlib.cronbach(X), 1.0, places=8)

    def test_alpha_rises_when_a_weak_item_is_dropped(self):
        rng = np.random.default_rng(5)
        lat = rng.normal(size=400)
        X = pd.DataFrame({"a": lat + rng.normal(0, .4, 400),
                          "b": lat + rng.normal(0, .4, 400),
                          "junk": rng.normal(0, 1, 400)})
        aid = paperlib.alpha_if_deleted(X)
        self.assertGreater(aid["junk"], paperlib.cronbach(X))
        self.assertLess(aid["a"], paperlib.cronbach(X))

    def test_single_item_scale_is_nan(self):
        self.assertTrue(np.isnan(paperlib.cronbach(pd.DataFrame({"a": [1.0, 2, 3]}))))

    def test_alpha_if_deleted_covers_every_item(self):
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(60, 3)),
                         columns=["x", "y", "z"])
        self.assertEqual(set(paperlib.alpha_if_deleted(X)), {"x", "y", "z"})


class IdentityMetricTests(unittest.TestCase):
    def test_identical_vectors(self):
        v = np.linspace(0, 1, 100)
        m = paperlib.identity_metrics(v, v)
        self.assertAlmostEqual(m["MAE"], 0.0, places=12)
        self.assertAlmostEqual(m["max_abs_difference"], 0.0, places=12)
        self.assertAlmostEqual(m["calibration_slope"], 1.0, places=8)
        self.assertAlmostEqual(m["calibration_intercept"], 0.0, places=8)

    def test_affine_rescaling_is_perfectly_correlated_but_not_identical(self):
        """This is exactly the failure mode r = 1.0000 could not rule out."""
        v = np.linspace(0.1, 0.4, 200)
        archived = 2.5 * v + 0.07
        m = paperlib.identity_metrics(v, archived)
        self.assertAlmostEqual(m["pearson_r"], 1.0, places=12)
        self.assertGreater(m["max_abs_difference"], 0.1)
        self.assertAlmostEqual(m["calibration_slope"], 2.5, places=8)
        self.assertAlmostEqual(m["calibration_intercept"], 0.07, places=8)
        self.assertLess(m["calibration_max_abs_residual"], 1e-8)

    def test_noise_shows_up_as_residual_not_just_slope(self):
        rng = np.random.default_rng(2)
        v = rng.normal(size=500)
        m = paperlib.identity_metrics(v, v + rng.normal(0, .05, 500))
        self.assertGreater(m["calibration_max_abs_residual"], 1e-3)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paperlib.identity_metrics(np.zeros(5), np.zeros(6))

    def test_nan_pairs_are_dropped(self):
        a = np.array([1.0, 2.0, np.nan, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(paperlib.identity_metrics(a, b)["n_compared"], 3)


class BootstrapTests(unittest.TestCase):
    def test_two_sided_p_is_one_when_centred_on_zero(self):
        d = np.concatenate([-np.ones(500), np.ones(500)])
        self.assertAlmostEqual(paperlib.two_sided_bootstrap_p(d), 1.0, places=6)

    def test_two_sided_p_is_small_when_all_one_sign(self):
        self.assertAlmostEqual(paperlib.two_sided_bootstrap_p(np.ones(1000)), 0.0,
                               places=6)

    def test_p_never_exceeds_one(self):
        self.assertLessEqual(paperlib.two_sided_bootstrap_p(np.zeros(100)), 1.0)

    def test_percentile_ci_brackets_the_centre(self):
        v = np.random.default_rng(4).normal(0, 1, 20000)
        lo, hi = paperlib.percentile_ci(v, 95)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertAlmostEqual(lo, -1.96, delta=0.08)

    def test_cluster_bootstrap_is_wider_than_ignoring_clusters(self):
        """A participant contributing many correlated events must not count as many
        independent observations; clustered resampling should widen the interval."""
        rng = np.random.default_rng(9)
        pid = np.repeat(np.arange(40), 10)
        person = rng.normal(0, 3, 40)
        vals = person[pid] + rng.normal(0, .2, 400)
        clustered = paperlib.cluster_bootstrap_median(vals, pid, n_boot=600, seed=1)
        naive = np.array([np.median(rng.choice(vals, len(vals), replace=True))
                          for _ in range(600)])
        c_lo, c_hi = paperlib.percentile_ci(clustered)
        n_lo, n_hi = paperlib.percentile_ci(naive)
        self.assertGreater(c_hi - c_lo, n_hi - n_lo)

    def test_cluster_bootstrap_is_reproducible_from_the_seed(self):
        pid = np.repeat(np.arange(20), 5)
        v = np.arange(100, dtype=float)
        a = paperlib.cluster_bootstrap_median(v, pid, n_boot=200, seed=42)
        b = paperlib.cluster_bootstrap_median(v, pid, n_boot=200, seed=42)
        self.assertTrue(np.array_equal(a, b))

    def test_cluster_bootstrap_centres_near_the_sample_median(self):
        pid = np.repeat(np.arange(50), 4)
        v = np.random.default_rng(6).normal(14, 3, 200)
        boots = paperlib.cluster_bootstrap_median(v, pid, n_boot=800, seed=2)
        self.assertAlmostEqual(float(np.median(boots)), float(np.median(v)), delta=0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
