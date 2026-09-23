"""可手算的数学结果、异常输入和示例数据回归测试。"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from src.weight import combined_weights, critic_weights, entropy_weights, normalize, score


class WeightTests(unittest.TestCase):
    def setUp(self):
        # a 与 b 零相关，a 与 c 完全负相关，b 与 c 零相关。
        self.X = pd.DataFrame({"a": [0., 1., 0., 1.],
                               "b": [0., 0., 1., 1.],
                               "c": [1., 0., 1., 0.]})

    def test_normalize_directions_negative_values_and_constant(self):
        raw = pd.DataFrame({"benefit": [-10, -5, 0], "cost": [-10, -5, 0],
                            "constant": [7, 7, 7]}, index=["A", "B", "C"])
        X = normalize(raw, ["benefit", "constant"], ["cost"])
        assert_allclose(X["benefit"], [0, .5, 1])
        assert_allclose(X["cost"], [1, .5, 0])
        assert_allclose(X["constant"], 0)
        self.assertEqual(list(X.index), ["A", "B", "C"])

    def test_entropy_hand_calculation_and_zero_probability(self):
        X = pd.DataFrame({"one": [0., 0., 0., 1.], "two": [0., 0., 1., 1.]})
        w, e = entropy_weights(X)
        assert_allclose(e, [0, .5], atol=1e-15)
        assert_allclose(w, [2/3, 1/3])

    def test_entropy_constants_never_receive_weight(self):
        X = self.X.assign(zero=0., nonzero=1.)
        w, e = entropy_weights(X)
        assert_allclose(w[["a", "b", "c"]], [1/3] * 3)
        assert_allclose(w[["zero", "nonzero"]], [0, 0])
        assert_allclose(e[["zero", "nonzero"]], [1, 1])

    def test_critic_hand_calculation_preserves_negative_correlation(self):
        w, info = critic_weights(self.X)
        assert_allclose(w, [3/8, 2/8, 3/8])
        assert_allclose(info, np.sqrt(1/3) * np.array([3, 2, 3]))

    def test_constant_column_does_not_change_critic_information(self):
        w, info = critic_weights(self.X)
        extended_w, extended_info = critic_weights(self.X.assign(constant=.7))
        assert_allclose(extended_w[self.X.columns], w)
        assert_allclose(extended_info[self.X.columns], info)
        self.assertEqual(extended_w["constant"], 0)
        self.assertEqual(extended_info["constant"], 0)

    def test_all_constant_single_indicator_and_no_conflict_are_explicit(self):
        constant = pd.DataFrame({"a": [0., 0., 0.], "b": [1., 1., 1.]})
        for method in [entropy_weights, critic_weights]:
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method(constant)
        for X in [self.X[["a"]], self.X[["a"]].assign(b=self.X["a"], zero=0.)]:
            with self.subTest(columns=list(X.columns)), self.assertRaises(ValueError):
                critic_weights(X)
        assert_allclose(entropy_weights(self.X[["a"]])[0], [1.])

    def test_single_row_empty_and_duplicate_columns_are_rejected(self):
        cases = [self.X.iloc[:1], self.X.iloc[:0], pd.DataFrame(index=[0, 1]),
                 pd.DataFrame([[0., 1.], [1., 0.]], columns=["a", "a"])]
        for X in cases:
            for method in [entropy_weights, critic_weights]:
                with self.subTest(shape=X.shape, method=method.__name__), self.assertRaises(ValueError):
                    method(X)
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                normalize(X, list(X.columns), [])

    def test_missing_infinite_non_numeric_bool_and_complex_are_rejected(self):
        for bad in [np.nan, np.inf, -np.inf, "not-a-number", True, 1j]:
            X = self.X.copy()
            X["a"] = [bad, 0., 1., 0.]
            for method in [entropy_weights, critic_weights]:
                with self.subTest(bad=bad, method=method.__name__), self.assertRaises(ValueError):
                    method(X)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize(X, list(X.columns), [])

    def test_out_of_range_direct_input_is_rejected(self):
        for bad in [-.1, 1.1]:
            X = self.X.copy()
            X.loc[0, "a"] = bad
            for method in [entropy_weights, critic_weights]:
                with self.subTest(bad=bad, method=method.__name__), self.assertRaises(ValueError):
                    method(X)

    def test_direction_coverage_must_be_exact(self):
        for benefit, cost in [(["a"], ["b"]), (["a", "b"], ["a", "c"]),
                              (["a", "a", "b"], ["c"]),
                              (["a", "b", "c", "unknown"], [])]:
            with self.subTest(benefit=benefit, cost=cost), self.assertRaises(ValueError):
                normalize(self.X, benefit, cost)

    def test_geometric_mean_linear_mix_and_label_alignment(self):
        ent = pd.Series([.8, .2], index=["a", "b"])
        cri = pd.Series([.4, .6], index=["b", "a"])
        geom, linear = combined_weights(ent, cri, alpha=.75)
        expected_a = np.sqrt(6) / (np.sqrt(6) + 1)
        assert_allclose(geom, [expected_a, 1 - expected_a])
        assert_allclose(linear, [.75, .25])
        assert_allclose(combined_weights(ent, cri, 1)[1], ent)
        assert_allclose(combined_weights(ent, cri, 0)[1], [.6, .4])

    def test_invalid_alpha_disjoint_support_and_mismatched_labels(self):
        w = pd.Series([.5, .5], index=["a", "b"])
        for alpha in [-.1, 1.1, np.nan, np.inf, "0.5", None, [.5], True, np.complex64(.5)]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                combined_weights(w, w, alpha)
        with self.assertRaises(ValueError):
            combined_weights(pd.Series([1., 0.], index=w.index),
                             pd.Series([0., 1.], index=w.index))
        with self.assertRaises(ValueError):
            combined_weights(w, pd.Series([.5, .5], index=["a", "c"]))

    def test_invalid_weights_and_incomplete_score_are_rejected(self):
        good = pd.Series([.5, .5], index=["a", "b"])
        cases = [pd.Series([.5, .5], index=["a", "a"]),
                 pd.Series([-.1, 1.1], index=good.index),
                 pd.Series([.2, .2], index=good.index),
                 pd.Series([np.nan, .5], index=good.index),
                 pd.Series([np.inf, .5], index=good.index)]
        for w in cases:
            with self.subTest(weights=w.to_list()), self.assertRaises(ValueError):
                combined_weights(w, good)
            with self.assertRaises(ValueError):
                score(self.X[["a", "b"]], w)
        with self.assertRaises(ValueError):
            score(self.X, good)

    def test_score_aligns_by_name_and_accepts_single_row(self):
        X = pd.DataFrame({"a": [.2], "b": [.8]}, index=["sample"])
        w = pd.Series([.75, .25], index=["b", "a"])
        result = score(X, w)
        assert_allclose(result, [.65])
        self.assertEqual(result.index.to_list(), ["sample"])

    def test_teaching_csv_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        raw = pd.read_csv(root / "examples" / "sample_data.csv").set_index("community")
        X = normalize(raw, ["bed_density", "staff_ratio", "coverage", "satisfaction"],
                      ["wait_days", "self_pay"])
        we, _ = entropy_weights(X)
        wc, _ = critic_weights(X)
        wg, wl = combined_weights(we, wc)
        for w in [we, wc, wg, wl]:
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue((w >= 0).all())
            self.assertAlmostEqual(w.sum(), 1)
            result = score(X, w)
            self.assertTrue(result.between(0, 1).all())
            # C 在每项教学指标都不劣于其他对象，D 每项都不优于其他对象。
            self.assertEqual(result.idxmax(), "C")
            self.assertEqual(result.idxmin(), "D")


if __name__ == "__main__":
    unittest.main()
