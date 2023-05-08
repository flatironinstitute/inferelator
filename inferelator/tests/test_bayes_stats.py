import unittest
import warnings

from inferelator.regression import bayes_stats
import numpy as np
import scipy.stats


PREDICTORS = np.array([[9, -10, 1, 9, -10, 5, -6, 5, 4, 6],
                       [5, 9, 5, -1, -8, -6, -2, 6, 7, 9],
                       [-10, -4, -3, -2, 0, 1, 5, 6, 8, 9]]).T

PREDICTORS_Z = scipy.stats.zscore(PREDICTORS)


class TestBayesStats(unittest.TestCase):

    def test_bbsr_one_good_predictor(self):

        x = PREDICTORS_Z.copy()
        y = scipy.stats.zscore(PREDICTORS[:, 2].copy().flatten())

        pp = np.array([1, 1, 1])
        weights = np.array([10, 10, 10])
        max_k = 3

        result = bayes_stats.bbsr(x, y, pp, weights, max_k)

        pp = np.array([1, 1, 1])
        betas = np.array([0.0, 0.0, 1.0])
        betas_resc = np.array([0.0, 0.0, 1.0])

        np.testing.assert_array_almost_equal(result['pp'], pp)
        np.testing.assert_array_almost_equal(result['betas'], betas)
        np.testing.assert_array_almost_equal(result['betas_resc'], betas_resc)

        result_2 = bayes_stats.bbsr(x, y, pp, weights, max_k, ordinary_least_squares=True)

        np.testing.assert_array_almost_equal(result['pp'], result_2['pp'])
        np.testing.assert_array_almost_equal(result['betas'], result_2['betas'])
        np.testing.assert_array_almost_equal(result['betas_resc'], result_2['betas_resc'])

    def test_bbsr_two_good_predictors(self):

        x = PREDICTORS_Z.copy()
        y = scipy.stats.zscore(np.sum(PREDICTORS[:, [False, True, True]], axis=1).flatten())

        pp = np.array([1, 1, 1])
        weights = np.array([10, 10, 10])
        max_k = 3

        result = bayes_stats.bbsr(x, y, pp, weights, max_k)

        pp = np.array([1, 1, 1])
        betas = np.array([0.0, 0.682945, 0.664451])
        betas_resc = np.array([0.0, 1.0, 1.0])

        np.testing.assert_array_almost_equal(result['pp'], pp)
        np.testing.assert_array_almost_equal(result['betas'], betas)
        np.testing.assert_array_almost_equal(result['betas_resc'], betas_resc)

        result_2 = bayes_stats.bbsr(x, y, pp, weights, max_k, ordinary_least_squares=True)

        np.testing.assert_array_almost_equal(result['pp'], result_2['pp'])
        np.testing.assert_array_almost_equal(result['betas'], result_2['betas'])
        np.testing.assert_array_almost_equal(result['betas_resc'], result_2['betas_resc'])

    def test_bbsr_three_good_predictors(self):

        x = PREDICTORS_Z.copy()
        y = scipy.stats.zscore(np.sum(PREDICTORS, axis=1).flatten())

        pp = np.array([1, 1, 2])
        weights = np.array([10, 10, 10])
        max_k = 3

        result = bayes_stats.bbsr(x, y, pp, weights, max_k)

        pp = np.array([1, 1, 1])
        betas = np.array([0.583039, 0.491769, 0.478452])
        betas_resc = np.array([1.0, 1.0, 1.0])

        np.testing.assert_array_almost_equal(result['pp'], pp)
        np.testing.assert_array_almost_equal(result['betas'], betas)
        np.testing.assert_array_almost_equal(result['betas_resc'], betas_resc)

        result_2 = bayes_stats.bbsr(x, y, pp, weights, max_k, ordinary_least_squares=True)

        np.testing.assert_array_almost_equal(result['pp'], result_2['pp'])
        np.testing.assert_array_almost_equal(result['betas'], result_2['betas'])
        np.testing.assert_array_almost_equal(result['betas_resc'], result_2['betas_resc'])

    def test_bbsr_2(self):
        # test when pp.sum() == 0
        X = np.array([[1, 0, 0], [2, 1, 0], [1, 1, 1], [0, 0, 1], [2, 1, 2]]).T
        y = np.array([0, 1, 0])
        pp = np.array([0, 0, 0, 0, 0])
        weights = np.array([1, 0, 2, 1, 5])
        max_k = 10
        result = bayes_stats.bbsr(X, y, pp, weights, max_k)
        pp = np.array([True, True, True, True, True])
        betas = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        betas_resc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        dict = {'pp':pp, 'betas':betas, 'betas_resc':betas_resc}
        np.testing.assert_equal(result, dict)

    def test_bbsr_3(self):
        # test when betas and betas_resc are not zero
        X = np.array([[1, 3, 1], [2, 1, 0], [1, 10, 5], [2, 6, 1], [2, 1, 8]]).T
        y = np.array([2, 1, 4])
        pp = np.array([10, 3, 1, 5, 4])
        weights = np.array([10, 10, 10, 10, 10])
        max_k = 3
        result = bayes_stats.bbsr(X, y, pp, weights, max_k)
        pp = np.array([1, 0, 0, 1, 1])
        betas = ([0.0, 0.0, 0.53623188])
        betas_resc = ([0.0, 0.0, 0.83820926])
        check = {'pp':pp, 'betas':betas, 'betas_resc':betas_resc}
        for component in check.keys():
            for idx in range(0, len(check[component])):
                np.testing.assert_array_almost_equal(result[component][idx], check[component][idx], 2)

    def test_best_subset_regression(self):
        x = np.array([[1, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
        y = np.array([1, 0, 2, 3, 1])
        gprior = np.array([[0, 1, 2, 3]])
        result = bayes_stats.best_subset_regression(x, y, gprior)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0, 0.0]))

    def test_best_subset_regression_lin_alg_error(self):
        x = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        y = np.array([0, 0, 0, 0, 0])
        gprior = np.array([[0, 0, 0, 0]])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = bayes_stats.best_subset_regression(x, y, gprior)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.dtype(float)))

    def test_reduce_predictors(self):
        # test for k = max_k
        x = np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3], [1, 1, 1]])
        y = np.array([1, 1, 0, 1])
        gprior = np.array([[3, 2, 1]])
        max_k = 3
        result = bayes_stats.reduce_predictors(x, y, gprior, max_k)
        np.testing.assert_array_equal(result, np.array([True, True, True]))

    def test_reduce_predictors_max_k_greater_than(self):
        # test for k > max_k
        x = np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3], [1, 1, 1]])
        y = np.array([1, 1, 0, 1])
        gprior = np.array([[3, 2, 1]])
        max_k = 2
        result = bayes_stats.reduce_predictors(x, y, gprior, max_k)
        np.testing.assert_array_equal(result, np.array([True, True, False]))

    def test_calc_all_expected_BIC(self):
        x = np.array([[1, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
        y = np.array([1, 0, 2, 3, 1])
        g = np.array([[0, 1, 2, 3]])
        combinations = np.array([[True, False, True, True], [True, False, False, True], [True, False, True, False],
                                 [True, True, True, False]])
        result = bayes_stats.calc_all_expected_BIC(x, y, g, combinations)
        np.testing.assert_array_almost_equal(result, np.array([12.9965, 8.1682, 11.387, 9.7776]), 4)

    def test_calc_rate(self):
        x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        y = np.array([[1, 2, 3], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
        xtx = np.dot(x.T, x)  # [k x k]
        xty = np.dot(x.T, y)  # [k x 1]
        gprior = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1]])

        model_beta = bayes_stats._solve_model(xtx, xty)
        model_ssr = bayes_stats.ssr(x, y, model_beta)
        scale_param = bayes_stats._calc_ig_scale(model_beta, model_ssr, xtx, gprior)

        np.testing.assert_array_equal(scale_param, np.array([[1.5, 1.5, 2.5], [1.5, 2.5, 3.5],
                                                        [2.5, 3.5, 5.5]]))

    def test_calc_rate_lin_alg_error(self):
        x = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12], [8, 16, 24]])
        y = np.array([[1, 2, 3], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
        xtx = np.dot(x.T, x)  # [k x k]
        xty = np.dot(x.T, y)  # [k x 1]
        gprior = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1]])
        with self.assertRaises(np.linalg.LinAlgError):
            model_beta = bayes_stats._solve_model(xtx, xty)
            model_ssr = bayes_stats.ssr(x, y, model_beta)
            scale_param = bayes_stats._calc_ig_scale(model_beta, model_ssr, xtx, gprior)

    def test_best_combo_idx(self):
        x = np.array([[0, 1, 2, 3], [0, 0, 1, 1], [1, 1, 1, 1]])
        bic = np.array([1, 0, 1, 0], dtype=np.dtype(float))
        combo = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 2, 3]])
        result = bayes_stats._best_combo_idx(x, bic, combo)
        np.testing.assert_array_equal(result, 3)

    def test_best_combo_idx_lin_alg_error(self):
        x = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 8, 12, 16]])
        bic = np.array([0, 0, 0, 0], dtype=np.dtype(float))
        combo = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 2, 3]])
        with self.assertRaises(np.linalg.LinAlgError):
            bayes_stats._best_combo_idx(x, bic, combo)

    def test_matrix_full_rank(self):
        mat = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 1]])
        result = bayes_stats._matrix_full_rank(mat)
        np.testing.assert_array_equal(result, True)

    def test_ssr(self):
        # if x is a N x M array, then beta must be a M x P array, then y must be a N x P array
        x = np.array([[1, 0, 4, 3, 2], [1, 1, 2, 2, 3]])
        y = np.array([[1, 1], [1, 2]])
        beta = np.array([[1, 2], [2, 3], [1, 1], [1, 2], [0, 1]])
        result = bayes_stats.ssr(x, y, beta)
        np.testing.assert_array_equal(result, 398)

    def test_ssr_zeros(self):
        x = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        y = np.array([[0, 0], [0, 0]])
        beta = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        result = bayes_stats.ssr(x, y, beta)
        np.testing.assert_array_equal(result, 0)

    def test_ssr_negative(self):
        x = np.array([[0, 1, -1, -2, 1], [-1, 2, 0, -1, 1]])
        y = np.array([[2, -1], [1, 2]])
        beta = np.array([[-1, -1], [1, 2], [2, 1], [-2, 1], [1, -1]])
        result = bayes_stats.ssr(x, y, beta)
        np.testing.assert_array_equal(result, 31)

    def test_combo_index(self):
        n = 3
        result = bayes_stats.combo_index(n)
        np.testing.assert_array_equal(result, np.array([[False, False, False, False,  True,  True,  True,  True],
       [False, False,  True,  True, False, False,  True,  True],
       [False,  True, False,  True, False,  True, False,  True]]))

    def test_select_index(self):
        n = 4
        result = bayes_stats.select_index(n)
        np.testing.assert_array_equal(result, np.array([[True,  True,  True, False, False, False],
       [True, False, False,  True,  True, False],
       [False,  True, False,  True, False,  True],
       [False, False,  True, False,  True,  True]]))

