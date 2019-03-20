import unittest
from inferelator_ng.regression import bayes_stats
import pandas as pd
import numpy as np
import os


class TestBayesStats(unittest.TestCase):

    def test_bbsr(self):
        # test when pp.sum() != 0
        X = np.array([[1, 0, 1], [2, 1, 0], [1, 1, 1], [0, 0, 1], [2, 1, 2]])
        y = np.array([0, 1, 0])
        pp = np.array([0, 1, 2, 1, 0])
        weights = np.array([1, 0, 2, 1, 5])
        max_k = 10
        result = bayes_stats.bbsr(X, y, pp, weights, max_k)


    def test_reduce_predictors(self):
        # test for k = max_k
        x = np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3], [1, 1, 1]])
        y = np.array([1, 1, 0, 1])
        gprior = np.array([3, 2, 1])
        max_k = 3
        result = bayes_stats.reduce_predictors(x, y, gprior, max_k)
        np.testing.assert_array_equal(result, np.array([True, True, True]))

    #def test_reduce_predictors_max_k_greater_than(self):
        # test for k > max_k
     #   x = np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3], [1, 1, 1]])
      #  y = np.array([1, 1, 0, 1])
       # gprior = np.array([3, 2, 1])
        #max_k = 2
        #result = bayes_stats.reduce_predictors(x, y, gprior, max_k)
        #np.testing.assert_array_equal(result, np.array([True, True, True]))
   # def test_best_subset_regression(self):
    #



   # def test_calc_rate(self):
    #

    def test_best_combo_idx(self):
        x = np.array([[0, 1, 2, 3], [0, 0, 1, 1], [1, 1, 1, 1]])
        bic = np.array([1, 0, 1, 0], dtype=np.dtype(float))
        combo = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 2, 3]])
        result = bayes_stats._best_combo_idx(x, bic, combo)
        np.testing.assert_array_equal(result, 3)

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

