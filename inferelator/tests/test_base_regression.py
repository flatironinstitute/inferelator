import unittest
from inferelator.regression import base_regression
import pandas as pd
import numpy as np
import os


class TestBaseRegression(unittest.TestCase):

    def test_recalculate_betas_from_selected(self):
        # testing rank(xtx) = shape(xtx)
        x = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        y = np.array([0, 1, 0, 1, 0])
        result = base_regression.recalculate_betas_from_selected(x, y, idx=None)
        np.testing.assert_array_almost_equal(result, np.array([-0.4, 0.4]))

    def test_recalculate_betas_from_selected_matrix_rank(self):
        # test that the matrix rank(A) = min(n,m)
        # dim(v) - rank(A) = null(A) = 0
        x = np.array([[2, 4, 6], [4, 8, 12]])
        y = np.array([1, 1])
        result = base_regression.recalculate_betas_from_selected(x, y, idx=None)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]), 2)

    def test_bool_to_index(self):
        arr = np.array([[0, 1], [1, 0]])
        result = base_regression.bool_to_index(arr)
        np.testing.assert_array_almost_equal(result, np.array([0, 1]))

    def test_predict_error_reduction(self):
        # len(pp_idx) != 1
        x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        y = np.array([0, 1, 0, 1, 0])
        betas = np.array([1, 1, 2])
        error_reduction = base_regression.predict_error_reduction(x, y, betas)
        np.testing.assert_array_almost_equal(error_reduction, np.array([-133.333, -133.333, -133.333]), 2)

