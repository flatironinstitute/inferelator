import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sps
from inferelator.regression import mi
from inferelator.utils import InferelatorData

L = InferelatorData(expression_data=np.array([[1, 2], [3, 4]]), transpose_expression=True)
L_sparse = InferelatorData(expression_data=sps.csr_matrix([[1, 2], [3, 4]]), transpose_expression=True)
L2 = InferelatorData(expression_data=np.array([[3, 4], [2, 1]]), transpose_expression=True)


class Test2By2(unittest.TestCase):

    def setUp(self):
        self.x_dataframe = L.copy()
        self.y_dataframe = L.copy()

    def test_12_34_identical(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_minus(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.y_dataframe.multiply(-1)
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_times_pi(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.y_dataframe.multiply(np.pi)
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_swapped(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.y_dataframe = L2.copy()
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_transposed(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        self.y_dataframe = InferelatorData(expression_data=np.array([[1, 2], [3, 4]]))
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_and_zeros(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.y_dataframe = InferelatorData(expression_data=np.zeros((2, 2)))
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        # the entire clr matrix is NAN
        self.assertTrue(np.isnan(self.clr_matrix.values).all())

    def test_12_34_and_ones(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        self.y_dataframe = InferelatorData(expression_data=np.ones((2, 2)))
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        self.assertTrue(np.isnan(self.clr_matrix.values).all())


class Test2By2Sparse(Test2By2):

    def setUp(self):
        self.x_dataframe = L_sparse.copy()
        self.y_dataframe = L_sparse.copy()


class Test2By3(unittest.TestCase):

    def test_12_34_identical(self):
        """Compute mi for identical arrays [[1, 2, 1], [2, 4, 6]]."""
        M = InferelatorData(expression_data=np.array([[1, 2, 1], [3, 4, 6]]), transpose_expression=True)
        self.x_dataframe = M.copy()
        self.y_dataframe = M.copy()
        self.clr_matrix, self.mi_matrix = mi.context_likelihood_mi(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)
