import unittest
import pandas as pd
import numpy as np
from inferelator.regression import mi


class Test2By2(unittest.TestCase):

    def test_12_34_identical(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_minus(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(-np.array(L))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_times_pi(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.pi * np.array(L))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_swapped(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        L2 = [[3, 4], [2, 1]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L2))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_transposed(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L).transpose())
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_12_34_and_zeros(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.zeros((2, 2)))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        # the entire clr matrix is NAN
        self.assertTrue(np.isnan(self.clr_matrix.values).all())

    def test_12_34_and_ones(self):
        """Compute mi for identical arrays [[1, 2], [2, 4]]."""
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.ones((2, 2)))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        self.assertTrue(np.isnan(self.clr_matrix.values).all())


class Test2By3(unittest.TestCase):

    def test_12_34_identical(self):
        """Compute mi for identical arrays [[1, 2, 1], [2, 4, 6]]."""
        L = [[1, 2, 1], [3, 4, 6]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.clr_matrix.values, expected)

    def test_mixed(self):
        """Compute mi for mixed arrays."""
        L = [[1, 2, 1], [3, 4, 6]]
        L2 = [[3, 7, 1], [9, 0, 2]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L2))
        self.clr_matrix, self.mi_matrix = mi.MIDriver().run(self.x_dataframe, self.y_dataframe)
        expected = np.array([[0, 1], [1, 0]])
        # np.testing.assert_almost_equal(self.clr_matrix.values, expected)
