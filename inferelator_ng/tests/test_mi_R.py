import unittest, os
import pandas as pd
import numpy as np
from .. import mi_R

my_dir = os.path.dirname(__file__)

class TestMI(unittest.TestCase):
    """
    Superclass for common methods
    """
    cores = bins = 10

    def calculate_mi(self):
        driver = mi_R.MIDriver()
        target = driver.target_directory = os.path.join(my_dir, "artifacts")
        if not os.path.exists(target):
            os.makedirs(target)
        driver.cores = self.cores
        driver.bins = self.bins
        self.mi_matrix = driver.run(self.x_dataframe, self.y_dataframe)

class Test2By2(TestMI):

    def test_12_34_identical(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L))
        self.calculate_mi()
        #print "MI ", L, L
        #print self.mi_matrix
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.mi_matrix.as_matrix(), expected)

    def test_12_34_minus(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(-np.array(L))
        self.calculate_mi()
        #print "MI ", self.x_dataframe, self.y_dataframe
        #print self.mi_matrix
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.mi_matrix.as_matrix(), expected)

    def test_12_34_times_pi(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.pi * np.array(L))
        self.calculate_mi()
        print "MI ", self.x_dataframe, self.y_dataframe
        print self.mi_matrix
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.mi_matrix.as_matrix(), expected)

    def test_12_34_swapped(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        L2 = [[3, 4], [2, 1]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L2))
        self.calculate_mi()
        #print "MI ", L, L2
        #print self.mi_matrix
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.mi_matrix.as_matrix(), expected)

    def test_12_34_transposed(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.array(L).transpose())
        self.calculate_mi()
        #print "MI ", L, self.y_dataframe
        #print self.mi_matrix
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_almost_equal(self.mi_matrix.as_matrix(), expected)

    def test_12_34_and_zeros(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.zeros((2,2)))
        self.calculate_mi()
        #print "MI ", L, np.array(L).transpose()
        #print self.mi_matrix
        # resulting matrix has only NANs everywhere
        self.assertTrue(np.isnan(self.mi_matrix.as_matrix()).all())

    def test_12_34_and_ones(self):
        "Compute mi for identical arrays [[1, 2], [2, 4]]."
        L = [[1, 2], [3, 4]]
        self.x_dataframe = pd.DataFrame(np.array(L))
        self.y_dataframe = pd.DataFrame(np.ones((2,2)))
        self.calculate_mi()
        #print "MI ", L, np.array(L).transpose()
        #print self.mi_matrix
        # resulting matrix has only NANs everywhere
        self.assertTrue(np.isnan(self.mi_matrix.as_matrix()).all())
