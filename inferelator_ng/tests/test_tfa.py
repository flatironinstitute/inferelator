import unittest
from .. import tfa
import pandas as pd
import numpy as np
import subprocess

units_in_the_last_place_tolerance = 9 
class TestTFA(unittest.TestCase):

    def generate_random_matrix(self, n, m):
        return np.array([np.random.rand(m) for x in range(n)])

    # Test for 5 genes, one of which is a TF, 5 condidtions, and 4 TFs.
    # where tau is equal to 1, so exp_mat and exp_mat_tau are equivalent
    def setup_max(self):
        exp = pd.DataFrame(self.generate_random_matrix(5, 5))
        exp.columns = ['s1', 's2', 's3', 's4', 's5']
        exp.index = ['g1', 't2', 'g3', 'g4', 'g5']
        priors = pd.DataFrame(np.array([[1,0,0,1], [0,0,0,0], [0,0,-1,0], [-1,0,0,-1], [0,0,1,0]]))
        priors.columns = ['t1', 't2', 't3', 't4']
        priors.index = ['g1', 't2', 'g3', 'g4', 'g5']
        self.tfa_python = tfa.TFA(priors, exp, exp)

    def setup(self):
        tau = 2
        exp = pd.DataFrame(np.array([[1, 3], [1, 2], [0, 3]]))
        exp.columns = ['s1', 's2']
        exp.index = ['g1', 'tf1', 'g3']
        priors = pd.DataFrame(np.array([[1], [1], [0]]))
        priors.columns = ['tf1']
        priors.index = exp.index
        self.tfa_python = tfa.TFA(priors, exp, exp/tau)

    def drop_prior(self):
        for i in self.tfa_python.prior.columns:
            self.tfa_python.prior = self.tfa_python.prior.drop(i, 1) 

    # Test what happens when there are no relevant columns in the prior matrix
    def test_priors_no_columns(self):
        self.setup()
        self.drop_prior()
        activities = self.tfa_python.tfa()
        # assert that there are no rows in the output activities matrix
        self.assertEqual(activities.shape[0], 0)

    def test_priors_is_zero_vector(self):
        self.setup()
        self.tfa_python.prior['tf1'] = [0, 0, 0]
        activities = self.tfa_python.tfa()
        np.testing.assert_equal(activities.values, [[1,2]])
        np.testing.assert_equal(self.tfa_python.prior.values, [[0], [0], [0]])

    # add a duplicate TF column to the priors matrix
    def test_duplicate_removal(self):
        self.setup()
        self.tfa_python.prior['g3'] = self.tfa_python.prior['tf1']
        activities = self.tfa_python.tfa()
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ 0.25,   0.625], [ 0.25,   0.625]]),
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_python.prior.values, np.array([[1, 1], [0, 1], [0, 0]]))

    # add a duplicate TF column to the priors matrix
    def test_duplicate_removal_does_not_happen_with_dupes_flag_false(self):
        self.setup()
        self.tfa_python.prior['g3'] = self.tfa_python.prior['tf1']
        activities = self.tfa_python.tfa(dup_self = False)
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ 0.25,   0.625], [ 0.25,   0.625]]),
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_python.prior.values, np.array([[1, 1], [0, 1], [0, 0]]))


    def test_tfa_default(self):
        self.setup()
        activities = self.tfa_python.tfa()
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ 0.5,   1.25]]),
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_python.prior.values, np.array([[1], [0], [0]]))
