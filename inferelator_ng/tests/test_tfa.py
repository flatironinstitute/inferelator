import unittest
from .. import tfa
import pandas as pd
import numpy as np
import subprocess

units_in_the_last_place_tolerance = 15
class TestTFA(unittest.TestCase):

    def generate_random_matrix(self, n, m):
        return np.array([np.random.rand(m) for x in range(n)])

    # Test for 5 genes, one of which is a TF, 5 condidtions, and 4 TFs.
    # where tau is equal to 1, so expression_matrix and expression_matrix_halftau are equivalent
    def setup_mouse_th17(self):
        tau = 1
        exp = pd.DataFrame(np.array([[12.28440, 12.55000, 11.86260, 11.86230, 11.88100], 
            [8.16000, 8.55360, 7.76500, 7.89030, 8.08710],
            [10.47820, 11.08340, 10.52270, 10.34180, 10.38780],
            [5.46000,5.48910, 4.90390, 4.69800, 5.07880],
            [7.96367, 7.86005, 7.82641, 7.94938, 7.67066]]))
        exp.columns = ['s1', 's2', 's3', 's4', 's5']
        exp.index = ['g1', 't2', 'g3', 'g4', 'g5']
        priors = pd.DataFrame(np.array([[1,0,0,1], 
            [0,0,0,0], 
            [0,0,-1,0], 
            [-1,0,0,-1], 
            [0,0,1,0]]))
        priors.columns = ['t1', 't2', 't3', 't4']
        priors.index = ['g1', 't2', 'g3', 'g4', 'g5']
        self.tfa_object = tfa.TFA(priors, exp, exp/1)

    def setup_three_columns(self):
        tau = 1
        exp = pd.DataFrame(np.array([[1, 3], [1, 2], [0, 3]]))
        exp.columns = ['s1', 's2']
        exp.index = ['g1', 'tf1', 'g3']
        priors = pd.DataFrame(np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]))
        priors.columns = ['tf1', 'tf2', 'tf3']
        priors.index = exp.index
        self.tfa_object = tfa.TFA(priors, exp, exp/tau)

    def setup_one_column(self):
        tau = 1
        exp = pd.DataFrame(np.array([[1, 3], [1, 2], [0, 3]]))
        exp.columns = ['s1', 's2']
        exp.index = ['g1', 'tf1', 'g3']
        priors = pd.DataFrame(np.array([[1], [1], [0]]))
        priors.columns = ['tf1']
        priors.index = exp.index
        self.tfa_object = tfa.TFA(priors, exp, exp/tau)

    def drop_prior(self):
        for i in self.tfa_object.prior.columns:
            self.tfa_object.prior = self.tfa_object.prior.drop(i, 1) 

    # Test what happens when there are no relevant columns in the prior matrix
    # TODO: should this raise an error?
    def test_priors_no_columns(self):
        self.setup_one_column()
        self.drop_prior()
        activities = self.tfa_object.compute_transcription_factor_activity()
        # assert that there are no rows in the output activities matrix
        self.assertEqual(activities.shape[0], 0)

    def test_when_prior_is_zero_vector_activity_is_expression_one_column(self):
        self.setup_one_column()
        self.tfa_object.prior['tf1'] = [0, 0, 0]
        activities = self.tfa_object.compute_transcription_factor_activity()
        np.testing.assert_equal(activities.values, [[1,2]])
        np.testing.assert_equal(self.tfa_object.prior.values, [[0], [0], [0]])

    # add a duplicate TF column to the priors matrix
    # verifying that self interaction remains
    def test_duplicate_removal_keeps_self_interaction_two_column(self):
        self.setup_one_column()
        self.tfa_object.prior['g3'] = self.tfa_object.prior['tf1']
        activities = self.tfa_object.compute_transcription_factor_activity(
            allow_self_interactions_for_duplicate_prior_columns = True)
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ .5,   1.25], [ .5,   1.25]]),
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1, 1], [1, 1], [0, 0]]))

    # add a duplicate TF column to the priors matrix
    def test_duplicate_removal_does_not_happen_with_dupes_flag_false_two_column(self):
        self.setup_one_column()
        self.tfa_object.prior['g3'] = self.tfa_object.prior['tf1']
        activities = self.tfa_object.compute_transcription_factor_activity(
            allow_self_interactions_for_duplicate_prior_columns = False)
        print(activities.values)
        np.testing.assert_allclose(activities.values,
            np.array([[ 0,   1], [ 1,   2]]),
            atol=1e-15)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1, 1], [0, 1], [0, 0]]))

    def test_tfa_default_one_column(self):
        self.setup_one_column()
        activities = self.tfa_object.compute_transcription_factor_activity()
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ 1,   3]]),
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1], [0], [0]]))

    def test_tfa_default_all_zero_prior_no_expression_data(self):
        self.setup_one_column()
        self.tfa_object.prior['tf2'] = [0, 0, 0] 
        activities = self.tfa_object.compute_transcription_factor_activity()
        # Assert that the all-zero no-expression tf was dropped from the activity matrix
        np.testing.assert_array_almost_equal_nulp(activities.values,
            np.array([[ 1,   3]]), 
            units_in_the_last_place_tolerance)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1], [0], [0]]))

    def test_tfa_default_three_columns(self):
        self.setup_three_columns()
        activities = self.tfa_object.compute_transcription_factor_activity()
        np.testing.assert_allclose(activities.values,
            np.array([[ .5, 1], [.5, 1], [0, 1  ]]),
            atol=1e-15)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]))

    def test_tfa_default_three_columns_dup_self_false(self):
        self.setup_three_columns()
        activities = self.tfa_object.compute_transcription_factor_activity(
            allow_self_interactions_for_duplicate_prior_columns = False)
        np.testing.assert_allclose(activities.values,
            np.array([[ 0, 0.5], [1, 2], [0, 0.5]]),
            atol=1e-15)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]))

    def test_tfa_default_using_mouse_th17(self):
        self.setup_mouse_th17()
        activities = self.tfa_object.compute_transcription_factor_activity()
        np.testing.assert_allclose(activities.values,
            np.array([[1.706100, 1.765225, 1.739675, 1.791075, 1.70055], 
                [8.160000, 8.553600, 7.765000, 7.890300, 8.08710], 
                [-1.257265, -1.611675, -1.348145, -1.196210, -1.35857],
                [1.706100, 1.765225, 1.739675, 1.791075, 1.70055]]),
            atol=1e-15)
        # Assert the final priors matrix has no self- interactions
        np.testing.assert_equal(self.tfa_object.prior.values, np.array([[1,0,0,1], 
            [0,0,0,0], 
            [0,0,-1,0], 
            [-1,0,0,-1], 
            [0,0,1,0]]))