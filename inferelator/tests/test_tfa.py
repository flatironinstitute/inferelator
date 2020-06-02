import unittest
from inferelator.preprocessing import tfa
from inferelator.utils import InferelatorData
import pandas as pd
import numpy as np

units_in_the_last_place_tolerance = 15


class TestTFA(unittest.TestCase):

    # Test for 5 genes, one of which is a TF, 5 condidtions, and 4 TFs.
    # where tau is equal to 1, so expression_matrix and expression_matrix_halftau are equivalent
    def setup_mouse_th17(self):
        exp = pd.DataFrame(np.array([[12.2844, 8.16, 10.4782, 5.46, 7.96367],
                                     [12.55, 8.5536, 11.0834, 5.4891, 7.86005],
                                     [11.8626, 7.765, 10.5227, 4.9039, 7.82641],
                                     [11.8623, 7.8903, 10.3418, 4.698, 7.94938],
                                     [11.881, 8.0871, 10.3878, 5.0788, 7.67066]]))
        exp.index = ['s1', 's2', 's3', 's4', 's5']
        exp.columns = ['g1', 't2', 'g3', 'g4', 'g5']

        self.exp = InferelatorData(exp)

        self.priors = pd.DataFrame(np.array([[1, 0, 0, 1],
                                             [0, 0, 0, 0],
                                             [0, 0, -1, 0],
                                             [-1, 0, 0, -1],
                                             [0, 0, 1, 0]]),
                                   columns=['t1', 't2', 't3', 't4'],
                                   index=['g1', 't2', 'g3', 'g4', 'g5'])

    def setup_three_columns(self):
        exp = pd.DataFrame(np.array([[1, 1, 0],
                                     [3, 2, 3]]),
                           index=['s1', 's2'],
                           columns=['g1', 'tf1', 'g3'])

        self.exp = InferelatorData(exp)

        self.priors = pd.DataFrame(np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]),
                                   columns=['tf1', 'tf2', 'tf3'],
                                   index=self.exp.gene_names)

    def setup_one_column(self):
        exp = pd.DataFrame(np.array([[1, 1, 0],
                                     [3, 2, 3]]),
                           index=['s1', 's2'],
                           columns=['g1', 'tf1', 'g3'])

        self.exp = InferelatorData(exp)

        self.priors = pd.DataFrame(np.array([[1], [1], [0]]),
                                   columns=['tf1'],
                                   index=self.exp.gene_names)

    def drop_prior(self):
        self.priors = self.priors.drop(self.priors.columns, axis=1)

    # TODO: should this raise an error?
    def test_priors_no_columns(self):
        self.setup_one_column()
        self.drop_prior()
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        # assert that there are no columns in the output activities matrix
        self.assertEqual(activities.shape[1], 0)

    def test_when_prior_is_zero_vector_activity_is_expression_one_column(self):
        self.setup_one_column()
        self.priors['tf1'] = [0, 0, 0]
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        np.testing.assert_equal(activities.expression_data.T, [[1, 2]])

    # add a duplicate TF column to the priors matrix
    # verifying that self interaction remains
    def test_duplicate_removal_keeps_self_interaction_two_column(self):
        self.setup_one_column()
        self.priors['g3'] = self.priors['tf1']
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp, keep_self=True)
        np.testing.assert_array_almost_equal_nulp(activities.expression_data.T,
                                                  np.array([[.5, 1.25], [.5, 1.25]]),
                                                  units_in_the_last_place_tolerance)

    # add a duplicate TF column to the priors matrix
    def test_duplicate_removal_does_not_happen_with_dupes_flag_false_two_column(self):
        self.setup_one_column()
        self.priors['g3'] = self.priors['tf1']
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        np.testing.assert_allclose(activities.expression_data.T,
                                   np.array([[0, 1], [1, 2]]),
                                   atol=1e-15)

    def test_tfa_default_one_column(self):
        self.setup_one_column()
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        np.testing.assert_array_almost_equal_nulp(activities.expression_data.T,
                                                  np.array([[1, 3]]),
                                                  units_in_the_last_place_tolerance)

    def test_tfa_default_all_zero_prior_no_expression_data(self):
        self.setup_one_column()
        self.priors['tf2'] = [0, 0, 0]
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        # Assert that the all-zero no-expression tf was dropped from the activity matrix
        np.testing.assert_array_almost_equal_nulp(activities.expression_data.T,
                                                  np.array([[1, 3]]),
                                                  units_in_the_last_place_tolerance)

    def test_tfa_default_three_columns(self):
        self.setup_three_columns()
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp, keep_self=True)
        np.testing.assert_allclose(activities.expression_data.T,
                                   np.array([[.5, 1], [.5, 1], [0, 1]]),
                                   atol=1e-15)

    def test_tfa_default_three_columns_dup_self_false(self):
        self.setup_three_columns()
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        np.testing.assert_allclose(activities.expression_data.T,
                                   np.array([[0, 0.5], [1, 2], [0, 0.5]]),
                                   atol=1e-15)

    def test_tfa_default_using_mouse_th17(self):
        self.setup_mouse_th17()
        activities = tfa.TFA().compute_transcription_factor_activity(self.priors, self.exp)
        np.testing.assert_allclose(activities.expression_data.T,
                                   np.array([[1.706100, 1.765225, 1.739675, 1.791075, 1.70055],
                                             [8.160000, 8.553600, 7.765000, 7.890300, 8.08710],
                                             [-1.257265, -1.611675, -1.348145, -1.196210, -1.35857],
                                             [1.706100, 1.765225, 1.739675, 1.791075, 1.70055]]),
                                   atol=1e-15)