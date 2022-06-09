import unittest
import pandas as pd
import pandas.testing as pdt
import numpy as np
from inferelator.distributed.inferelator_mp import MPControl
from inferelator.regression import bbsr_python
from inferelator.regression import bayes_stats
from inferelator.regression import base_regression
from inferelator.utils import InferelatorData


class TestBBSRrunnerPython(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()

    def setUp(self):
        self.brd = bbsr_python.BBSR

    def run_bbsr(self):
        return self.brd(self.X, self.Y, self.clr, self.priors).run()

    def set_all_zero_priors(self):
        self.priors = pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])

    def set_all_zero_clr(self):
        self.clr = pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])

    def assert_matrix_is_square(self, size, matrix):
        self.assertEqual(matrix.shape, (size, size))

    def test_two_genes(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = InferelatorData(pd.DataFrame([0, 0], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([0, 0], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)

        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_nonzero(self):
        self.set_all_zero_priors()
        self.X = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']),
                                 transpose_expression=True)
        self.clr = pd.DataFrame([[.1, .1], [.1, .2]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_negative_influence(self):
        self.set_all_zero_priors()
        self.X = InferelatorData(pd.DataFrame([[1, 2], [2, 1]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[1, 2], [2, 1]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.clr = pd.DataFrame([[.1, .1], [.1, .2]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1], [-1, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_negative_influence(self):
        self.set_all_zero_priors()
        self.X = InferelatorData(pd.DataFrame([[0, 2], [2, 0]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.clr = pd.DataFrame([[.1, .1], [.1, .2]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1], [-1, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_zero_clr_two_conditions_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = InferelatorData(pd.DataFrame([[1, 2], [2, 1]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[1, 2], [2, 1]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_zero_clr_two_conditions_zero_gene1_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = InferelatorData(pd.DataFrame([[0, 2], [2, 0]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0], [0, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_positive_influence(self):
        self.set_all_zero_priors()
        self.X = InferelatorData(pd.DataFrame([[1, 2], [1, 2]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[1, 2], [1, 2]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.clr = pd.DataFrame([[.1, .1], [.1, .2]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))

    def test_Best_Subset_Regression_all_zero_predictors(self):
        self.X = np.array([[0, 0], [0, 0]])
        self.Y = np.array([1, 2])
        g = np.array([1, 1])
        betas = bayes_stats.best_subset_regression(self.X, self.Y, g)
        self.assertTrue((betas == [0., 0.]).all())

    def test_PredictErrorReduction_all_zero_predictors(self):
        self.X = np.array([[0, 0], [0, 0]])
        self.Y = np.array([1, 2])
        betas = np.array([0., 0.])
        result = base_regression.predict_error_reduction(self.X, self.Y, betas)
        self.assertTrue((result == [0., 0.]).all())

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_positive_influence(self):
        self.set_all_zero_priors()
        self.X = InferelatorData(pd.DataFrame([[0, 2], [0, 2]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.Y = InferelatorData(pd.DataFrame([[1, 2], [1, 2]], index=['gene1', 'gene2'], columns=['ss1', 'ss2']),
                                 transpose_expression=True)
        self.clr = pd.DataFrame([[.1, .1], [.1, .2]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        (betas, resc) = self.run_bbsr()
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                   columns=['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'],
                                                  columns=['gene1', 'gene2']).astype(float))
