import unittest, os
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
from subprocess import CalledProcessError
from .. import bbsr_python
from .. import utils

my_dir = os.path.dirname(__file__)

class TestBBSRrunnerPython(unittest.TestCase):

    def setUp(self):
        self.brd = bbsr_python.BBSR_runner()

    def set_all_zero_priors(self):
        self.priors =  pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])

    def set_all_zero_clr(self):
        self.clr = pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])

    def assert_matrix_is_square(self, size, matrix):
        self.assertEqual(matrix.shape, (size, size))

    def test_two_genes(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([0, 0], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([0, 0], index = ['gene1', 'gene2'], columns = ['ss'])

        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    '''
    def test_fails_with_one_gene(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.Y = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.assertRaises(CalledProcessError, self.brd.run, self.X, self.Y, self.clr, self.priors)
    '''
    def test_two_genes_nonzero(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    # BBSR fails when there's only one column in the design (or response) matrix
    # That seems like unexpected behavior to me. If it is expected, there should be checks for it earlier -Nick DV
    '''
    @unittest.skip("""
        There's some unexpected behavior in bayesianRegression.R: a 2 x 1 matrix is getting transformed into a NaN matrix
               ss
        gene1 NaN
        gene2 NaN
        attr(,"scaled:center")
        gene1 gene2
            1     2
        attr(,"scaled:scale")
        gene1 gene2
            0     0
    """)
    '''
    def test_two_genes_nonzero_clr_nonzero(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_negative_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1],[-1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_negative_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[0, 2], [2, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1],[-1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_two_genes_zero_clr_two_conditions_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_two_genes_zero_clr_two_conditions_zero_gene1_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([[0, 2], [2, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_two_genes_nonzero_clr_two_conditions_positive_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))

    def test_Best_Subset_Regression_all_zero_predictors(self):
        self.X = np.array([[0, 0], [0, 0]])
        self.Y = np.array([1, 2])
        g = np.matrix([1, 1])
        betas = bbsr_python.BestSubsetRegression(self.Y, self.X, g)
        self.assertTrue((betas == [ 0.,  0.]).all())

    def test_PredictErrorReduction_all_zero_predictors(self):
        self.X = np.array([[0, 0], [0, 0]])
        self.Y = np.array([1, 2])
        betas = np.array([ 0.,  0.])
        result = bbsr_python.PredictErrorReduction(self.Y, self.X, betas)
        self.assertTrue((result == [ 0.,  0.]).all())

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_positive_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[0, 2], [0, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']).astype(float))
