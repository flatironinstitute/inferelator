import unittest, os
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
from subprocess import CalledProcessError
from .. import bbsr_R

my_dir = os.path.dirname(__file__)

class TestDR(unittest.TestCase):

    def setUp(self):
        self.brd = bbsr_R.BBSR_driver()
        self.brd.target_directory = os.path.join(my_dir, "artifacts")

        if not os.path.exists(self.brd.target_directory):
            os.makedirs(self.brd.target_directory)

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
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_fails_with_one_gene(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.Y = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.assertRaises(CalledProcessError, self.brd.run, self.X, self.Y, self.clr, self.priors)

    def test_two_genes_nonzero(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    # BBSR fails when there's only one column in the design (or response) matrix
    # That seems like unexpected behavior to me. If it is expected, there should be checks for it earlier -Nick DV
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
    def test_two_genes_nonzero_clr_nonzero(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_nonzero_clr_two_conditions_negative_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1],[-1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_negative_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[0, 2], [2, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, -1],[-1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_zero_clr_two_conditions_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [2, 1]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_zero_clr_two_conditions_zero_gene1_zero_betas(self):
        self.set_all_zero_priors()
        self.set_all_zero_clr()
        self.X = pd.DataFrame([[0, 2], [2, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_nonzero_clr_two_conditions_positive_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_nonzero_clr_two_conditions_zero_gene1_positive_influence(self):
        self.set_all_zero_priors()
        self.X = pd.DataFrame([[0, 2], [0, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.Y = pd.DataFrame([[1, 2], [1, 2]], index = ['gene1', 'gene2'], columns = ['ss1', 'ss2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = self.brd.run(self.X, self.Y, self.clr, self.priors)
        self.assert_matrix_is_square(2, betas)
        self.assert_matrix_is_square(2, resc)
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 1],[1, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))