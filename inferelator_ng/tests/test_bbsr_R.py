import unittest, os
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
from subprocess import CalledProcessError
from .. import bbsr_R

my_dir = os.path.dirname(__file__)

class TestDR(unittest.TestCase):
    """
    Superclass for common methods
    """
    def test_two_genes(self):
        brd = bbsr_R.BBSR_driver()
        brd.target_directory = os.path.join(my_dir, "artifacts")
        self.X = pd.DataFrame([0, 0], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([0, 0], index = ['gene1', 'gene2'], columns = ['ss'])
        self.priors =  pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        self.clr = pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = brd.run(self.X, self.Y, self.clr, self.priors)
        self.assertEqual(betas.shape, (2, 2))
        self.assertEqual(resc.shape, (2, 2))
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_fails_with_one_gene(self):
        brd = bbsr_R.BBSR_driver()
        brd.target_directory = os.path.join(my_dir, "artifacts")
        self.X = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.Y = pd.DataFrame([0], index = ['gene1'], columns = ['ss'])
        self.priors = pd.DataFrame([0], index = ['gene1'], columns = ['gene1'])
        self.clr = pd.DataFrame([0], index = ['gene1'], columns = ['gene1'])
        self.assertRaises(CalledProcessError, brd.run, self.X, self.Y, self.clr, self.priors)

    def test_two_genes_nonzero(self):
        brd = bbsr_R.BBSR_driver()
        brd.target_directory = os.path.join(my_dir, "artifacts")
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.priors =  pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        self.clr = pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = brd.run(self.X, self.Y, self.clr, self.priors)
        self.assertEqual(betas.shape, (2, 2))
        self.assertEqual(resc.shape, (2, 2))
        print betas
        print resc
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))

    def test_two_genes_nonzero_clr_nonzero(self):
        brd = bbsr_R.BBSR_driver()
        brd.target_directory = os.path.join(my_dir, "artifacts")
        self.X = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.Y = pd.DataFrame([1, 2], index = ['gene1', 'gene2'], columns = ['ss'])
        self.priors =  pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        self.clr = pd.DataFrame([[.1, .1],[.1, .2]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2'])
        (betas, resc) = brd.run(self.X, self.Y, self.clr, self.priors)
        self.assertEqual(betas.shape, (2, 2))
        self.assertEqual(resc.shape, (2, 2))
        print betas
        print resc
        pdt.assert_frame_equal(betas, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
        pdt.assert_frame_equal(resc, pd.DataFrame([[0, 0],[0, 0]], index = ['gene1', 'gene2'], columns = ['gene1', 'gene2']))
