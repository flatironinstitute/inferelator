import unittest
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
from inferelator_ng import amusr_regression
import numpy.testing as npt

class TestAMuSRrunner(unittest.TestCase):

    def test_format_priors_noweight(self):
        runner = amusr_regression.AMuSR_regression([pd.DataFrame()], [pd.DataFrame()], None)
        tfs = ['tf1', 'tf2']
        priors = [pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = tfs),
                  pd.DataFrame([[0, 0], [1, 0]], index = ['gene1', 'gene2'], columns = tfs)]
        gene1_prior = runner.format_prior(priors, 'gene1', [0, 1], 1)
        gene2_prior = runner.format_prior(priors, 'gene2', [0, 1], 1)
        npt.assert_almost_equal(gene1_prior, np.array([[1., 1.], [1., 1.]]))
        npt.assert_almost_equal(gene2_prior, np.array([[1., 1.], [1., 1.]]))

    def test_format_priors_pweight(self):
        runner = amusr_regression.AMuSR_regression([pd.DataFrame()], [pd.DataFrame()], None)
        tfs = ['tf1', 'tf2']
        priors = [pd.DataFrame([[0, 1], [1, 0]], index = ['gene1', 'gene2'], columns = tfs),
                  pd.DataFrame([[0, 0], [1, 0]], index = ['gene1', 'gene2'], columns = tfs)]
        gene1_prior = runner.format_prior(priors, 'gene1', [0, 1], 1.2)
        gene2_prior = runner.format_prior(priors, 'gene2', [0, 1], 1.2)
        npt.assert_almost_equal(gene1_prior, np.array([[1.09090909, 1.], [0.90909091, 1.]]))
        npt.assert_almost_equal(gene2_prior, np.array([[0.90909091, 0.90909091], [1.09090909, 1.09090909]]))

    def test_sum_squared_errors(self):
        X = [np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
             np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])]
        Y = [np.array([3, 3, 3]),
             np.array([3, 3, 3])]
        W = np.array([[1, 0], [1, 0], [1, 0]])
        self.assertEqual(amusr_regression.sum_squared_errors(X, Y, W, 0), 0)
        self.assertEqual(amusr_regression.sum_squared_errors(X, Y, W, 1), 27)

    def test_amusr_regression(self):

        des = [np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]), np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]])]
        res = [np.array([1, 2, 3]).reshape(-1, 1), np.array([1, 2, 3]).reshape(-1, 1)]
        tfs = ['tf1', 'tf2', 'tf3']
        targets = ['gene1', 'gene2']
        priors = [pd.DataFrame([[0, 1, 1], [1, 0, 1]], index = targets, columns = tfs),
                  pd.DataFrame([[0, 0, 1], [1, 0, 1]], index = targets, columns = tfs)]
        runner = amusr_regression.AMuSR_regression([pd.DataFrame(des[0], columns=tfs)],
                                                   [pd.DataFrame(res[0], columns=["gene1"])],
                                                   None)
        gene1_prior = runner.format_prior(priors, 'gene1', [0, 1], 1.)
        gene2_prior = runner.format_prior(priors, 'gene2', [0, 1], 1.)
        output = []
        output.append(amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene1', gene1_prior))
        output.append(amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene2', gene2_prior))
        out0 = pd.DataFrame([['tf3', 'gene1', -1, 1],
                             ['tf3', 'gene1', -1, 1]],
                            index = pd.MultiIndex(levels=[[0, 1], [0]],
                                                  labels=[[0, 1], [0, 0]]),
                            columns = ['regulator', 'target', 'weights', 'resc_weights'])
        out1 = pd.DataFrame([['tf3', 'gene2', -1, 1],
                             ['tf3', 'gene2', -1, 1]],
                            index = pd.MultiIndex(levels=[[0, 1], [0]],
                                                  labels=[[0, 1], [0, 0]]),
                            columns = ['regulator', 'target', 'weights', 'resc_weights'])
        pdt.assert_frame_equal(pd.concat(output[0]), out0, check_dtype=False)
        pdt.assert_frame_equal(pd.concat(output[1]), out1, check_dtype=False)