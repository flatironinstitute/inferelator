import os
import unittest
import copy

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from inferelator import workflow
from inferelator.tests.artifacts.test_stubs import TaskDataStub
from inferelator.regression import amusr_regression
from inferelator.utils import InferelatorData

data_path = os.path.join(os.path.dirname(__file__), "../../data/dream4")

class TestAMuSRrunner(unittest.TestCase):

    def test_format_priors_noweight(self):
        tfs = ['tf1', 'tf2']
        priors = [pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=tfs),
                  pd.DataFrame([[0, 0], [1, 0]], index=['gene1', 'gene2'], columns=tfs)]
        gene1_prior = amusr_regression.format_prior(priors, 'gene1', [0, 1], 1)
        gene2_prior = amusr_regression.format_prior(priors, 'gene2', [0, 1], 1)
        npt.assert_almost_equal(gene1_prior, np.array([[1., 1.], [1., 1.]]))
        npt.assert_almost_equal(gene2_prior, np.array([[1., 1.], [1., 1.]]))

    def test_format_priors_pweight(self):
        tfs = ['tf1', 'tf2']
        priors = [pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=tfs),
                  pd.DataFrame([[0, 0], [1, 0]], index=['gene1', 'gene2'], columns=tfs)]
        gene1_prior = amusr_regression.format_prior(priors, 'gene1', [0, 1], 1.2)
        gene2_prior = amusr_regression.format_prior(priors, 'gene2', [0, 1], 1.2)
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


class TestAMuSRRegresionEBIC:

    use_numba = False

    def test_amusr_regression(self):
        des = [np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float),
               np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float)]
        res = [np.array([1, 2, 3]).reshape(-1, 1).astype(float),
               np.array([1, 2, 3]).reshape(-1, 1).astype(float)]

        tfs = ['tf1', 'tf2', 'tf3']
        targets = ['gene1', 'gene2']

        priors = [pd.DataFrame([[0, 1, 1], [1, 0, 1]], index=targets, columns=tfs),
                  pd.DataFrame([[0, 0, 1], [1, 0, 1]], index=targets, columns=tfs)]
        
        gene1_prior = amusr_regression.format_prior(priors, 'gene1', [0, 1], 1.)
        gene2_prior = amusr_regression.format_prior(priors, 'gene2', [0, 1], 1.)
        output = [amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene1', gene1_prior,
                                                       scale_data=True, use_numba=self.use_numba),
                  amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene2', gene2_prior,
                                                       scale_data=True, use_numba=self.use_numba)]

        out0 = pd.DataFrame([['tf3', 'gene1', -1, 1],
                             ['tf3', 'gene1', -1, 1]],
                            index=pd.MultiIndex(levels=[[0, 1], [0]],
                                                codes=[[0, 1], [0, 0]]),
                            columns=['regulator', 'target', 'weights', 'resc_weights'])

        out1 = pd.DataFrame([['tf3', 'gene2', -1, 1],
                             ['tf3', 'gene2', -1, 1]],
                            index=pd.MultiIndex(levels=[[0, 1], [0]],
                                                codes=[[0, 1], [0, 0]]),
                            columns=['regulator', 'target', 'weights', 'resc_weights'])
                            
        pdt.assert_frame_equal(pd.concat(output[0]), out0, check_dtype=False)
        pdt.assert_frame_equal(pd.concat(output[1]), out1, check_dtype=False)

    def test_unaligned_regression_genes(self):
        tfs = ['tf1', 'tf2', 'tf3']
        targets = ['gene1', 'gene2', 'gene3']
        targets1 = ['gene1', 'gene2']
        targets2 = ['gene1', 'gene3']

        des = [InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs)),
               InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs))]

        res = [InferelatorData(pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=targets1)),
               InferelatorData(pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=targets2))]
        priors = pd.DataFrame([[0, 1, 1], [1, 0, 1], [1, 0, 1]], index=targets, columns=tfs)

        r = amusr_regression.AMuSR_regression(des, res, tfs=tfs, genes=targets, priors=priors, use_numba=self.use_numba)

        out = [pd.DataFrame([['tf3', 'gene1', -1, 1], ['tf3', 'gene1', -1, 1]],
                            index=pd.MultiIndex(levels=[[0, 1], [0]], codes=[[0, 1], [0, 0]]),
                            columns=['regulator', 'target', 'weights', 'resc_weights']),
               pd.DataFrame([['tf3', 'gene2', -1, 1]],
                            index=pd.MultiIndex(levels=[[0, 1], [0]], codes=[[0], [0]]),
                            columns=['regulator', 'target', 'weights', 'resc_weights']),
               pd.DataFrame([['tf3', 'gene3', -1, 1]],
                            index=pd.MultiIndex(levels=[[0, 1], [0]], codes=[[1], [0]]),
                            columns=['regulator', 'target', 'weights', 'resc_weights'])]

        regress_data = r.regress()
        for i in range(len(targets)):
            pdt.assert_frame_equal(pd.concat(regress_data[i]), out[i], check_dtype=False)

        weights, resc_weights = r.pileup_data(regress_data)

class TestAMuSRREgressionEBICNumba(TestAMuSRRegresionEBIC):

    use_numba = True


class TestAMuSRParams(unittest.TestCase):

    def setUp(self):

        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression="amusr")
        self.workflow.create_output_dir = lambda *x: None

        tfs = ['tf1', 'tf2', 'tf3']
        targets = ['gene1', 'gene2']

        self.workflow._task_design = [
            InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs)),
            InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs))
        ]

        self.workflow._task_response = [
            InferelatorData(pd.DataFrame(np.array([[1, 1], [2, 0], [3, 0]]).astype(float), columns=targets)),
            InferelatorData(pd.DataFrame(np.array([[1, 3], [2, 3], [3, 3]]).astype(float), columns=targets))
        ]

        self.priors_data = pd.DataFrame([[0, 1, 1], [1, 0, 1]], index=targets, columns=tfs)
        self.gold_standard = self.priors_data.copy()

    def test_lamb_b(self):

        lamb_b = np.arange(0, 5, dtype=float)

        regress = amusr_regression.AMuSR_regression(self.workflow._task_design, self.workflow._task_response,
                                                    priors=self.workflow.priors_data, lambda_Bs=lamb_b)

        def is_passed(X, Y, TFs, tasks, gene, prior, Cs=None, Ss=None, lambda_Bs=None,
                      lambda_Ss=None, scale_data=False, **kwargs):

            npt.assert_array_equal(lambda_Bs, lamb_b)

            return amusr_regression.run_regression_EBIC(X, Y, TFs, tasks, gene, prior, Cs, Ss, lambda_Bs,
                                                        lambda_Ss, scale_data)

        regress.regression_function = is_passed

        self.workflow.set_regression_parameters(lambda_Bs=lamb_b)
        npt.assert_array_equal(self.workflow.lambda_Bs, lamb_b)

        output = regress.run()

    def test_lamb_s(self):

        lamb_s = np.arange(0, 5, dtype=float)

        regress = amusr_regression.AMuSR_regression(self.workflow._task_design, self.workflow._task_response,
                                                    priors=self.workflow.priors_data, lambda_Ss=lamb_s)

        def is_passed(X, Y, TFs, tasks, gene, prior, Cs=None, Ss=None, lambda_Bs=None,
                      lambda_Ss=None, scale_data=False,  **kwargs):

            npt.assert_array_equal(lambda_Ss, lamb_s)

            return amusr_regression.run_regression_EBIC(X, Y, TFs, tasks, gene, prior, Cs, Ss, lambda_Bs,
                                                        lambda_Ss, scale_data)

        regress.regression_function = is_passed

        self.workflow.set_regression_parameters(lambda_Ss=lamb_s)
        npt.assert_array_equal(self.workflow.lambda_Ss, lamb_s)

        output = regress.run()

    def test_heuristic_c(self):

        set_Cs = np.arange(0, 5, dtype=float) / 10

        regress = amusr_regression.AMuSR_regression(self.workflow._task_design, self.workflow._task_response,
                                                    priors=self.workflow.priors_data, Cs=set_Cs)

        def is_passed(X, Y, TFs, tasks, gene, prior, Cs=None, Ss=None, lambda_Bs=None,
                      lambda_Ss=None, scale_data=False, **kwargs):

            npt.assert_array_equal(set_Cs, Cs)

            return amusr_regression.run_regression_EBIC(X, Y, TFs, tasks, gene, prior, Cs, Ss, lambda_Bs,
                                                        lambda_Ss, scale_data)

        regress.regression_function = is_passed

        self.workflow.set_regression_parameters(heuristic_Cs=set_Cs)
        npt.assert_array_equal(self.workflow.heuristic_Cs, set_Cs)

        output = regress.run()
