import os
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

from inferelator import workflow
from inferelator.tests.artifacts.test_stubs import TaskDataStub
from inferelator.regression import amusr_regression

data_path = os.path.join(os.path.dirname(__file__), "../../data/dream4")


class TestAMuSRWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression="amusr")
        self.workflow.create_output_dir = lambda *x: None
        self.workflow.gold_standard = TaskDataStub.priors_data.copy()

    def test_create_task(self):
        self.assertIsNone(self.workflow.task_objects)
        self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                  meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                  priors_file="gold_standard.tsv")
        self.assertEqual(len(self.workflow.task_objects), 1)

        new_task = self.workflow.task_objects[0]

        self.assertEqual(new_task.expression_matrix_file, "expression.tsv")
        self.assertIsNone(new_task.expression_matrix)

        self.assertEqual(new_task.meta_data_file, "meta_data.tsv")
        self.assertIsNone(new_task.meta_data)

        self.assertEqual(new_task.priors_file, "gold_standard.tsv")
        self.assertIsNone(new_task.priors_data)

        self.assertEqual(new_task.tf_names_file, "tf_names.tsv")
        self.assertIsNone(new_task.tf_names)

    def test_taskdata_loading(self):
        self.assertIsNone(self.workflow.task_objects)
        self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                  meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                  priors_file="gold_standard.tsv")
        self.assertEqual(len(self.workflow.task_objects), 1)
        self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=None,
                                  meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                  priors_file="gold_standard.tsv")
        self.assertEqual(len(self.workflow.task_objects), 2)
        self.workflow.input_dir = data_path
        self.workflow._load_tasks()

        task1 = self.workflow.task_objects[0]
        task2 = self.workflow.task_objects[1]

        self.assertEqual(task1.expression_matrix.shape, (100, 421))
        np.testing.assert_allclose(task1.expression_matrix.sum().sum(), 13507.22145160)
        self.assertEqual(len(task1.tf_names), 100)
        self.assertListEqual(task1.tf_names, list(map(lambda x: "G" + str(x), list(range(1, 101)))))
        self.assertEqual(task1.priors_data.shape, (100, 100))
        self.assertEqual(task1.meta_data.shape, (421, 5))

        self.assertEqual(task1.input_dir, task2.input_dir)

    def test_taskdata_processing(self):
        self.workflow.task_objects = [TaskDataStub()]
        # Test the TaskData processing
        self.assertEqual(len(self.workflow.task_objects), 1)
        self.workflow._load_tasks()
        self.assertEqual(len(self.workflow.task_objects), 3)

        # Test processing the TaskData objects into data structures in MultitaskLearningWorkflow
        self.assertEqual(self.workflow.n_tasks, 3)
        self.assertEqual(list(map(lambda x: x.expression_matrix.shape, self.workflow.task_objects)),
                         [(6, 2), (6, 4), (6, 4)])
        self.assertEqual(list(map(lambda x: x.meta_data.shape, self.workflow.task_objects)),
                         [(2, 2), (4, 2), (4, 2)])

    def test_task_processing(self):
        self.workflow.task_objects = [TaskDataStub()]
        self.workflow._load_tasks()
        self.workflow.startup_finish()
        self.assertEqual(self.workflow.regulators.tolist(), ["gene3", "gene6"])
        self.assertEqual(self.workflow.targets.tolist(), ["gene1", "gene2", "gene4", "gene6"])
        self.assertEqual(len(self.workflow.task_design), 3)
        self.assertEqual(len(self.workflow.task_response), 3)
        self.assertEqual(len(self.workflow.task_meta_data), 3)
        self.assertEqual(len(self.workflow.task_bootstraps), 3)
        pdt.assert_frame_equal(self.workflow.task_design[0],
                               pd.DataFrame([[16., 5.], [15., 15.]], index=["gene3", "gene6"], columns=[0, 6]),
                               check_dtype=False)
        pdt.assert_frame_equal(self.workflow.task_response[0],
                               pd.DataFrame([[2, 3], [28, 27], [16, 5], [3, 4]],
                                            index=["gene1", "gene2", "gene4", "gene6"], columns=[0, 6]),
                               check_dtype=False)

    def test_result_processor_random(self):
        self.workflow.task_objects = [TaskDataStub()]
        self.workflow._load_tasks()

        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0], [0, 1], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta3 = pd.DataFrame(np.array([[0.5, 0.2], [0.5, 0.1], [0.5, 0.2], [0.5, 0.2]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb1 = pd.DataFrame(np.array([[0.75, 0], [0.25, 0], [0.75, 0], [0.25, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb2 = pd.DataFrame(np.array([[0, 0], [1, 0], [0, 0], [1, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb3 = pd.DataFrame(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])

        self.workflow.startup_finish()
        self.workflow.gold_standard_filter_method = 'overlap'
        self.workflow.emit_results([[beta1, beta1], [beta2, beta2], [beta3, beta3]],
                                   [[rb1, rb1], [rb2, rb2], [rb3, rb3]],
                                   self.workflow.gold_standard,
                                   self.workflow.priors_data)
        self.assertAlmostEqual(self.workflow.results.score, 0.37777, places=4)

    def test_result_processor_perfect(self):
        self.workflow.task_objects = [TaskDataStub()]
        self.workflow._load_tasks()

        beta1 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta2 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta3 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0.2]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb1 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.25, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb2 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb3 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0.5]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])

        self.workflow.startup_finish()
        self.workflow.emit_results([[beta1, beta1], [beta2, beta2], [beta3, beta3]],
                                   [[rb1, rb1], [rb2, rb2], [rb3, rb3]],
                                   self.workflow.gold_standard,
                                   self.workflow.priors_data)
        self.assertAlmostEqual(self.workflow.results.score, 1)


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
        output = []
        output.append(
            amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene1', gene1_prior))
        output.append(
            amusr_regression.run_regression_EBIC(des, res, ['tf1', 'tf2', 'tf3'], [0, 1], 'gene2', gene2_prior))
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
        des = [pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs),
               pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=tfs)]
        res = [pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=targets1),
               pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=targets2)]
        priors = pd.DataFrame([[0, 1, 1], [1, 0, 1], [1, 0, 1]], index=targets, columns=tfs)

        r = amusr_regression.AMuSR_regression(des, res, tfs=tfs, genes=targets, priors=priors)

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
