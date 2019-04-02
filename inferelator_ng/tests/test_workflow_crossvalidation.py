import unittest
import os
import numpy as np
import pandas as pd

from inferelator_ng import workflow
from inferelator_ng import crossvalidation_workflow

my_dir = os.path.dirname(__file__)


class TestCVWorkers(unittest.TestCase):

    def setUp(self):
        self.data = workflow.WorkflowBase()
        self.data.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.data.get_data()
        self.beta = [pd.DataFrame(np.array([[0, 1], [0.5, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        self.beta_resc = [pd.DataFrame(np.array([[0, 1], [1, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        self.prior = pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=['tf1', 'tf2'])
        self.gold_standard = pd.DataFrame([[0, 1], [1, 0]], index=['gene1', 'gene2'], columns=['tf1', 'tf2'])
        self.puppet_workflow = crossvalidation_workflow.create_puppet_workflow()
        self.puppet_workflow = self.puppet_workflow(self.data.expression_matrix, self.data.meta_data,
                                                    self.data.priors_data, self.data.gold_standard)

    def test_cv_worker(self):
        self.puppet_workflow.startup_run()
        with self.assertRaises(NotImplementedError):
            self.puppet_workflow.startup_finish()

    def test_cv_worker_postprocess(self):
        self.puppet_workflow.write_network = False
        self.puppet_workflow.gold_standard_filter_method = 'keep_all_gold_standard'
        self.puppet_workflow.emit_results(betas=self.beta, rescaled_betas=self.beta_resc, priors=self.prior,
                                          gold_standard=self.gold_standard)
        self.assertEqual(self.puppet_workflow.aupr, 1)
        self.assertEqual(self.puppet_workflow.precision_interact, 3)
        self.assertEqual(self.puppet_workflow.n_interact, 0)


class WorkflowPuppeteer(crossvalidation_workflow.PuppeteerWorkflow, workflow.WorkflowBase):
    pass


class TestCVMakers(unittest.TestCase):

    def setUp(self):
        self.workflow = WorkflowPuppeteer()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.get_data()

    def test_puppet_maker(self):
        puppet_workflow = self.workflow.new_puppet(self.workflow.expression_matrix, self.workflow.meta_data)
        self.assertTrue(puppet_workflow.pr_curve_file_name is not None)
        self.assertTrue(puppet_workflow.network_file_name is not None)
        self.assertTrue(isinstance(puppet_workflow, workflow.WorkflowBase))
