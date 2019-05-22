import unittest

from inferelator.crossvalidation_workflow import create_puppet_workflow
from inferelator import tfa_workflow
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask
from inferelator.regression.bbsr_multitask import BBSRByTaskRegressionWorkflow

import pandas as pd


class TestRegressionFactory(unittest.TestCase):

    def setUp(self):
        self.expr = pd.DataFrame([[2, 28, 1, 16, 2, 3], [6, 21, 1, 3, 0, 3], [4, 39, 0, 17, 1, 3],
                                  [8, 34, 0, 7, 0, 3], [6, 26, 0, 3, 1, 3], [1, 31, 0, 1, 1, 4],
                                  [3, 27, 1, 5, 2, 4], [8, 34, 2, 9, 2, 3], [1, 22, 2, 3, 3, 4],
                                  [9, 33, 0, 17, 1, 2]],
                                 columns=["gene1", "gene2", "gene3", "gene4", "gene5", "gene6"])
        self.meta = pd.DataFrame({"Condition": ["A", "B", "C", "C", "B", "B", "A", "C", "B", "C"],
                                  "Genotype": ['WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT', 'WT']})
        self.prior = pd.DataFrame([[0, 1], [0, 1], [1, 0], [0, 0]], index=["gene1", "gene2", "gene4", "gene5"],
                                  columns=["gene3", "gene6"])
        self.gold_standard = self.prior.copy()
        self.gene_list = pd.DataFrame({"SystematicName": ["gene1", "gene2", "gene3", "gene4", "gene7", "gene6"]})
        self.tf_names = ["gene3", "gene6"]


    def test_base(self):

        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow)
        self.workflow = self.workflow(self.expr.transpose(), self.meta, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.read_metadata()
        with self.assertRaises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):

        self.workflow = create_puppet_workflow(base_class="tfa",
                                               regression_class="bbsr")
        self.workflow = self.workflow(self.expr.transpose(), self.meta, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.read_metadata()
        self.workflow.run()
        self.assertEqual(self.workflow.aupr, 1)

    def test_elasticnet(self):

        self.workflow = create_puppet_workflow(base_class="tfa",
                                               regression_class="elasticnet")
        self.workflow = self.workflow(self.expr.transpose(), self.meta, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.read_metadata()
        self.workflow.run()
        self.assertEqual(self.workflow.aupr, 1)

    def test_amusr(self):
        self.workflow = create_puppet_workflow(base_class="amusr",
                                               regression_class="amusr",
                                               result_processor_class=ResultsProcessorMultiTask)

        self.workflow = self.workflow([self.expr.transpose(), self.expr.transpose()],
                                      [self.meta, self.meta], self.prior, self.gold_standard)

        self.workflow.n_tasks = 2
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.run()
        self.assertAlmostEqual(self.workflow.aupr, 0.81666, places=4)

    def test_mtl_bbsr(self):
        self.workflow = create_puppet_workflow(base_class="amusr",
                                               regression_class=BBSRByTaskRegressionWorkflow,
                                               result_processor_class=ResultsProcessorMultiTask)

        self.workflow = self.workflow([self.expr.transpose(), self.expr.transpose()],
                                      [self.meta, self.meta], self.prior, self.gold_standard)

        self.workflow.n_tasks = 2
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.run()
        self.assertEqual(self.workflow.aupr, 1)
