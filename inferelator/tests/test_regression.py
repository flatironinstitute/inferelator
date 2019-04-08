import unittest

from inferelator.crossvalidation_workflow import create_puppet_workflow
from inferelator import tfa_workflow
from inferelator.regression import base_regression
from inferelator.regression import bbsr_python
from inferelator.regression import elasticnet_python
from inferelator.regression import amusr_regression

import pandas as pd


class TestRegressionFactory(unittest.TestCase):

    def setUp(self):
        self.expr = pd.DataFrame([[2, 28, 0, 16, 1, 3], [6, 21, 0, 3, 1, 3], [4, 39, 0, 17, 1, 3],
                                  [8, 34, 0, 7, 1, 3], [6, 26, 0, 3, 1, 3], [1, 31, 0, 1, 1, 4],
                                  [3, 27, 0, 5, 1, 4], [8, 34, 0, 9, 1, 3], [1, 22, 0, 3, 1, 4],
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
        self.workflow.meta_data = self.workflow.create_default_meta_data(self.workflow.expression_matrix)
        with self.assertRaises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):

        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow,
                                               regression_class=bbsr_python.BBSRRegressionWorkflow)
        self.workflow = self.workflow(self.expr.transpose(), self.meta, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data = self.workflow.create_default_meta_data(self.workflow.expression_matrix)
        self.workflow.run()
        self.assertEqual(self.workflow.aupr, 1)

