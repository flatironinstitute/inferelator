import unittest

from inferelator_ng.single_cell_puppeteer_workflow import create_puppet_workflow
from inferelator_ng import tfa_workflow
from inferelator_ng import regression
from inferelator_ng import bbsr_python
from inferelator_ng import elasticnet_python
from inferelator_ng import amusr_regression

import pandas as pd

class TestPatching(unittest.TestCase):

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
        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow)
        self.workflow = self.workflow(self.expr.transpose(), self.meta, self.prior,self.gold_standard)
        self.workflow.gene_list = pd.DataFrame({"SystematicName":
                                                    ["gene1", "gene2", "gene3", "gene4", "gene7", "gene6"]})
        self.workflow.tf_names = ["gene3", "gene6"]
        self.workflow.meta_data = self.workflow.create_default_meta_data(self.workflow.expression_matrix)

    def test_base(self):

        with self.assertRaises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):

        self.workflow.regression_type = bbsr_python
        self.workflow.regression_type.patch_workflow(self.workflow)
        self.workflow.run()
        self.assertEqual(self.workflow.aupr, 1)

