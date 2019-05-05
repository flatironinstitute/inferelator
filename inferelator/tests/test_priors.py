"""
Test workflow logic outline using completely
artificial stubs for dependancies.
"""

import unittest
from inferelator.preprocessing.priors import ManagePriors
from inferelator import workflow

import os
import numpy as np
import pandas.testing as pdt

my_dir = os.path.dirname(__file__)


class TestPriorManager(unittest.TestCase):
    workflow = None

    @classmethod
    def setUpClass(cls):
        cls.workflow = workflow.WorkflowBase()
        cls.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        cls.workflow.expression_matrix_file = "expression.tsv"
        cls.workflow.meta_data_file = "meta_data.tsv"
        cls.workflow.tf_names_file = "tf_names.tsv"
        cls.workflow.priors_file = "gold_standard.tsv"
        cls.workflow.gold_standard_file = "gold_standard.tsv"
        cls.workflow.get_data()

    def setUp(self):
        self.priors_data = self.workflow.priors_data.copy()
        self.gold_standard = self.workflow.gold_standard.copy()
        self.expression_matrix = self.workflow.expression_matrix.copy()
        self.tf_names = self.workflow.tf_names
        self.gene_list = self.workflow.expression_matrix.index.tolist()[:35]

    def test_priors_tf_names(self):
        npr1 = ManagePriors.filter_to_tf_names_list(self.priors_data, self.tf_names)
        self.assertListEqual(npr1.columns.tolist(), self.tf_names)
        self.assertListEqual(npr1.index.tolist(), self.priors_data.index.tolist())

        npr2 = ManagePriors.filter_to_tf_names_list(self.priors_data, self.tf_names[:10])
        self.assertListEqual(npr2.columns.tolist(), self.tf_names[:10])
        self.assertListEqual(npr2.index.tolist(), self.priors_data.index.tolist())

        npr3 = ManagePriors.filter_to_tf_names_list(self.priors_data, self.tf_names + ["fake1", "fake2"])
        self.assertListEqual(npr3.columns.tolist(), self.tf_names)
        self.assertListEqual(npr3.index.tolist(), self.priors_data.index.tolist())

        with self.assertRaises(ValueError):
            ManagePriors.filter_to_tf_names_list(self.priors_data, ["fake1", "fake2"])

    def test_gene_list_filter(self):
        npr1, nexp1 = ManagePriors.filter_to_gene_list(self.priors_data, self.expression_matrix, self.gene_list)
        self.assertListEqual(nexp1.index.tolist(), self.gene_list)
        self.assertListEqual(npr1.index.tolist(), self.gene_list)

        gene_list2 = self.gene_list + ["fake1", "fake2"]
        npr2, nexp2 = ManagePriors.filter_to_gene_list(self.priors_data, self.expression_matrix, gene_list2)
        self.assertListEqual(nexp2.index.tolist(), self.gene_list)
        self.assertListEqual(npr2.index.tolist(), self.gene_list)

        with self.assertRaises(ValueError):
            nexp3 = self.expression_matrix.copy()
            nexp3.index = list(range(nexp3.shape[0]))
            npr3, nexp3 = ManagePriors.filter_to_gene_list(self.priors_data, nexp3, self.gene_list)

        with self.assertRaises(ValueError):
            npr3 = self.priors_data.copy()
            npr3.index = list(range(npr3.shape[0]))
            npr3, nexp3 = ManagePriors.filter_to_gene_list(npr3, self.expression_matrix, self.gene_list)

    def test_cv_genes(self):
        npr1, ngs1 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 0, 0.5, 42)
        self.assertEqual(npr1.shape, ngs1.shape)
        self.assertEqual(len(npr1.index.intersection(ngs1.index)), 0)

    def test_cv_tfs(self):
        npr1, ngs1 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 1, 0.5, 42)
        self.assertEqual(npr1.shape, ngs1.shape)
        self.assertEqual(len(npr1.columns.intersection(ngs1.columns)), 0)

    def test_cv_downsample(self):
        npr1, ngs1 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, None, 0.5, 42)
        self.assertEqual(npr1.shape, ngs1.shape)
        self.assertListEqual(npr1.index.tolist(), ngs1.index.tolist())
        self.assertListEqual(npr1.columns.tolist(), ngs1.columns.tolist())
        self.assertEqual((npr1 != 0).sum().sum(), (ngs1 != 0).sum().sum() * 0.5)
        self.assertEqual((self.gold_standard != 0).sum().sum(), (ngs1 != 0).sum().sum())
        pdt.assert_frame_equal(self.gold_standard, ngs1)

    def test_shuffle_index(self):
        idx = list(range(20))
        idx1 = ManagePriors._make_shuffled_index(20, seed=42)
        idx2 = ManagePriors._make_shuffled_index(20, seed=42)
        idx3 = ManagePriors._make_shuffled_index(20, seed=43)

        self.assertListEqual(idx1, idx2)
        self.assertFalse(idx == idx1)
        self.assertFalse(idx1 == idx3)
        self.assertEqual(len(set(idx1).symmetric_difference(set(idx))), 0)
        self.assertEqual(len(set(idx1).symmetric_difference(set(idx2))), 0)
        self.assertEqual(len(set(idx1).symmetric_difference(set(idx3))), 0)

    def test_validation_passthrough(self):
        ngs = self.gold_standard
        ngs.index = ["A"] * ngs.shape[0]
        ngs.index = ["B"] * ngs.shape[1]

        npr = self.gold_standard
        npr.index = ["C"] * npr.shape[0]
        npr.index = ["D"] * npr.shape[1]

        ngs1, npr1 = ManagePriors.validate_priors_gold_standard(npr, ngs)
        pdt.assert_frame_equal(ngs, ngs1)
        pdt.assert_frame_equal(npr, npr1)
