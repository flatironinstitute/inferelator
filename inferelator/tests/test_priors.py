"""
Test workflow logic outline using completely
artificial stubs for dependancies.
"""

import unittest
from inferelator.preprocessing.priors import ManagePriors
from inferelator import workflow

import os
import pandas.testing as pdt
import pandas as pd

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
        cls.workflow.expression_matrix_columns_are_genes = False
        cls.workflow.get_data()

    def setUp(self):
        self.priors_data = self.workflow.priors_data.copy()
        self.gold_standard = self.workflow.gold_standard.copy()
        self.data = self.workflow.data.copy()
        self.tf_names = self.workflow.tf_names
        self.gene_list = self.workflow.data.gene_names.tolist()[:35]

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
        npr1 = ManagePriors.filter_priors_to_genes(self.priors_data, self.gene_list)
        self.assertListEqual(npr1.index.tolist(), self.gene_list)

        gene_list2 = self.gene_list + ["fake1", "fake2"]
        npr2 = ManagePriors.filter_priors_to_genes(self.priors_data, gene_list2)
        self.assertListEqual(npr2.index.tolist(), self.gene_list)

        with self.assertRaises(ValueError):
            npr3 = ManagePriors.filter_priors_to_genes(self.priors_data, [])

        with self.assertRaises(ValueError):
            npr3 = self.priors_data.copy()
            npr3.index = list(range(npr3.shape[0]))
            npr3 = ManagePriors.filter_priors_to_genes(npr3, self.gene_list)

    def test_cv_genes(self):
        npr1, ngs1 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 0, 0.5, 42)
        self.assertEqual(npr1.shape, ngs1.shape)
        self.assertEqual(len(npr1.index.intersection(ngs1.index)), 0)
        pdt.assert_index_equal(npr1.columns, self.priors_data.columns)
        pdt.assert_index_equal(ngs1.columns, self.gold_standard.columns)

        npr2, ngs2 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 0, 0.5, 43)
        npr3, ngs3 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 0, 0.5, 42)

        pdt.assert_frame_equal(npr1, npr3)
        pdt.assert_frame_equal(ngs1, ngs3)

        with self.assertRaises(AssertionError):
            pdt.assert_frame_equal(npr1, npr2)

        with self.assertRaises(AssertionError):
            pdt.assert_frame_equal(ngs1, ngs2)

    def test_cv_tfs(self):

        with self.assertWarns(UserWarning):
            npr1, ngs1 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 1, 0.5, 42)

        self.assertEqual(npr1.shape, ngs1.shape)
        self.assertEqual(len(npr1.columns.intersection(ngs1.columns)), 0)
        pdt.assert_index_equal(npr1.index, self.priors_data.index)
        pdt.assert_index_equal(ngs1.index, self.gold_standard.index)

        with self.assertWarns(UserWarning):
            npr2, ngs2 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 1, 0.5, 43)
            npr3, ngs3 = ManagePriors.cross_validate_gold_standard(self.priors_data, self.gold_standard, 1, 0.5, 42)

        pdt.assert_frame_equal(npr1, npr3)
        pdt.assert_frame_equal(ngs1, ngs3)

        with self.assertRaises(AssertionError):
            pdt.assert_frame_equal(npr1, npr2)

        with self.assertRaises(AssertionError):
            pdt.assert_frame_equal(ngs1, ngs2)

    def test_cv_downsample(self):
        with self.assertWarns(UserWarning):
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
        ngs.columns = ["B"] * ngs.shape[1]

        npr = self.priors_data
        npr.index = ["C"] * npr.shape[0]
        npr.columns = ["D"] * npr.shape[1]

        npr1, ngs1 = ManagePriors.validate_priors_gold_standard(npr, ngs)
        pdt.assert_frame_equal(ngs, ngs1)
        pdt.assert_frame_equal(npr, npr1)

    def test_align_priors_1(self):
        npr = self.priors_data.iloc[list(range(10)),:]
        npr = ManagePriors.align_priors_to_expression(npr, self.data.gene_names)
        self.assertEqual(len(npr.index), len(self.data.gene_names))
        self.assertListEqual(npr.columns.tolist(), self.priors_data.columns.tolist())
        self.assertListEqual(npr.index.tolist(), self.data.gene_names.tolist())

    def test_align_priors_2(self):
        npr = self.priors_data
        npr.index = list(range(npr.shape[0]))

        with self.assertRaises(ValueError):
            npr = ManagePriors.align_priors_to_expression(npr, self.data.gene_names)

    def test_shuffle_priors_none(self):
        npr1 = ManagePriors.shuffle_priors(self.priors_data, None, 42)
        pdt.assert_frame_equal(npr1, self.priors_data)

    def test_shuffle_priors_gene(self):
        npr2 = ManagePriors.shuffle_priors(self.priors_data, 0, 42)
        pdt.assert_series_equal(self.priors_data.sum(axis=0), npr2.sum(axis=0))
        pdt.assert_index_equal(npr2.index, self.priors_data.index)
        pdt.assert_index_equal(npr2.columns, self.priors_data.columns)

    def test_shuffle_priors_tf(self):
        npr3 = ManagePriors.shuffle_priors(self.priors_data, 1, 42)
        pdt.assert_series_equal(self.priors_data.sum(axis=1), npr3.sum(axis=1))
        pdt.assert_index_equal(npr3.index, self.priors_data.index)
        pdt.assert_index_equal(npr3.columns, self.priors_data.columns)

    def test_add_noise_to_no_priors(self):
        priors_data = pd.DataFrame(0, index=self.priors_data.index, columns=self.priors_data.columns)
        npr1 = ManagePriors.add_prior_noise(priors_data, 0.1, random_seed=50)

        self.assertEqual((priors_data != 0).sum().sum(), 0)
        self.assertEqual((npr1 != 0).sum().sum(), int(npr1.size * 0.1))

    def test_add_noise_to_priors(self):
        npr1 = ManagePriors.add_prior_noise(self.priors_data, 1, random_seed=50)

        self.assertEqual(npr1.max().max(), 1)
        self.assertEqual((npr1 != 0).sum().sum(), npr1.size)