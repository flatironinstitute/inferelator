import unittest
from inferelator.single_cell_workflow import SingleCellWorkflow
from inferelator.preprocessing import single_cell, metadata_parser
from inferelator.crossvalidation_workflow import create_puppet_workflow
from inferelator.artifacts.test_data import TestDataSingleCellLike
import numpy as np
import pandas as pd
import os

my_dir = os.path.dirname(__file__)


class SingleCellTestCase(unittest.TestCase):
    def setUp(self):
        self.expr = TestDataSingleCellLike.expression_matrix.transpose()
        self.meta = TestDataSingleCellLike.meta_data
        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.gene_list = TestDataSingleCellLike.gene_metadata
        self.tf_names = TestDataSingleCellLike.tf_names
        self.workflow = create_puppet_workflow(base_class=SingleCellWorkflow)(self.expr.transpose(), self.meta,
                                                                              self.prior, self.gold_standard)


class SingleCellPreprocessTest(SingleCellTestCase):

    def test_var_filter(self):
        expr_filtered, _ = single_cell.filter_genes_for_var(self.expr, self.meta)
        self.assertEqual(expr_filtered.columns.tolist(), ["gene1", "gene2", "gene4", "gene6"])

    def test_count_filter(self):
        expr_filtered_1, _ = single_cell.filter_genes_for_count(self.expr, self.meta)
        self.assertEqual(expr_filtered_1.columns.tolist(), ["gene1", "gene2", "gene4", "gene6"])
        expr_filtered_2, _ = single_cell.filter_genes_for_count(self.expr, self.meta, count_minimum=4)
        self.assertEqual(expr_filtered_2.columns.tolist(), ["gene1", "gene2", "gene4"])
        expr_filtered_3, _ = single_cell.filter_genes_for_count(self.expr, self.meta, count_minimum=20)
        self.assertEqual(expr_filtered_3.columns.tolist(), ["gene2"])

        with self.assertRaises(ValueError):
            single_cell.filter_genes_for_count(self.expr - 3, self.meta, count_minimum=1, check_for_scaling=True)

    def test_library_to_one_norm(self):
        expr_normed, _ = single_cell.normalize_expression_to_one(self.expr, self.meta)
        np.testing.assert_almost_equal(expr_normed.sum(axis=1).tolist(), [1] * 10)

    def test_median_scaling_norm(self):
        expr_normed1, meta_data1 = single_cell.normalize_medians_for_batch(self.expr, self.meta,
                                                                           batch_factor_column="Condition")
        meta_data1['umi'] = expr_normed1.sum(axis=1)
        np.testing.assert_almost_equal(meta_data1.groupby("Condition")['umi'].median().tolist(), [45, 45, 45])
        expr_normed2, meta_data2 = single_cell.normalize_medians_for_batch(self.expr, self.meta,
                                                                           batch_factor_column="Genotype")
        meta_data2['umi'] = expr_normed2.sum(axis=1)
        np.testing.assert_almost_equal(meta_data2.groupby("Genotype")['umi'].median().tolist(), [45])

    def test_size_factor_scaling_norm(self):
        expr_normed1, meta_data1 = single_cell.normalize_sizes_within_batch(self.expr, self.meta,
                                                                            batch_factor_column="Condition")
        test_umi = pd.Series({"A": 45.0, "B": 36.0, "C": 58.5})
        meta_data1['umi'] = expr_normed1.sum(axis=1)
        for group in meta_data1['Condition'].unique():
            idx = meta_data1['Condition'] == group
            np.testing.assert_almost_equal(meta_data1.loc[idx, 'umi'].tolist(), [test_umi[group]] * idx.sum())

    def test_log_scaling(self):
        expr_log1, _ = single_cell.log10_data(self.expr, self.meta)
        np.testing.assert_almost_equal(np.log10(self.expr + 1).values, expr_log1)

        expr_log2, _ = single_cell.log2_data(self.expr, self.meta)
        np.testing.assert_almost_equal(np.log2(self.expr + 1).values, expr_log2)

        expr_log3, _ = single_cell.ln_data(self.expr, self.meta)
        np.testing.assert_almost_equal(np.log(self.expr + 1).values, expr_log3)

        expr_sqrt, _ = single_cell.tf_sqrt_data(self.expr, self.meta)
        np.testing.assert_almost_equal(np.sqrt(self.expr + 1).values + np.sqrt(self.expr).values - 1, expr_sqrt)


class SingleCellWorkflowTest(SingleCellTestCase):

    def TestPreprocessing(self):
        self.workflow.add_preprocess_step(single_cell.log2_data)
        self.workflow.single_cell_normalize()
        expr_filtered, _ = single_cell.filter_genes_for_var(self.expr, self.meta)
        np.testing.assert_almost_equal(np.log2(expr_filtered.transpose() + 1).values, self.workflow.expression_matrix)

    def TestFilter(self):
        self.workflow.gene_metadata = self.gene_list
        self.workflow.gene_list_index = "SystematicName"
        self.workflow.tf_names = self.tf_names
        self.workflow.process_priors_and_gold_standard()
        self.workflow.align_priors_and_expression()
        genes = ["gene1", "gene2", "gene3", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(self.workflow.expression_matrix.index.tolist(), genes)
        self.assertEqual(self.workflow.priors_data.index.tolist(), genes)
        self.assertEqual(self.workflow.priors_data.columns.tolist(), tfs)

    def TestStack(self):
        self.workflow.gene_metadata = self.gene_list
        self.workflow.gene_list_index = "SystematicName"
        self.workflow.tf_names = self.tf_names
        self.workflow.startup()
        genes = ["gene1", "gene2", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(self.workflow.design.index.tolist(), tfs)
        self.assertEqual(self.workflow.response.index.tolist(), genes)
        self.assertEqual(self.workflow.response.columns.tolist(), self.workflow.design.columns.tolist())


class TestSingleCellWorkflow(unittest.TestCase):
    test_count_data = pd.DataFrame([[0, 0, 0], [10, 0, 10], [4, 0, 5], [0, 0, 0]])
    test_meta_data = metadata_parser.MetadataParserBranching.create_default_meta_data(test_count_data)

    def setUp(self):
        self.workflow = SingleCellWorkflow()
        self.workflow.expression_matrix_columns_are_genes = True
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")

    def tearDown(self):
        del self.workflow

    def prep1(self, expr, meta, **kwargs):
        return expr, meta

    def prep2(self, expr, meta, **kwargs):
        return expr, meta

    def test_preprocessing_flow(self):
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.get_data()
        self.workflow.add_preprocess_step(self.prep1)
        self.workflow.add_preprocess_step(self.prep2)
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.expression_matrix.shape, (100, 421))

    def test_preprocessing_filter(self):
        self.workflow.expression_matrix = self.test_count_data.transpose()
        self.workflow.meta_data = self.test_meta_data
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.expression_matrix.shape, (2, 4))

    def test_preprocessing_nan_pre(self):
        self.workflow.expression_matrix = self.test_count_data.transpose()
        self.workflow.expression_matrix.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.workflow.single_cell_normalize()

    def test_preprocessing_nan_post(self):
        self.workflow.expression_matrix = self.test_count_data.transpose() - 1
        self.workflow.add_preprocess_step(single_cell.log2_data)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            with self.assertRaises(ValueError):
                self.workflow.single_cell_normalize()
