import unittest
import warnings

from inferelator.workflows.single_cell_workflow import SingleCellWorkflow
from inferelator.preprocessing import single_cell, metadata_parser
from inferelator.tests.artifacts.test_stubs import TestDataSingleCellLike, create_puppet_workflow, TEST_DATA
import numpy as np
import pandas as pd
import os

my_dir = os.path.dirname(__file__)

test_count_data = pd.DataFrame([[0, 0, 0], [10, 0, 10], [4, 0, 5], [0, 0, 0]])
test_meta_data = metadata_parser.MetadataParserBranching.create_default_meta_data(test_count_data.index)


class SingleCellTestCase(unittest.TestCase):
    def setUp(self):
        self.data = TEST_DATA.copy()
        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names
        self.workflow = create_puppet_workflow(base_class=SingleCellWorkflow)(self.data, self.prior, self.gold_standard)
        self.gene_data = TestDataSingleCellLike.gene_metadata
        self.gene_list_index = TestDataSingleCellLike.gene_list_index


class SingleCellPreprocessTest(SingleCellTestCase):

    def test_count_filter(self):
        expr_filtered_1 = self.data.copy()
        single_cell.filter_genes_for_count(expr_filtered_1)
        self.assertEqual(expr_filtered_1.gene_names.tolist(), ["gene1", "gene2", "gene4", "gene6"])

        expr_filtered_2 = self.data.copy()
        single_cell.filter_genes_for_count(expr_filtered_2, count_minimum=4)
        self.assertEqual(expr_filtered_2.gene_names.tolist(), ["gene1", "gene2", "gene4"])

        expr_filtered_3 = self.data.copy()
        single_cell.filter_genes_for_count(expr_filtered_3, count_minimum=20)
        self.assertEqual(expr_filtered_3.gene_names.tolist(), ["gene2"])

        with self.assertRaises(ValueError):
            self.data.subtract(3)
            single_cell.filter_genes_for_count(self.data, count_minimum=1)

    def test_library_to_one_norm(self):
        single_cell.normalize_expression_to_one(self.data)
        np.testing.assert_almost_equal(self.data.expression_data.sum(axis=1).tolist(), [1] * 10)

    def test_median_scaling_norm(self):
        data = self.data.copy()
        single_cell.normalize_medians_for_batch(data, batch_factor_column="Condition")
        data.meta_data['umi'] = data.expression_data.sum(axis=1)
        np.testing.assert_almost_equal(data.meta_data.groupby("Condition")['umi'].median().tolist(), [45, 45, 45])

        data = self.data.copy()
        single_cell.normalize_medians_for_batch(data, batch_factor_column="Genotype")
        data.meta_data['umi'] = data.expression_data.sum(axis=1)
        np.testing.assert_almost_equal(data.meta_data.groupby("Genotype")['umi'].median().tolist(), [45])

    def test_size_factor_scaling_norm(self):
        single_cell.normalize_sizes_within_batch(self.data, batch_factor_column="Condition")
        test_umi = pd.Series({"A": 45.0, "B": 36.0, "C": 58.5})
        meta_data1 = self.data.meta_data
        meta_data1['umi'] = np.sum(self.data.expression_data, axis=1)
        for group in meta_data1['Condition'].unique():
            idx = meta_data1['Condition'] == group
            np.testing.assert_almost_equal(meta_data1.loc[idx, 'umi'].tolist(), [test_umi[group]] * idx.sum(),
                                           decimal=4)

    def test_log_scaling(self):
        data = self.data.copy()
        single_cell.log10_data(data)
        np.testing.assert_almost_equal(np.log10(self.data.expression_data + 1), data.expression_data)

        data = self.data.copy()
        single_cell.log2_data(data)
        np.testing.assert_almost_equal(np.log2(self.data.expression_data + 1), data.expression_data)

        data = self.data.copy()
        single_cell.ln_data(data)
        np.testing.assert_almost_equal(np.log(self.data.expression_data + 1), data.expression_data)

        data = self.data.copy()
        single_cell.tf_sqrt_data(data)
        np.testing.assert_almost_equal(np.sqrt(self.data.expression_data + 1) + np.sqrt(self.data.expression_data) - 1,
                                       data.expression_data)


class SingleCellWorkflowTest(SingleCellTestCase):

    def TestPreprocessing(self):
        self.workflow.add_preprocess_step(single_cell.log2_data)
        self.workflow.single_cell_normalize()
        self.workflow.data.trim_genes()
        compare_data = TEST_DATA.copy()
        compare_data.trim_genes()
        np.testing.assert_almost_equal(np.log2(compare_data.expression_data + 1), self.workflow.data.expression_data)

    def TestFilter(self):
        self.workflow.tf_names = self.tf_names
        self.workflow.process_priors_and_gold_standard()
        self.workflow.align_priors_and_expression()
        genes = ["gene1", "gene2", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(self.workflow.data.gene_names.tolist(), genes)
        self.assertEqual(self.workflow.priors_data.index.tolist(), genes)
        self.assertEqual(self.workflow.priors_data.columns.tolist(), tfs)

    def TestStack(self):
        self.workflow.tf_names = self.tf_names
        self.workflow.startup()
        genes = ["gene1", "gene2", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(self.workflow.design.gene_names.tolist(), tfs)
        self.assertEqual(self.workflow.response.gene_names.tolist(), genes)
        self.assertEqual(self.workflow.response.sample_names.tolist(), self.workflow.design.sample_names.tolist())


class TestSingleCellWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = SingleCellWorkflow()
        self.workflow.set_file_paths(input_dir=os.path.join(my_dir, "../../data/dream4"),
                                     expression_matrix_file="expression.tsv",
                                     meta_data_file="meta_data.tsv",
                                     priors_file="gold_standard.tsv",
                                     gold_standard_file="gold_standard.tsv")
        self.workflow.set_file_properties(expression_matrix_columns_are_genes=False)

    def tearDown(self):
        del self.workflow

    def prep1(self, data, **kwargs):
        pass

    def prep2(self, data, **kwargs):
        pass

    def test_preprocessing_flow(self):
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.get_data()
        self.workflow.add_preprocess_step(self.prep1)
        self.workflow.add_preprocess_step(self.prep2)
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.data.shape, (421, 100))

    def test_preprocessing_filter(self):
        self.workflow.data = TEST_DATA.copy()
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.data.shape, (10, 4))

    def test_preprocessing_nan_pre(self):
        self.workflow.data = TEST_DATA.copy()
        self.workflow.data.convert_to_float()
        self.workflow.data.expression_data[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.workflow.single_cell_normalize()

    def test_preprocessing_nan_post(self):
        self.workflow.data = TEST_DATA.copy()
        self.workflow.data._adata.X -= 3
        self.workflow.add_preprocess_step(single_cell.log2_data)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with self.assertRaises(ValueError):
                self.workflow.single_cell_normalize()
