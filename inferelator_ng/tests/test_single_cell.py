import unittest
from inferelator_ng import single_cell
from inferelator_ng import single_cell_puppeteer_workflow
import numpy as np
import pandas as pd


class SingleCellTestCase(unittest.TestCase):
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
        self.gene_list = pd.DataFrame({"SystematicName":["gene1", "gene2", "gene3", "gene4", "gene7", "gene6"]})
        self.tf_names = ["gene3", "gene6"]
        self.workflow = single_cell_puppeteer_workflow.create_puppet_workflow()(None, 0, self.expr, self.meta,
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


class SingleCellWorkflowTest(SingleCellTestCase):

    def TestPreprocessing(self):
        self.workflow.add_preprocess_step(single_cell.log2_data)
        self.workflow.single_cell_normalize()
        expr_filtered, _ = single_cell.filter_genes_for_var(self.expr, self.meta)
        np.testing.assert_almost_equal(np.log2(expr_filtered + 1).values, self.workflow.expression_matrix)

    def TestFilter(self):
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.filter_expression_and_priors()
        genes = ["gene1", "gene2", "gene3", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(self.workflow.expression_matrix.index.tolist(), genes)
        self.assertEqual(self.workflow.priors_data.index.tolist(), genes)
        self.assertEqual(set(self.workflow.priors_data.columns.tolist()), set(tfs))

    def TestStack(self):
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.startup()
        genes = ["gene1", "gene2", "gene4", "gene6"]
        tfs = ["gene3", "gene6"]
        self.assertEqual(set(self.workflow.design.index.tolist()), set(tfs))
        self.assertEqual(self.workflow.response.index.tolist(), genes)
        self.assertEqual(self.workflow.response.columns.tolist(), self.workflow.design.columns.tolist())
