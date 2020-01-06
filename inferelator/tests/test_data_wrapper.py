import unittest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt
from scipy import sparse, linalg
from anndata import AnnData
from inferelator.tests.artifacts.test_data import TestDataSingleCellLike, CORRECT_GENES_INTERSECT, CORRECT_GENES_NZ_VAR
from inferelator.utils import InferelatorData


class TestWrapperSetup(unittest.TestCase):

    def setUp(self):
        self.expr = TestDataSingleCellLike.expression_matrix.copy().T
        self.expr_sparse = sparse.csr_matrix(TestDataSingleCellLike.expression_matrix.values).T.astype(np.int32)
        self.meta = TestDataSingleCellLike.meta_data.copy()
        self.adata = InferelatorData(self.expr, transpose_expression=False)
        self.adata_sparse = InferelatorData(self.expr_sparse,
                                            gene_names=TestDataSingleCellLike.expression_matrix.index,
                                            transpose_expression=False,
                                            meta_data=TestDataSingleCellLike.meta_data.copy())


class TestCreate(TestWrapperSetup):

    def test_create_df(self):
        adata = InferelatorData(self.expr)
        npt.assert_array_equal(adata.expression_data, self.expr.values)

    def test_create_df_transpose(self):
        adata = InferelatorData(self.expr, transpose_expression=True)
        npt.assert_array_equal(adata.expression_data, self.expr.values.T)

    def test_create_array(self):
        adata = InferelatorData(self.expr.values, gene_names=self.expr.columns.astype(str),
                                sample_names=self.expr.index.astype(str))
        InferelatorData._make_idx_str(self.expr)
        pdt.assert_frame_equal(self.expr, adata._adata.to_df())

    def test_create_sparse(self):
        data = sparse.csr_matrix(self.expr.values)
        adata = InferelatorData(data, gene_names=self.expr.columns.astype(str),
                                sample_names=self.expr.index.astype(str))
        InferelatorData._make_idx_str(self.expr)
        pdt.assert_frame_equal(self.expr, adata._adata.to_df())

    def test_create_metadata(self):
        adata = InferelatorData(self.expr, meta_data=self.meta)
        pdt.assert_frame_equal(adata.meta_data, self.meta)

    def test_add_metadata(self):
        adata = InferelatorData(self.expr, meta_data=self.meta)
        self.meta.index = self.meta.index.astype(str)
        pdt.assert_frame_equal(self.meta, adata.meta_data)

    def test_add_genedata(self):
        gene_data = TestDataSingleCellLike.gene_metadata
        gene_data.index = gene_data.iloc[:, 0]

        adata = InferelatorData(self.expr, gene_data=gene_data)
        pdt.assert_index_equal(adata.gene_names, self.expr.columns)
        pdt.assert_index_equal(adata._adata.uns["trim_gene_list"], CORRECT_GENES_INTERSECT)


class TestProps(TestWrapperSetup):

    def test_gene_names(self):
        pdt.assert_index_equal(self.adata.gene_names, self.expr.columns)

    def test_sample_names(self):
        pdt.assert_index_equal(self.adata.sample_names, self.expr.index)

    def test_non_finite(self):

        adata = InferelatorData(self.expr.values.astype(float),
                                gene_names=self.expr.columns,
                                sample_names=self.expr.index)

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 0)
        self.assertIsNone(name_nf)

        adata.expression_data[0, 0] = np.nan

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 1)
        self.assertListEqual(name_nf.tolist(), ["gene1"])

        adata.expression_data[0, 1] = np.nan

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 2)
        self.assertListEqual(name_nf.tolist(), ["gene1", "gene2"])

    def test_non_finite_sparse(self):

        adata = InferelatorData(sparse.csr_matrix(self.expr.values.astype(float)),
                                gene_names=self.expr.columns,
                                sample_names=self.expr.index)

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 0)
        self.assertIsNone(name_nf)

        adata.expression_data[0, 0] = np.nan

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 1)
        self.assertListEqual(name_nf.tolist(), ["gene1"])

        adata.expression_data[0, 1] = np.nan

        nnf, name_nf = adata.non_finite
        self.assertEqual(nnf, 2)
        self.assertListEqual(name_nf.tolist(), ["gene1", "gene2"])

    def test_sample_counts(self):

        umis = np.sum(self.expr.values, axis=0)
        npt.assert_array_equal(umis, self.adata.sample_counts)
        npt.assert_array_equal(umis, self.adata_sparse.sample_counts)

    def test_gene_counts(self):

        umis = np.sum(self.expr.values, axis=1)
        npt.assert_array_equal(umis, self.adata.gene_counts)
        npt.assert_array_equal(umis, self.adata_sparse.gene_counts)


class TestTrim(TestWrapperSetup):

    def test_trim_dense(self):
        gene_data = TestDataSingleCellLike.gene_metadata
        gene_data.index = gene_data.iloc[:, 0]

        adata = InferelatorData(self.expr, gene_data=gene_data)
        adata.trim_genes(remove_constant_genes=False)

        pdt.assert_frame_equal(self.expr.reindex(CORRECT_GENES_INTERSECT, axis=1).astype(np.int32),
                               adata._adata.to_df())

        adata.trim_genes(remove_constant_genes=True)
        pdt.assert_frame_equal(self.expr.reindex(CORRECT_GENES_NZ_VAR, axis=1).astype(np.int32),
                               adata._adata.to_df())

    def test_trim_sparse(self):
        gene_data = TestDataSingleCellLike.gene_metadata
        gene_data.index = gene_data.iloc[:, 0]

        adata_sparse = InferelatorData(sparse.csr_matrix(TestDataSingleCellLike.expression_matrix.values.T),
                                       gene_names=TestDataSingleCellLike.expression_matrix.index,
                                       meta_data=TestDataSingleCellLike.meta_data.copy(),
                                       gene_data=gene_data)

        adata_sparse.trim_genes(remove_constant_genes=False)
        pdt.assert_frame_equal(self.expr.reindex(CORRECT_GENES_INTERSECT, axis=1),
                               adata_sparse._adata.to_df())

        adata_sparse.trim_genes(remove_constant_genes=True)
        pdt.assert_frame_equal(self.expr.reindex(CORRECT_GENES_NZ_VAR, axis=1),
                               adata_sparse._adata.to_df())


class TestFunctions(TestWrapperSetup):

    def setUp(self):
        super(TestFunctions, self).setUp()

        self.adata.trim_genes()
        self.adata_sparse.trim_genes()

    def test_setup(self):
        pdt.assert_frame_equal(self.adata._adata.to_df(), self.adata_sparse._adata.to_df())

    def test_transform_pseudocount(self):
        expr_vals = self.expr.loc[:, self.adata.gene_names].values

        self.adata.transform(lambda x: x, add_pseudocount=False)
        npt.assert_array_equal(self.adata.expression_data, expr_vals)

        self.adata.transform(lambda x: x - 1, add_pseudocount=True)
        npt.assert_array_equal(self.adata.expression_data, expr_vals)

    def test_transform_log2_d(self):

        self.adata.transform(np.log2, add_pseudocount=True, memory_efficient=True)
        npt.assert_array_equal(self.adata.expression_data, np.log2(self.expr.loc[:, self.adata.gene_names].values + 1))

    def test_transform_log2_d_ineff(self):

        self.adata.transform(np.log2, add_pseudocount=True, memory_efficient=False)
        npt.assert_array_equal(self.adata.expression_data, np.log2(self.expr.loc[:, self.adata.gene_names].values + 1))

    def test_transform_log2_s(self):

        self.adata_sparse.transform(np.log2, add_pseudocount=True)
        npt.assert_array_equal(self.adata_sparse.expression_data.A,
                               np.log2(self.expr.loc[:, self.adata.gene_names].values + 1))

    def test_dot_dense(self):
        inv_expr = linalg.pinv(self.adata.expression_data)
        eye_expr = np.eye(self.adata.shape[1])

        dot1 = self.adata.dot(eye_expr)
        npt.assert_array_equal(dot1, self.expr.loc[:, self.adata.gene_names].values)

        dot2 = self.adata.dot(inv_expr, other_is_right_side=False)
        npt.assert_array_almost_equal(dot2, eye_expr)

    def test_dot_sparse(self):
        inv_expr = linalg.pinv(self.adata_sparse.expression_data.A)
        eye_expr = np.eye(self.adata_sparse.shape[1])

        sdot1a = self.adata_sparse.dot(eye_expr).A
        sdot1b = self.adata_sparse.dot(sparse.csr_matrix(eye_expr)).A
        npt.assert_array_almost_equal(sdot1a, sdot1b)

        original_data = self.expr_sparse[:, TestDataSingleCellLike.expression_matrix.index.isin(CORRECT_GENES_NZ_VAR)].A
        npt.assert_array_almost_equal(self.adata_sparse.expression_data.A, original_data)
        npt.assert_array_almost_equal(sdot1b, original_data)

        sdot2a = self.adata_sparse.dot(inv_expr, other_is_right_side=False).A
        sdot2b = self.adata_sparse.dot(sparse.csr_matrix(inv_expr), other_is_right_side=False).A
        npt.assert_array_almost_equal(sdot2a, sdot2b)
        npt.assert_array_almost_equal(sdot2b, eye_expr)

    def test_dot_force_dense(self):
        inv_expr = linalg.pinv(self.adata_sparse.expression_data.A)
        eye_expr = np.eye(self.adata_sparse.shape[1])

        sdot1 = self.adata_sparse.dot(inv_expr, other_is_right_side=False, force_dense=True)
        sdot2 = self.adata.dot(inv_expr, other_is_right_side=False, force_dense=True)
        npt.assert_array_almost_equal(sdot1, sdot1)
        npt.assert_array_almost_equal(sdot2, eye_expr)

    def test_make_float32(self):

        original_data = self.expr.loc[:, TestDataSingleCellLike.expression_matrix.index.isin(CORRECT_GENES_NZ_VAR)]

        npt.assert_array_equal(original_data, self.adata.expression_data)
        self.assertTrue(self.adata.expression_data.dtype == np.int32)

        self.adata.convert_to_float()

        npt.assert_array_almost_equal(original_data, self.adata.expression_data)
        self.assertTrue(self.adata.expression_data.dtype == np.float32)

    def test_make_float64(self):

        original_data = self.expr.loc[:, TestDataSingleCellLike.expression_matrix.index.isin(CORRECT_GENES_NZ_VAR)]
        self.adata._adata.X = self.adata._adata.X.astype(np.int64)

        npt.assert_array_equal(original_data, self.adata.expression_data)
        self.assertTrue(self.adata.expression_data.dtype == np.int64)

        self.adata.convert_to_float()

        npt.assert_array_almost_equal(original_data, self.adata.expression_data)
        self.assertTrue(self.adata.expression_data.dtype == np.float64)

    def test_copy(self):

        adata2 = self.adata.copy()

        pdt.assert_frame_equal(self.adata._adata.to_df(), adata2._adata.to_df())
        pdt.assert_frame_equal(self.adata.meta_data, adata2.meta_data)
        pdt.assert_frame_equal(self.adata.gene_data, adata2.gene_data)

        adata2.expression_data[0, 0] = 100
        self.assertEqual(adata2.expression_data[0, 0], 100)
        self.assertNotEqual(self.adata.expression_data[0, 0], 100)



if __name__ == '__main__':
    unittest.main()
