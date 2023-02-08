import unittest
import numpy as np
import numpy.testing as npt

from scipy import sparse, stats
from sklearn.preprocessing import RobustScaler

from inferelator.tests.artifacts.test_data import (
    TestDataSingleCellLike
)

from inferelator.utils import InferelatorData
from inferelator.preprocessing.data_normalization import PreprocessData


class TestNormalizationSetup(unittest.TestCase):

    def setUp(self):
        self.expr = TestDataSingleCellLike.expression_matrix.copy().T
        self.expr_sparse = sparse.csr_matrix(
            TestDataSingleCellLike.expression_matrix.values.T
        ).astype(np.int32)
        self.meta = TestDataSingleCellLike.meta_data.copy()

        self.adata = InferelatorData(
            self.expr,
            transpose_expression=False,
            meta_data=self.meta.copy()
        )
        self.adata_sparse = InferelatorData(
            self.expr_sparse,
            gene_names=TestDataSingleCellLike.expression_matrix.index,
            transpose_expression=False,
            meta_data=self.meta.copy()
        )

    def tearDown(self):
        PreprocessData.set_preprocessing_method(
            'zscore',
            scale_limit=None
        )


class TestZScore(TestNormalizationSetup):

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)
        design_scipy = stats.zscore(self.expr, ddof=1)
        design_scipy[np.isnan(design_scipy)] = 0.
        npt.assert_almost_equal(
            design.values,
            design_scipy
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)
        design_scipy = stats.zscore(self.expr, ddof=1)
        design_scipy[np.isnan(design_scipy)] = 0.
        npt.assert_almost_equal(
            design.values,
            design_scipy
        )

    def test_limit_d(self):
        PreprocessData.set_preprocessing_method(
            'zscore',
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata)
        design_scipy = stats.zscore(self.expr, ddof=1)
        design_scipy[np.isnan(design_scipy)] = 0.
        design_scipy[design_scipy > 1] = 1
        design_scipy[design_scipy < -1] = -1

        npt.assert_almost_equal(
            design.values,
            design_scipy
        )

    def test_limit_s(self):
        PreprocessData.set_preprocessing_method(
            'zscore',
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata_sparse)
        design_scipy = stats.zscore(self.expr, ddof=1)
        design_scipy[np.isnan(design_scipy)] = 0.
        design_scipy[design_scipy > 1] = 1
        design_scipy[design_scipy < -1] = -1

        npt.assert_almost_equal(
            design.values,
            design_scipy
        )


class TestRobustScaler(TestNormalizationSetup):

    def setUp(self):
        PreprocessData.set_preprocessing_method(
            'robustscaler'
        )
        return super().setUp()

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)
        design_sklearn = RobustScaler(with_centering=False).fit_transform(self.expr)
        npt.assert_almost_equal(
            design.values,
            design_sklearn
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)
        design_sklearn = RobustScaler(with_centering=False).fit_transform(self.expr)
        npt.assert_almost_equal(
            design.values.A,
            design_sklearn
        )

    def test_limit_d(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata)
        design_sklearn = RobustScaler(with_centering=False).fit_transform(self.expr)
        design_sklearn[design_sklearn > 1] = 1
        design_sklearn[design_sklearn < -1] = -1

        npt.assert_almost_equal(
            design.values,
            design_sklearn
        )

    def test_limit_s(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata_sparse)
        design_sklearn = RobustScaler(with_centering=False).fit_transform(self.expr)
        design_sklearn[design_sklearn > 1] = 1
        design_sklearn[design_sklearn < -1] = -1

        npt.assert_almost_equal(
            design.values.A,
            design_sklearn
        )


class TestNoScaler(TestNormalizationSetup):

    def setUp(self):
        PreprocessData.set_preprocessing_method(
            'raw'
        )
        return super().setUp()

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)
        npt.assert_almost_equal(
            design.values,
            self.expr
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)
        npt.assert_almost_equal(
            design.values.A,
            self.expr
        )

    def test_limit_d(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata)
        npt.assert_almost_equal(
            design.values,
            self.expr
        )

    def test_limit_s(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata_sparse)
        npt.assert_almost_equal(
            design.values.A,
            self.expr
        )
