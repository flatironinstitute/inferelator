import pytest
import numpy as np
import numpy.testing as npt
import warnings

from scipy import sparse, stats
from sklearn.preprocessing import RobustScaler, StandardScaler

from inferelator.tests.artifacts.test_data import (
    TestDataSingleCellLike
)

from inferelator.utils import InferelatorData, todense
from inferelator.preprocessing.data_normalization import PreprocessData


class TestNormalizationSetup:

    def setup_method(self):
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

    def teardown_method(self):
        PreprocessData.set_preprocessing_method(
            method_predictors='zscore',
            method_response='zscore',
            method_tfa='raw',
            scale_limit=None,
            scale_limit_tfa=None
        )

    def test_set_values_both(self):
        PreprocessData.set_preprocessing_method(
            'raw',
            scale_limit=None
        )

        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None

        PreprocessData.set_preprocessing_method(
            'robustscaler',
            scale_limit=10
        )

        assert PreprocessData.method_predictors == 'robustscaler'
        assert PreprocessData.method_response == 'robustscaler'
        assert PreprocessData.scale_limit_predictors == 10
        assert PreprocessData.scale_limit_response == 10

        with pytest.raises(ValueError):
            PreprocessData.set_preprocessing_method(
                'rawscaler',
                scale_limit=None
            )

    def test_set_values_predictors(self):
        PreprocessData.set_preprocessing_method(
            'raw',
            scale_limit=None
        )

        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None

        PreprocessData.set_preprocessing_method(
            scale_limit_predictors=10
        )
        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_response is None
        assert PreprocessData.scale_limit_predictors == 10

        PreprocessData.set_preprocessing_method(
            method_predictors='zscore'
        )
        assert PreprocessData.method_predictors == 'zscore'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_response is None
        assert PreprocessData.scale_limit_predictors == 10

    def test_set_values_response(self):
        PreprocessData.set_preprocessing_method(
            'raw',
            scale_limit=None
        )

        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None

        PreprocessData.set_preprocessing_method(
            scale_limit_response=10
        )
        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response == 10

        PreprocessData.set_preprocessing_method(
            method_response='zscore'
        )
        assert PreprocessData.method_response == 'zscore'
        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response == 10

    def test_set_values_tfa(self):
        PreprocessData.set_preprocessing_method(
            method='raw',
            scale_limit=None
        )

        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.method_tfa == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None
        assert PreprocessData.scale_limit_tfa is None

        PreprocessData.set_preprocessing_method(
            scale_limit_tfa=10
        )
        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.method_tfa == 'raw'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None
        assert PreprocessData.scale_limit_tfa == 10

        PreprocessData.set_preprocessing_method(
            method_tfa='zscore'
        )
        assert PreprocessData.method_predictors == 'raw'
        assert PreprocessData.method_response == 'raw'
        assert PreprocessData.method_tfa == 'zscore'
        assert PreprocessData.scale_limit_predictors is None
        assert PreprocessData.scale_limit_response is None
        assert PreprocessData.scale_limit_tfa == 10


class TestZScore(TestNormalizationSetup):

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            design_scipy = stats.zscore(self.expr, ddof=1)
        
        design_scipy[np.isnan(design_scipy)] = 0.
        npt.assert_almost_equal(
            design.values,
            design_scipy
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            design_scipy = stats.zscore(self.expr, ddof=1)

        design_scipy[np.isnan(design_scipy)] = 0.
        design_scipy[design_scipy > 1] = 1
        design_scipy[design_scipy < -1] = -1

        npt.assert_almost_equal(
            design.values,
            design_scipy
        )

    def test_response_no_limit(self):
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = stats.zscore(self.expr.iloc[:, 0], ddof=1)
        npt.assert_almost_equal(
            response,
            response_scipy
        )

    def test_response_limit(self):
        PreprocessData.set_preprocessing_method(
            'zscore',
            scale_limit=1
        )
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = stats.zscore(self.expr.iloc[:, 0], ddof=1)
        response_scipy[response_scipy > 1] = 1
        response_scipy[response_scipy < -1] = -1

        npt.assert_almost_equal(
            response,
            response_scipy
        )


class TestRobustScaler(TestNormalizationSetup):

    def setup_method(self):
        PreprocessData.set_preprocessing_method(
            'robustscaler'
        )
        super().setup_method()

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
            todense(design.values),
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
            todense(design.values),
            design_sklearn
        )

    def test_response_no_limit(self):
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = RobustScaler(with_centering=False).fit_transform(
            self.expr.iloc[:, 0].values.reshape(-1, 1)
        ).ravel()
        npt.assert_almost_equal(
            response,
            response_scipy
        )

    def test_response_limit(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = RobustScaler(with_centering=False).fit_transform(
            self.expr.iloc[:, 0].values.reshape(-1, 1)
        ).ravel()
        response_scipy[response_scipy > 1] = 1
        response_scipy[response_scipy < -1] = -1

        npt.assert_almost_equal(
            response,
            response_scipy
        )


class TestNoScaler(TestNormalizationSetup):

    def setup_method(self):
        PreprocessData.set_preprocessing_method(
            'raw'
        )
        super().setup_method()

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)
        npt.assert_almost_equal(
            design.values,
            self.expr
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)
        npt.assert_almost_equal(
            todense(design.values),
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
            todense(design.values),
            self.expr
        )

    def test_response_no_limit(self):
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        npt.assert_almost_equal(
            response,
            self.expr.iloc[:, 0].values
        )

    def test_response_limit(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        npt.assert_almost_equal(
            response,
            self.expr.iloc[:, 0].values
        )


class TestTruncRobustScaler(TestNormalizationSetup):

    def setup_method(self):
        PreprocessData.set_preprocessing_method(
            'truncrobustscaler',
            scale_limit=None
        )
        super().setup_method()

    def _right_answer(self, x):

        print(np.var(x, axis=0))

        x = RobustScaler(with_centering=False).fit_transform(x)

        ss = StandardScaler(with_mean=False).fit(x)

        print(ss.var_)
        x[:, ss.var_ > 1] = ss.transform(x)[:, ss.var_ > 1]

        return x

    def test_no_limit_d(self):
        design = PreprocessData.preprocess_design(self.adata)
        design_sklearn = self._right_answer(self.expr)
        npt.assert_almost_equal(
            design.values,
            design_sklearn
        )

    def test_no_limit_s(self):
        design = PreprocessData.preprocess_design(self.adata_sparse)
        design_sklearn = self._right_answer(self.expr)
        npt.assert_almost_equal(
            todense(design.values),
            design_sklearn
        )

    def test_limit_d(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        design = PreprocessData.preprocess_design(self.adata)
        design_sklearn = self._right_answer(self.expr)
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
        design_sklearn = self._right_answer(self.expr)
        design_sklearn[design_sklearn > 1] = 1
        design_sklearn[design_sklearn < -1] = -1

        npt.assert_almost_equal(
            todense(design.values),
            design_sklearn
        )

    def test_response_no_limit(self):
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = self._right_answer(
            self.expr.iloc[:, 0].values.reshape(-1, 1)
        ).ravel()
        npt.assert_almost_equal(
            response,
            response_scipy
        )

    def test_response_limit(self):
        PreprocessData.set_preprocessing_method(
            scale_limit=1
        )
        response = PreprocessData.preprocess_response_vector(
            self.adata.get_gene_data(["gene1"], flatten=True)
        )
        response_scipy = self._right_answer(
            self.expr.iloc[:, 0].values.reshape(-1, 1)
        ).ravel()
        response_scipy[response_scipy > 1] = 1
        response_scipy[response_scipy < -1] = -1

        npt.assert_almost_equal(
            response,
            response_scipy
        )

    def test_truncation(self):

        self.adata._adata.X[:, 0] = np.zeros_like(self.adata._adata.X[:, 0])
        self.adata._adata.X[0, 0] = 1e7

        original = self.adata._adata.X.copy()

        design_sklearn = self._right_answer(original)
        design = PreprocessData.preprocess_design(self.adata)
        npt.assert_almost_equal(
            design.values,
            design_sklearn
        )

        with pytest.raises(AssertionError):
            npt.assert_almost_equal(
                self.adata._adata.X,
                original
            )
