import warnings
import pytest
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import sparse

from inferelator.distributed.inferelator_mp import MPControl

from inferelator.workflows import tfa_workflow
from inferelator import workflow
from inferelator.tests.artifacts.test_data import (
    TestDataSingleCellLike,
    TEST_DATA,
    TEST_DATA_SPARSE
)
from inferelator.tests.artifacts.test_stubs import (
    TaskDataStub,
    create_puppet_workflow
)
from inferelator.utils import DotProduct
from inferelator.preprocessing.metadata_parser import MetadataHandler


"""
These are full-stack integration tests covering
the post-loading regression workflows
"""

import sys

if sys.version_info[1] > 10:
    DASK = False

    MPControl.set_multiprocess_engine('joblib')
    MPControl.set_processes(2)
    MPControl.connect()

else:
    DASK = True

    MPControl.set_multiprocess_engine('dask-local')
    MPControl.set_processes(2)
    MPControl.connect()

LASSO_TEST_DATA = TEST_DATA.copy()
RNG = np.random.default_rng(200)
LASSO_TEST_DATA.add(
    np.abs(
        np.hstack(
            list(
                RNG.standard_normal(
                    size=(LASSO_TEST_DATA.values.shape[0], 1)
                ) * np.std(LASSO_TEST_DATA.values[:, i])
                for i in range(LASSO_TEST_DATA.shape[1])
            )
        )
    )
)


class SetUpDenseData:

    def setup_method(self):
        sample_names = TestDataSingleCellLike.expression_matrix.columns
        meta_data = MetadataHandler.get_handler('branching').create_default_meta_data(sample_names)

        self.data = TEST_DATA.copy()
        self.data.meta_data = meta_data

        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names


class SetUpLassoData(SetUpDenseData):

    def setup_method(self):
        super().setup_method()
        self.data = LASSO_TEST_DATA.copy()


class SetUpSparseData:

    def setup_method(self):
        sample_names = TestDataSingleCellLike.expression_matrix.columns
        meta_data = MetadataHandler.get_handler('branching').create_default_meta_data(sample_names)

        self.data = TEST_DATA_SPARSE.copy()
        self.data.meta_data = meta_data

        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names


class SetUpSparseLassoData(SetUpLassoData):

    def setup_method(self):
        super().setup_method()
        self.data._adata.X = sparse.csr_matrix(self.data._adata.X)


class SetUpDenseDataMTL(SetUpDenseData):

    def setup_method(self):
        super().setup_method()
        self._task_objects = [TaskDataStub(), TaskDataStub()]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False


class SetUpDenseLassoDataMTL(SetUpLassoData):

    def setup_method(self):
        super().setup_method()
        self._task_objects = [
            TaskDataStub(),
            TaskDataStub()
        ]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False
        self._task_objects[0].data = self.data.copy()
        self._task_objects[1].data = self.data.copy()


class SetUpSparseDataMTL(SetUpSparseData):

    def setup_method(self):
        super().setup_method()
        self._task_objects = [
            TaskDataStub(sparse=True),
            TaskDataStub(sparse=True)
        ]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False


class SetUpSparseLassoDataMTL(SetUpSparseLassoData):

    def setup_method(self):
        super().setup_method()
        self._task_objects = [TaskDataStub(), TaskDataStub()]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False
        self._task_objects[0].data = self.data.copy()
        self._task_objects[1].data = self.data.copy()


class TestSingleTaskRegressionFactory(SetUpDenseData):

    def test_base(self):
        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow)
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names
        with pytest.raises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names
        self.workflow.run()

        assert self.workflow.results.score == 1

    def test_bbsr_clr_only(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.set_regression_parameters(clr_only=True)
        self.workflow.tf_names = self.tf_names
        self.workflow.run()
        assert self.workflow.results.score == 1

    def test_elasticnet(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="elasticnet")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        assert self.workflow.results.score == 1

    def test_sklearn(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="sklearn")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names

        with pytest.raises(ValueError):
            self.workflow.set_regression_parameters(model=42)

        with pytest.raises(ValueError):
            self.workflow.set_regression_parameters(model=LinearRegression())

        self.workflow.set_regression_parameters(model=LinearRegression)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        assert self.workflow.results.score == 1


class TestSingleTaskNoPriors(SetUpDenseData):

    def test_bbsr(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, None, None)
        self.workflow.tf_names = self.tf_names

        with pytest.warns(UserWarning):
            self.workflow.set_network_data_flags(
                use_no_prior=True,
                use_no_gold_standard=True
            )
        self.workflow.validate_data()
        self.workflow.run()

        assert np.isnan(self.workflow.results.score)


class TestSingleTaskStabilityRegressionFactory(SetUpLassoData):

    @pytest.mark.skip(reason="Unknown regression with new numpy or sklearn")
    def test_stars(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="stars")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)

        with pytest.warns(UserWarning):
            self.workflow.set_regression_parameters(num_subsamples=2)

        self.workflow.tf_names = self.tf_names

        self.workflow.run()

        # Not enough data / variance to get the right result
        assert abs(self.workflow.results.score - 0.657777) < 5e-5

    def test_stars_ridge(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="stars")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)

        with pytest.warns(UserWarning):
            self.workflow.set_regression_parameters(method='ridge', ridge_threshold=1e-3, num_subsamples=2)

        self.workflow.tf_names = self.tf_names

        self.workflow.run()

        assert abs(self.workflow.results.score - 1) < 5e-5


class TestSingleTaskRegressionFactorySparse(SetUpSparseData, TestSingleTaskRegressionFactory):

    @classmethod
    def setup_class(cls):
        DotProduct.set_mkl(True)

    @classmethod
    def teardown_class(cls):
        DotProduct.set_mkl(False)


class TestSingleTaskStabilityRegressionFactory(SetUpSparseLassoData, TestSingleTaskStabilityRegressionFactory):

    @classmethod
    def setup_class(cls):
        DotProduct.set_mkl(True)

    @classmethod
    def teardown_class(cls):
        DotProduct.set_mkl(False)


class TestMultitaskFactory(SetUpDenseLassoDataMTL):

    def reset_workflow(self):
        self.workflow.priors_data = self.prior
        self.workflow.gold_standard = self.gold_standard
        self.workflow._task_objects = self._task_objects
        self.workflow.read_priors = lambda *x: None
        self.workflow.create_output_dir = lambda *x: None
        self.workflow.tf_names = self.tf_names

    def test_amusr(self):
        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression="amusr")
        self.reset_workflow()

        self.workflow.run()
        assert abs(self.workflow.results.score - 1) < 5e-5

    def test_mtl_bbsr(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="bbsr")

        with pytest.warns(Warning):
            self.workflow.set_regression_parameters(prior_weight=2.)
            self.workflow.set_regression_parameters(prior_weight=1.)

        self.reset_workflow()

        self.workflow.run()
        assert self.workflow.results.score == 1

    def test_mtl_elasticnet(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="elasticnet")
        self.workflow.set_regression_parameters(copy_X=True)
        self.reset_workflow()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        assert self.workflow.results.score == 1

    @pytest.mark.skip(reason="Unknown regression with new numpy or sklearn")
    def test_mtl_stars_lasso(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="stars")

        with pytest.warns(UserWarning):
            self.workflow.set_regression_parameters(num_subsamples=2)

        self.reset_workflow()

        self.workflow.run()
        assert abs(self.workflow.results.score - 0.657777) < 5e-5

    def test_mtl_stars_ridge(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="stars")

        with pytest.warns(UserWarning):
            self.workflow.set_regression_parameters(method='ridge', ridge_threshold=1e-3, num_subsamples=2)

        self.reset_workflow()

        self.workflow.run()
        assert abs(self.workflow.results.score - 1) < 5e-5


class TestMultitaskFactorySparse(SetUpSparseLassoDataMTL, TestMultitaskFactory):
    pass
