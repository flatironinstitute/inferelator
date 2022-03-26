import warnings
import unittest
import tempfile
import pandas as pd
import shutil
import numpy as np
from sklearn.linear_model import LinearRegression

from inferelator.distributed.inferelator_mp import MPControl

from inferelator.workflows import tfa_workflow
from inferelator import workflow
from inferelator.tests.artifacts.test_data import TestDataSingleCellLike, TEST_DATA, TEST_DATA_SPARSE
from inferelator.tests.artifacts.test_stubs import TaskDataStub, create_puppet_workflow
from inferelator.utils import InferelatorData, DotProduct
from inferelator.preprocessing.metadata_parser import MetadataHandler


from dask import distributed
from inferelator.distributed import dask_local_controller
from inferelator.distributed import dask_functions

"""
These are full-stack integration tests covering the post-loading regression workflows
"""


class SetUpDenseData(unittest.TestCase):

    def setUp(self):
        sample_names = TestDataSingleCellLike.expression_matrix.columns
        meta_data = MetadataHandler.get_handler('branching').create_default_meta_data(sample_names)

        self.data = TEST_DATA.copy()
        self.data.meta_data = meta_data

        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names


class SetUpSparseData(unittest.TestCase):

    def setUp(self):
        sample_names = TestDataSingleCellLike.expression_matrix.columns
        meta_data = MetadataHandler.get_handler('branching').create_default_meta_data(sample_names)

        self.data = TEST_DATA_SPARSE.copy()
        self.data.meta_data = meta_data

        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names


class SetUpDenseDataMTL(SetUpDenseData):

    def setUp(self):
        super(SetUpDenseDataMTL, self).setUp()
        self._task_objects = [TaskDataStub(), TaskDataStub()]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False


class SetUpSparseDataMTL(SetUpSparseData):

    def setUp(self):
        super(SetUpSparseDataMTL, self).setUp()
        self._task_objects = [TaskDataStub(sparse=True), TaskDataStub(sparse=True)]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False


class TestSingleTaskRegressionFactory(SetUpDenseData):

    @classmethod
    def setUpClass(cls):
        if not MPControl.is_initialized:
            MPControl.set_multiprocess_engine("local")
            MPControl.connect()

    def test_base(self):
        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow)
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names
        with self.assertRaises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names
        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_bbsr_clr_only(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.set_regression_parameters(clr_only=True)
        self.workflow.tf_names = self.tf_names
        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_elasticnet(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="elasticnet")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()
            
        self.assertEqual(self.workflow.results.score, 1)

    def test_sklearn(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="sklearn")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.tf_names = self.tf_names

        with self.assertRaises(ValueError):
            self.workflow.set_regression_parameters(model=42)

        with self.assertRaises(ValueError):
            self.workflow.set_regression_parameters(model=LinearRegression())

        self.workflow.set_regression_parameters(model=LinearRegression)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        self.assertEqual(self.workflow.results.score, 1)

    def test_stars(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="stars")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.set_regression_parameters(num_subsamples=5)
        self.workflow.tf_names = self.tf_names

        self.workflow.run()
        self.assertAlmostEqual(self.workflow.results.score, 0.32222, places=4)


class TestSingleTaskRegressionFactorySparse(SetUpSparseData, TestSingleTaskRegressionFactory):

    @classmethod
    def setUpClass(cls):

        if not MPControl.is_initialized:
            MPControl.set_multiprocess_engine("local")
            MPControl.connect()

        DotProduct.set_mkl(True)

    @classmethod
    def tearDownClass(cls):
        DotProduct.set_mkl(False)


class TestMultitaskFactory(SetUpDenseDataMTL):

    @classmethod
    def setUpClass(cls):
        if not MPControl.is_initialized:
            MPControl.connect()

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
        self.assertAlmostEqual(self.workflow.results.score, 0.84166, places=4)

    def test_mtl_bbsr(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="bbsr")

        with self.assertWarns(Warning):
            self.workflow.set_regression_parameters(prior_weight=1.)

        self.reset_workflow()

        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_mtl_elasticnet(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="elasticnet")
        self.workflow.set_regression_parameters(copy_X=True)
        self.reset_workflow()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        self.assertEqual(self.workflow.results.score, 1)

    def test_mtl_stars(self):
        self.workflow = workflow.inferelator_workflow(workflow="multitask", regression="stars")
        self.workflow.set_regression_parameters(num_subsamples=5)
        self.reset_workflow()

        self.workflow.run()
        self.assertAlmostEqual(self.workflow.results.score, 0.32222, places=4)


class TestMultitaskFactorySparse(SetUpSparseDataMTL, TestMultitaskFactory):
    pass


class SwitchToDask(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("dask-local")
        MPControl.connect(local_dir=cls.tempdir, n_workers=1, processes=False)

    @classmethod
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)


class TestSTLDask(SwitchToDask, TestSingleTaskRegressionFactory):

    def test_dask_function_mi(self):
        """Compute mi for identical arrays [[1, 2, 1], [2, 4, 6]]."""

        L = [[0, 0], [9, 3], [0, 9]]
        x_dataframe = InferelatorData(pd.DataFrame(L))
        y_dataframe = InferelatorData(pd.DataFrame(L))
        mi = dask_functions.build_mi_array_dask(x_dataframe.values, y_dataframe.values, 10, np.log)
        expected = np.array([[0.63651417, 0.63651417], [0.63651417, 1.09861229]])
        np.testing.assert_almost_equal(mi, expected)


class TestSTLSparseDask(SwitchToDask, TestSingleTaskRegressionFactorySparse):
    pass


class TestMTLDask(SwitchToDask, TestMultitaskFactory):
    pass


class TestMTLSparseDask(SwitchToDask, TestMultitaskFactorySparse):
    pass
