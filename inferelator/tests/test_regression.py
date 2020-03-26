import warnings
import unittest
import tempfile
import pandas as pd
import os
import shutil
import numpy as np

from inferelator.distributed.inferelator_mp import MPControl

from inferelator import tfa_workflow
from inferelator import workflow
from inferelator.tests.artifacts.test_data import TestDataSingleCellLike
from inferelator.tests.artifacts.test_stubs import TaskDataStub, create_puppet_workflow
from inferelator.regression.bbsr_multitask import BBSRByTaskRegressionWorkflow
from inferelator.regression.elasticnet_multitask import ElasticNetByTaskRegressionWorkflow
from inferelator.utils import InferelatorData
from inferelator.preprocessing.metadata_parser import MetadataHandler

try:
    from dask import distributed
    from inferelator.distributed import dask_local_controller
    from inferelator.distributed import dask_functions

    TEST_DASK_LOCAL = True
except ImportError:
    TEST_DASK_LOCAL = False


class TestRegressionFactory(unittest.TestCase):

    def setUp(self):
        meta_data = MetadataHandler.get_handler('branching')\
            .create_default_meta_data(TestDataSingleCellLike.expression_matrix.columns)
        self.data = InferelatorData(TestDataSingleCellLike.expression_matrix.T,
                                    meta_data=meta_data,
                                    gene_data=TestDataSingleCellLike.gene_metadata,
                                    gene_data_idx_column="SystematicName")
        self.prior = TestDataSingleCellLike.priors_data
        self.gold_standard = self.prior.copy()
        self.tf_names = TestDataSingleCellLike.tf_names


class TestSingleTaskRegressionFactory(TestRegressionFactory):

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


class TestMultitaskFactory(TestRegressionFactory):

    def setUp(self):
        super(TestMultitaskFactory, self).setUp()
        self._task_objects = [TaskDataStub(), TaskDataStub()]
        self._task_objects[0].tasks_from_metadata = False
        self._task_objects[1].tasks_from_metadata = False

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
        self.assertAlmostEqual(self.workflow.results.score, 0.85, places=4)

    def test_mtl_bbsr(self):
        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression=BBSRByTaskRegressionWorkflow)
        self.reset_workflow()

        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_mtl_elasticnet(self):
        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression=ElasticNetByTaskRegressionWorkflow)
        self.reset_workflow()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.workflow.run()

        self.assertEqual(self.workflow.results.score, 1)


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
class SwitchToDask(unittest.TestCase):
    tempdir = None

    @classmethod
    @unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("dask-local")
        MPControl.connect(local_dir=cls.tempdir, n_workers=1, processes=False)

    @classmethod
    @unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
class TestSTLDask(TestSingleTaskRegressionFactory, SwitchToDask):

    def test_dask_function_mi(self):
        """Compute mi for identical arrays [[1, 2, 1], [2, 4, 6]]."""

        L = [[0, 0], [9, 3], [0, 9]]
        x_dataframe = InferelatorData(pd.DataFrame(L))
        y_dataframe = InferelatorData(pd.DataFrame(L))
        mi = dask_functions.build_mi_array_dask(x_dataframe.values, y_dataframe.values, 10, np.log)
        expected = np.array([[0.63651417, 0.63651417], [0.63651417, 1.09861229]])
        np.testing.assert_almost_equal(mi, expected)


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
class TestMTLDask(TestMultitaskFactory, SwitchToDask):
    pass
