import unittest
import warnings

from inferelator import tfa_workflow
from inferelator import workflow
from inferelator.tests.artifacts.test_data import TestDataSingleCellLike
from inferelator.tests.artifacts.test_stubs import TaskDataStub, create_puppet_workflow
from inferelator.regression.bbsr_multitask import BBSRByTaskRegressionWorkflow
from inferelator.regression.elasticnet_multitask import ElasticNetByTaskRegressionWorkflow
from inferelator.utils import InferelatorData
from inferelator.preprocessing.metadata_parser import MetadataHandler

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
        self.gene_list = TestDataSingleCellLike.gene_metadata
        self.tf_names = TestDataSingleCellLike.tf_names


class TestSingleTaskRegressionFactory(TestRegressionFactory):

    def test_base(self):
        self.workflow = create_puppet_workflow(base_class=tfa_workflow.TFAWorkFlow)
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        with self.assertRaises(NotImplementedError):
            self.workflow.run()

    def test_bbsr(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_bbsr_clr_only(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="bbsr")
        self.workflow = self.workflow(self.expr, self.meta, self.prior, self.gold_standard)
        self.workflow.set_regression_parameters(clr_only=True)
        self.workflow.gene_list = self.gene_list
        self.workflow.tf_names = self.tf_names
        self.workflow.meta_data_file = None
        self.workflow.read_metadata()
        self.workflow.run()
        self.assertEqual(self.workflow.results.score, 1)

    def test_elasticnet(self):
        self.workflow = create_puppet_workflow(base_class="tfa", regression_class="elasticnet")
        self.workflow = self.workflow(self.data, self.prior, self.gold_standard)
        self.workflow.gene_list = self.gene_list
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
        self.workflow.gene_list = self.gene_list
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
