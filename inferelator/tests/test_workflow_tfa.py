"""
Test TFA workflow stepwise.
"""

import os
import types
import unittest
import tempfile

import numpy as np
import pandas as pd

from inferelator.workflows import tfa_workflow
from inferelator import workflow
from inferelator.tests.artifacts.test_stubs import FakeResultProcessor, FakeRegressionMixin, FakeDRD
from inferelator.preprocessing import tfa
from inferelator.preprocessing import design_response_translation as drt

my_dir = os.path.dirname(__file__)

DEFAULT_EXPRESSION_FILE = "expression.tsv"
DEFAULT_TFNAMES_FILE = "tf_names.tsv"
DEFAULT_METADATA_FILE = "meta_data.tsv"
DEFAULT_PRIORS_FILE = "gold_standard.tsv"
DEFAULT_GOLDSTANDARD_FILE = "gold_standard.tsv"


class TestTFASetup(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow._factory_build_inferelator(regression=FakeRegressionMixin,
                                                            workflow=tfa_workflow.TFAWorkFlow)()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.expression_matrix_file = DEFAULT_EXPRESSION_FILE
        self.workflow.tf_names_file = DEFAULT_TFNAMES_FILE
        self.workflow.meta_data_file = DEFAULT_METADATA_FILE
        self.workflow.priors_file = DEFAULT_PRIORS_FILE
        self.workflow.gold_standard_file = DEFAULT_GOLDSTANDARD_FILE
        self.workflow.get_data()

    def tearDown(self):
        del self.workflow

class TestAbstract(unittest.TestCase):

    def test_abstractness(self):
        self.workflow = workflow._factory_build_inferelator(regression='base',
                                                            workflow=tfa_workflow.TFAWorkFlow)()
        with self.assertRaises(NotImplementedError):
            self.workflow.run_bootstrap([])

class TestTFAWorkflow(TestTFASetup):

    def test_compute_common_data(self):
        self.workflow.drd_driver = FakeDRD
        self.workflow.compute_common_data()
        self.assertTrue(self.workflow.data is None)
        np.testing.assert_array_almost_equal_nulp(self.workflow.design.expression_data,
                                                  self.workflow.response.expression_data)

    def test_compute_activity(self):
        self.workflow.drd_driver = FakeDRD
        self.workflow.tfa_driver = tfa.NoTFA
        self.workflow.compute_common_data()
        self.workflow.compute_activity()

    def test_set_tf_params(self):

        self.workflow.set_tfa(tfa_driver=False)
        self.assertIs(self.workflow.tfa_driver, tfa.NoTFA)

        self.workflow.set_tfa(tfa_driver=True)
        self.assertIs(self.workflow.tfa_driver, tfa.TFA)

        self.workflow.set_tfa(tfa_output_file="test.tsv")
        self.assertEqual(self.workflow._tfa_output_file, "test.tsv")

    def test_set_drd_params(self):

        self.workflow.set_design_settings(timecourse_response_driver=False)
        self.assertIsNone(self.workflow.drd_driver)

        self.workflow.set_design_settings(timecourse_response_driver=True)
        self.assertIs(self.workflow.drd_driver, drt.PythonDRDriver)


class TestTFAWorkflowRegression(TestTFASetup):

    def test_regression(self):
        self.workflow.startup_run()
        self.workflow.drd_driver = FakeDRD
        self.workflow.tfa_driver = tfa.NoTFA
        self.workflow.compute_common_data()
        self.workflow.compute_activity()
        self.assertTrue(self.workflow.run_bootstrap([]))

    def test_result_processor(self):
        self.workflow._result_processor_driver = FakeResultProcessor

        def no_output(self):
            pass

        self.workflow.create_output_dir = types.MethodType(no_output, self.workflow)
        self.workflow.emit_results(None, None, None, None)


class TestTFAWrite(TestTFASetup):

    def test_tfa_tsv(self):
        try:
            self.workflow.output_dir = tempfile.gettempdir()
            self.workflow.set_tfa(tfa_output_file="test.tsv")
            self.tfa_file_name = os.path.join(tempfile.gettempdir(), "test.tsv")

            self.workflow.startup()

            self.assertTrue(os.path.exists(self.tfa_file_name))
            tfa = pd.read_csv(self.tfa_file_name, sep="\t", index_col=0)

            self.assertTupleEqual(self.workflow.design.shape, tfa.shape)
            pd.testing.assert_frame_equal(tfa, self.workflow.design.to_df())
        finally:
            os.remove(self.tfa_file_name)



