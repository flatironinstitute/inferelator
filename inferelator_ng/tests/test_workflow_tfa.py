"""
Test TFA workflow stepwise.
"""

import unittest
import os
import types
import numpy as np

from inferelator_ng import workflow
from inferelator_ng import tfa_workflow
from inferelator_ng.preprocessing import tfa
from inferelator_ng.regression.base_regression import RegressionWorkflow

my_dir = os.path.dirname(__file__)


class FakeDRD:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, expr, meta):
        return expr, expr, expr


class FakeRegression(RegressionWorkflow):

    def run_bootstrap(self, bootstrap):
        return True


class FakeResultProcessor:

    def __init__(self, *args, **kwargs):
        pass

    def summarize_network(self, *args, **kwargs):
        pass


class TestTFAWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.create_inferelator_workflow(regression=None,
                                                             workflow=tfa_workflow.TFAWorkFlow)()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.get_data()

    def tearDown(self):
        del self.workflow

    def test_compute_common_data(self):
        self.workflow.drd_driver = FakeDRD
        self.workflow.compute_common_data()
        self.assertTrue(self.workflow.expression_matrix is None)
        np.testing.assert_array_almost_equal_nulp(self.workflow.design.values, self.workflow.response.values)

    def test_compute_activity(self):
        self.workflow.drd_driver = FakeDRD
        self.workflow.tfa_driver = tfa.NoTFA
        self.workflow.compute_common_data()
        self.workflow.compute_activity()

    def test_abstractness(self):
        with self.assertRaises(NotImplementedError):
            self.workflow.run_bootstrap([])

class TestTFAWorkflowRegression(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.create_inferelator_workflow(regression=FakeRegression,
                                                             workflow=tfa_workflow.TFAWorkFlow)()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.get_data()

    def tearDown(self):
        del self.workflow

    def test_regression(self):
        self.workflow.regression_type = FakeRegression
        self.workflow.startup_run()
        self.workflow.drd_driver = FakeDRD
        self.workflow.tfa_driver = tfa.NoTFA
        self.workflow.compute_common_data()
        self.workflow.compute_activity()
        self.assertTrue(self.workflow.run_bootstrap([]))

    def test_result_processor(self):
        self.workflow.result_processor_driver = FakeResultProcessor

        def no_output(self):
            pass

        self.workflow.create_output_dir = types.MethodType(no_output, self.workflow)
        self.workflow.emit_results(None, None, None, None)
