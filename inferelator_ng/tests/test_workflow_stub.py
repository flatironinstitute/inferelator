"""
Test workflow logic outline using completely
artificial stubs for dependancies.
"""

import unittest
import os
import numpy as np

from inferelator_ng import workflow
from inferelator_ng.distributed.inferelator_mp import MPControl

my_dir = os.path.dirname(__file__)


class TestWorkflowStartup(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.WorkflowBase()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")

    def tearDown(self):
        del self.workflow

    def test_load_expression(self):
        self.workflow.read_expression()
        self.assertEqual(self.workflow.expression_matrix.shape, (100, 421))
        np.testing.assert_allclose(self.workflow.expression_matrix.sum().sum(), 13507.22145160)

    def test_load_tf_names(self):
        self.workflow.read_tfs()
        self.assertEqual(len(self.workflow.tf_names), 100)
        tf_names = list(map(lambda x: "G" + str(x), list(range(1, 101))))
        self.assertListEqual(self.workflow.tf_names, tf_names)

    def test_load_metadata(self):
        self.workflow.read_metadata()
        self.assertEqual(self.workflow.meta_data.shape, (421, 5))

    def test_multiprocessing_init(self):
        MPControl.shutdown()
        self.workflow.multiprocessing_controller = "local"
        self.workflow.initialize_multiprocessing()
        self.assertTrue(MPControl.is_initialized)

    def test_abstractness(self):
        with self.assertRaises(NotImplementedError):
            self.workflow.startup()
        with self.assertRaises(NotImplementedError):
            self.workflow.startup_run()
        with self.assertRaises(NotImplementedError):
            self.workflow.startup_finish()
        with self.assertRaises(NotImplementedError):
            self.workflow.run()


class StubWorkflow(workflow.WorkflowBase):
    """
    Artificial work flow for logic testing.
    """

    cd_called = False
    test_case = None

    def __init__(self):
        pass

    def run(self):
        self.get_data()

    def compute_common_data(self):
        cd_called = True

    def get_priors(self):
        return [StubPrior(1), StubPrior(2)]

    def get_bootstraps(self):
        return [StubBootstrap("A"), StubBootstrap("B")]

    def emit_results(self, priors):
        # check that everything got initialized properly
        assert self.exp_mat is not None
        assert self.meta_data is not None
        assert self.tf_names is not None
        assert self.priors_data is not None
        assert self.gold_standard is not None
        test = self.test_case
        idents = [1, 2]
        names = ["A", "B"]
        test.assertEqual([p.ident for p in priors], idents)
        for p in priors:
            test.assertEqual(p.names, names)


class StubBootstrap(object):

    def __init__(self, name):
        self.name = name


class StubPrior(object):
    """
    Completely artificial prior object shell implementation.
    """

    def __init__(self, ident):
        self.ident = ident

    names = []

    def run_bootstrap(self, workflow, bootstrap_parameters):
        self.names = self.names + [bootstrap_parameters.name]

    bootstrap_results = None

    def combine_bootstrap_results(self, workflow, results):
        self.bootstrap_results = results


class TestWorkflowStub(unittest.TestCase):

    def test_stub(self):
        # create and configure the work flow
        work = StubWorkflow()
        work.input_dir = os.path.join(my_dir, "../../data/dream4")
        work.test_case = self
        # run the workflow (validation tests in emit_results)
        work.run()

    def test_stub_without_metadata(self):
        # create and configure the work flow
        work = StubWorkflow()
        work.input_dir = os.path.join(my_dir, "../../data/dream4_no_metadata_for_test_purposes")
        work.test_case = self
        # run the workflow (validation tests in emit_results)
        work.run()
        self.assertEqual(work.meta_data.shape, (421, 5))
        self.assertEqual(set(work.meta_data.columns.tolist()),
                         set(['condName', 'del.t', 'is1stLast', 'isTs', 'prevCol']))
