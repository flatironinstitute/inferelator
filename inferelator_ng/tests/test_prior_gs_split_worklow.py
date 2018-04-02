"""
Test workflow logic outline using completely
artificial stubs for dependancies.
"""

import unittest
from .. import prior_gs_split_workflow

import os
import numpy as np

my_dir = os.path.dirname(__file__)

class StubWorkflow(prior_gs_split_workflow.PriorGoldStandardSplitWorkflowBase):

    """
    Artificial work flow for logic testing.
    """

    test_case = None

    def run(self):
        self.get_data()

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

class TestPriorGoldStandardSplitWorkflowBaseStub(unittest.TestCase):

    def test_gs_and_prior_same_size(self):
        # create and configure the work flow
        work = StubWorkflow()
        work.input_dir = os.path.join(my_dir, "../../data/dream4")
        work.test_case = self
        # run the workflow (validation tests in emit_results)
        work.run()
        self.assertEqual(np.sum(work.priors_data.sum()), np.sum(work.gold_standard.sum()))

    def test_prior_half_size(self):
        # create and configure the work flow
        work = StubWorkflow()
        work.input_dir = os.path.join(my_dir, "../../data/dream4_no_metadata_for_test_purposes")
        work.test_case = self
        # run the workflow (validation tests in emit_results)
        work.run()
        original_priors = work.input_dataframe(work.priors_file)
        self.assertEqual(np.sum(work.priors_data.sum()), np.sum(original_priors.sum() / 2))

