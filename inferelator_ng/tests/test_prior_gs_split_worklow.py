"""
Test workflow logic outline using completely
artificial stubs for dependancies.
"""

import unittest
from inferelator_ng import prior_gs_split_workflow
from inferelator_ng import workflow

import os
import numpy as np

my_dir = os.path.dirname(__file__)

class StubWorkflow(prior_gs_split_workflow.PriorGoldStandardSplitWorkflowBase, workflow.WorkflowBase):

    """
    Artificial work flow for logic testing.
    """

    test_case = None

    def __init__(self, axis=None):
        self.axis = axis

    def run(self):
        self.get_data()

    def set_gold_standard_and_priors(self, gold_standard_split=prior_gs_split_workflow.DEFAULT_SPLIT):
        super(StubWorkflow, self).set_gold_standard_and_priors(gold_standard_split, self.axis)
        print(self.priors_data.shape)
        print(self.gold_standard.shape)

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

    def test_gs_and_prior_same_size_split_on_gene(self):
        # create and configure the work flow
        work = StubWorkflow(axis=0)
        work.input_dir = os.path.join(my_dir, "../../data/dream4")
        work.test_case = self
        # run the workflow (validation tests in emit_results)
        work.run()
        self.assertEqual(work.priors_data.shape, work.gold_standard.shape)

