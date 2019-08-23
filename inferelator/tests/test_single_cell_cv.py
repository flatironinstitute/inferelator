from __future__ import division

import unittest
import os
import pandas as pd

from inferelator import default
from inferelator import single_cell_cv_workflow
from inferelator.tests.artifacts.test_stubs import FakeRegression, FakeWriter, FakeResultProcessor

my_dir = os.path.dirname(__file__)

class TestSingleCellBase(unittest.TestCase):

    def setUp(self):
        self.workflow = single_cell_cv_workflow.SingleCellPuppeteerWorkflow()

    def test_abstractness(self):
        with self.assertRaises(NotImplementedError):
            self.workflow.modeling_method()


class TestSizeSampling(unittest.TestCase):

    def setUp(self):
        self.workflow = single_cell_cv_workflow.SingleCellSizeSampling()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.expression_matrix_file = default.DEFAULT_EXPRESSION_FILE
        self.workflow.tf_names_file = default.DEFAULT_TFNAMES_FILE
        self.workflow.meta_data_file = default.DEFAULT_METADATA_FILE
        self.workflow.priors_file = default.DEFAULT_PRIORS_FILE
        self.workflow.gold_standard_file = default.DEFAULT_GOLDSTANDARD_FILE
        self.workflow.get_data()
        self.workflow.meta_data['condition'] = pd.Series(["A"] * 100 + ["B"] * 321)
        self.workflow.stratified_batch_lookup = 'condition'
        self.workflow.csv_writer = FakeWriter()

    def test_cv_sampling(self):
        self.assertEqual(len(self.workflow.get_sample_index(sample_size=50)), 50)
        self.assertEqual(len(self.workflow.get_sample_index(sample_ratio=0.5)), 210)
        stratified_sample = self.workflow.get_sample_index(sample_size=50, stratified_sampling=True)
        self.assertEqual(sum(stratified_sample >= 100), 50)
        self.assertEqual(sum(stratified_sample < 100), 50)

    def test_cv_modeling(self):
        self.workflow.cv_regression_type = FakeRegression
        self.workflow.cv_result_processor_type = FakeResultProcessor
        self.workflow.sizes = list(map(lambda x: x / 10, range(1, 3)))
        self.workflow.seeds = list(range(42, 44))
        auprs = self.workflow.modeling_method()
        self.assertEqual(len(auprs), 4)

class TestDropoutSampling(unittest.TestCase):

    def setUp(self):
        self.workflow = single_cell_cv_workflow.SingleCellDropoutConditionSampling()
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.expression_matrix_file = default.DEFAULT_EXPRESSION_FILE
        self.workflow.tf_names_file = default.DEFAULT_TFNAMES_FILE
        self.workflow.meta_data_file = default.DEFAULT_METADATA_FILE
        self.workflow.priors_file = default.DEFAULT_PRIORS_FILE
        self.workflow.gold_standard_file = default.DEFAULT_GOLDSTANDARD_FILE
        self.workflow.get_data()
        self.workflow.meta_data['condition'] = pd.Series(["A"] * 100 + ["B"] * 321)
        self.workflow.meta_data.index = self.workflow.expression_matrix.columns
        self.workflow.stratified_batch_lookup = 'condition'
        self.workflow.csv_writer = FakeWriter()

    def test_dropout_sampling(self):
        self.workflow.drop_column = 'condition'
        self.workflow.sample_batches_to_size = 10
        self.workflow.cv_regression_type = FakeRegression
        self.workflow.cv_result_processor_type = FakeResultProcessor
        self.workflow.seeds = list(range(42, 44))
        auprs = self.workflow.modeling_method()
        self.assertEqual(len(auprs), 12)
