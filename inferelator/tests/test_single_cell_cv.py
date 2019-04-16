from __future__ import division

import unittest
import os
import numpy as np
import pandas as pd

from inferelator import single_cell_cv_workflow
from inferelator.regression.base_regression import RegressionWorkflow

my_dir = os.path.dirname(__file__)


class FakeWriter(object):
    def writerow(self, *args, **kwargs):
        pass


class FakeRegression(RegressionWorkflow):

    def run_regression(self):
        beta = [pd.DataFrame(np.array([[0, 1], [0.5, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        beta_resc = [pd.DataFrame(np.array([[0, 1], [1, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        return beta, beta_resc

    def run_bootstrap(self, bootstrap):
        return True


class FakeResultProcessor:

    def __init__(self, *args, **kwargs):
        pass

    def summarize_network(self, *args, **kwargs):
        return 1, 0, 0


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
