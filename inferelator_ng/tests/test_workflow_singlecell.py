"""
Test base workflow stepwise.
"""

import unittest
import os
import numpy as np
import pandas as pd

from inferelator_ng.single_cell_workflow import SingleCellWorkflow
from inferelator_ng.preprocessing import single_cell

my_dir = os.path.dirname(__file__)


def prep1(expr, meta, **kwargs):
    return expr, meta


def prep2(expr, meta, **kwargs):
    return expr, meta


test_count_data = pd.DataFrame([[0, 0, 0], [10, 0, 10], [4, 0, 5], [0, 0, 0]])
test_meta_data = SingleCellWorkflow.create_default_meta_data(test_count_data)


class TestSingleCellWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = SingleCellWorkflow()
        self.workflow.expression_matrix_columns_are_genes = True
        self.workflow.input_dir = os.path.join(my_dir, "../../data/dream4")

    def tearDown(self):
        del self.workflow

    def test_preprocessing_flow(self):
        self.workflow.expression_matrix_columns_are_genes = False
        self.workflow.get_data()
        self.workflow.add_preprocess_step(prep1)
        self.workflow.add_preprocess_step(prep2)
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.expression_matrix.shape, (100, 421))

    def test_preprocessing_filter(self):
        self.workflow.expression_matrix = test_count_data
        self.workflow.meta_data = test_meta_data
        self.workflow.single_cell_normalize()
        self.assertEqual(self.workflow.expression_matrix.shape, (4, 2))

    def test_preprocessing_nan_pre(self):
        self.workflow.expression_matrix = test_count_data
        self.workflow.expression_matrix.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.workflow.single_cell_normalize()

    def test_preprocessing_nan_post(self):
        self.workflow.expression_matrix = test_count_data - 1
        self.workflow.add_preprocess_step(single_cell.log2_data)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            with self.assertRaises(ValueError):
                self.workflow.single_cell_normalize()
