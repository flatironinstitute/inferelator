import unittest
import pandas as pd
import types
import numpy as np
import tempfile

from inferelator import crossvalidation_workflow
from inferelator.workflow import WorkflowBase

fake_metadata = pd.DataFrame({"CONST": ["A"] * 1000,
                              "VAR": ["A"] * 100 + ["B"] * 200 + ["C"] * 1 + ["D"] * 99 + ["E"] * 500 + ["F"] * 100})


class FakeResult(object):

    score=1
    name="NAME"


class FakeWorkflow(WorkflowBase):

    meta_data = fake_metadata.copy()
    seed = 10

    def run(self):
        return FakeResult()

    def get_data(self):
        return "GotData"


def fake_class_method(slf):
    pass


class FakeWriter(object):

    csv_lil = None

    def __init__(self, *args, **kwargs):
        pass

    def writerow(self, line, **kwargs):
        if self.csv_lil is None:
            self.csv_lil = []
        self.csv_lil.append(line)


class TestCVSampleIndexing(unittest.TestCase):

    def setUp(self):
        wkf = FakeWorkflow()
        wkf.output_dir = tempfile.gettempdir()
        self.cv = crossvalidation_workflow.CrossValidationManager(wkf)
        self.cv._csv_writer_object = FakeWriter
        self.cv._create_csv_handle = types.MethodType(fake_class_method, self.cv)
        self.cv._create_output_path = types.MethodType(fake_class_method, self.cv)

    def test_dropin_set(self):
        self.assertIsNone(self.cv.dropin_column)

        self.cv.add_grouping_dropin("VAR", group_size=100, seed=50)

        self.assertEqual(self.cv.dropin_column, "VAR")
        self.assertEqual(self.cv.dropin_max_size, 100)
        self.assertEqual(self.cv.dropin_seed, 50)

    def test_dropout_set(self):
        self.assertIsNone(self.cv.dropout_column)

        self.cv.add_grouping_dropout("VAR", group_size=100, seed=50)

        self.assertEqual(self.cv.dropout_column, "VAR")
        self.assertEqual(self.cv.dropout_max_size, 100)
        self.assertEqual(self.cv.dropout_seed, 50)

    def test_size_sample_set(self):
        self.assertIsNone(self.cv.size_sample_vector)

        self.cv.add_size_subsampling([0.1, 0.2, 1], stratified_column_name="VAR", seed=50)

        self.assertListEqual(self.cv.size_sample_vector, [0.1, 0.2, 1])
        self.assertEqual(self.cv.size_sample_stratified_column, "VAR")
        self.assertEqual(self.cv.size_sample_seed, 50)

    def test_add_grid_param(self):
        self.assertIsNone(self.cv.grid_params)
        self.assertIsNone(self.cv.grid_param_values)

        self.cv.add_gridsearch_parameter("seed", [1, 2, 3])
        self.cv.add_gridsearch_parameter("test", [3, 4, 5])

        self.assertListEqual(self.cv.grid_params, ["seed", "test"])
        self.assertListEqual(self.cv.grid_param_values['seed'], [1, 2, 3])
        self.assertListEqual(self.cv.grid_param_values['test'], [3, 4, 5])

    def test_load_initial(self):

        self.assertEqual(self.cv.workflow.get_data(), "GotData")
        self.cv._initial_data_load()
        self.assertIsNone(self.cv.workflow.get_data())

    def test_get_copy(self):

        copied_work = self.cv._get_workflow_copy()
        copied_work.seed = 50

        self.assertEqual(self.cv.workflow.seed, 10)
        self.assertEqual(copied_work.seed, 50)

    def test_csv(self):
        self.cv.add_gridsearch_parameter("seed", [1, 2, 3])
        self.cv.add_gridsearch_parameter("test", [3, 4, 5])

        self.assertIsNone(self.cv.csv_header)
        self.cv._create_writer()
        self.assertListEqual(self.cv.csv_header, ["seed", "test", "Test", "Value", "Num_Obs", "aupr"])

    def test_validate_params(self):
        self.cv.add_gridsearch_parameter("seed", [1, 2, 3])
        self.cv._check_grid_search_params_exist()

        self.cv.add_gridsearch_parameter("test", [3, 4, 5])
        with self.assertRaises(ValueError):
            self.cv._check_grid_search_params_exist()

    def test_validate_meta_cols(self):
        self.cv.dropin_column = "VAR"
        self.cv.dropout_column = "CONST"

        self.cv._check_metadata()

        self.cv.size_sample_stratified_column = "NOTREAL"
        with self.assertRaises(ValueError):
            self.cv._check_metadata()

    def test_group_index_masker(self):

        self.assertEqual(crossvalidation_workflow.group_index(fake_metadata, "CONST", "A").sum(), 1000)
        self.assertEqual(crossvalidation_workflow.group_index(fake_metadata, "CONST", "B").sum(), 0)
        self.assertEqual(crossvalidation_workflow.group_index(fake_metadata, "CONST", "A", max_size=100).sum(), 100)
        self.assertEqual(crossvalidation_workflow.group_index(fake_metadata, "CONST", "A", size_ratio=0.5).sum(), 500)

        rgen = np.random.RandomState(10)
        idx_1 = crossvalidation_workflow.group_index(fake_metadata, "CONST", "A", size_ratio=0.1, rgen=rgen)
        idx_2 = crossvalidation_workflow.group_index(fake_metadata, "CONST", "A", size_ratio=0.1, rgen=rgen)

        self.assertEqual(idx_1.sum(), 100)
        self.assertEqual(idx_2.sum(), 100)
        self.assertEqual((idx_1 & idx_2).sum(), 8)

    def test_grid_search(self):
        self.cv.add_gridsearch_parameter("seed", [1, 2, 3])
        self.cv._create_writer()
        self.cv._grid_search()

        self.assertEqual(len(self.cv.csv_writer.csv_lil), 4)