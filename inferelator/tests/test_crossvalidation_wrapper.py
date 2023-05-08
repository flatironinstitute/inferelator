import unittest
import pandas as pd
import types
import numpy as np
import tempfile
import os

from inferelator import crossvalidation_workflow
from inferelator.workflow import WorkflowBase
from inferelator.utils.inferelator_data import InferelatorData

from numpy.random import default_rng

fake_obsnames = list(map(str, range(1000)))

fake_metadata = pd.DataFrame({"CONST": ["A"] * 1000,
                              "VAR": ["A"] * 100 + ["B"] * 200 + ["C"] * 1 + ["D"] * 99 + ["E"] * 500 + ["F"] * 100},
                              index = fake_obsnames)

fake_data_object = InferelatorData(default_rng(12345).random(size=1000).reshape((1000,1)), meta_data=fake_metadata, sample_names=fake_obsnames)

TEMP_DIR = tempfile.gettempdir()
TEMP_DIR_1 = os.path.join(TEMP_DIR, "test1")


class FakeResult(object):

    score = 1
    name = "NAME"

    all_names = ["NAME"]
    all_scores = {"NAME": 1}


class FakeWorkflow(WorkflowBase):

    seed = 10

    def __init__(self):
        self.data = fake_data_object.copy()

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

    def close(self):
        pass


class TestCV(unittest.TestCase):

    def setUp(self):
        wkf = FakeWorkflow()
        wkf.output_dir = TEMP_DIR
        self.cv = crossvalidation_workflow.CrossValidationManager(wkf)
        self.cv._csv_writer_object = FakeWriter
        self.cv._open_csv_handle = types.MethodType(fake_class_method, self.cv)
        self.cv._create_output_path = types.MethodType(fake_class_method, self.cv)


class TestCVSetup(TestCV):

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
        self.cv.workflow.metric = "aupr"

        self.assertIsNone(self.cv._csv_header)
        self.cv._create_writer()
        self.assertListEqual(self.cv._csv_header, ["seed", "test", "Test", "Value", "Num_Obs", "AUPR"])

    def test_csv_combined(self):
        self.cv.add_gridsearch_parameter("seed", [1, 2, 3])
        self.cv.add_gridsearch_parameter("test", [3, 4, 5])
        self.cv.metric = "combined"

        self.assertIsNone(self.cv._csv_header)
        self.cv._create_writer()
        self.assertListEqual(self.cv._csv_header, ["seed", "test", "Test", "Value", "Num_Obs", "AUPR", "F1", "MCC"])

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


class TestCVProperties(TestCV):

    def test_output_dir_cv(self):
        self.assertEqual(TEMP_DIR, self.cv.output_dir)
        self.cv.append_to_path('output_dir', 'test1')
        self.assertEqual(TEMP_DIR_1, self.cv.output_dir)

    def test_set_output_dir_cv(self):
        self.cv.output_dir = TEMP_DIR_1
        self.assertEqual(TEMP_DIR_1, self.cv.workflow.output_dir)

    def test_input_dir_cv(self):
        self.cv.workflow.input_dir = TEMP_DIR
        self.assertEqual(TEMP_DIR, self.cv.input_dir)

    def test_set_input_dir_cv(self):
        self.cv.input_dir = TEMP_DIR_1
        self.assertEqual(TEMP_DIR_1, self.cv.workflow.input_dir)

    def test_harmonize(self):
        self.cv.workflow.output_dir = None
        self.cv.workflow.input_dir = None

        self.cv._baseline_input_dir = TEMP_DIR
        self.cv._baseline_output_dir = TEMP_DIR_1

        self.cv._harmonize_paths()

        self.assertEqual(TEMP_DIR_1, self.cv.workflow.output_dir)
        self.assertEqual(TEMP_DIR, self.cv.workflow.input_dir)


class TestCVSampleIndexing(TestCV):

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

        self.assertEqual(len(self.cv._csv_writer.csv_lil), 4)

    def test_group_dropout_no_limit(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "dropout")
            self.assertTrue(value in slf.workflow.data.meta_data[slf.dropout_column].unique())
            self.assertListEqual((slf.workflow.data.meta_data[slf.dropout_column] != value).tolist(),
                                 mask_function().tolist())

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.dropout_column = "VAR"
        self.cv.dropout_max_size = None
        self.cv.dropout_seed = 50

        self.cv._dropout_cv()

    def test_group_dropout_limit(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "dropout")
            uniques = slf.workflow.data.meta_data[slf.dropout_column].unique()

            mask = mask_function()
            unique_counts = slf.workflow.data.meta_data[slf.dropout_column].value_counts()
            unique_counts[unique_counts > slf.dropout_max_size] = slf.dropout_max_size

            if value == "all":
                self.assertEqual(unique_counts.sum(), mask.sum())
            else:
                self.assertTrue(value in uniques)
                unique_counts[value] = 0
                self.assertEqual(unique_counts.sum(), mask.sum())
                self.assertEqual(sum((self.cv.workflow.data.meta_data[self.cv.dropout_column] == value)[mask]), 0)

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.dropout_column = "VAR"
        self.cv.dropout_max_size = 50
        self.cv.dropout_seed = 50

        self.cv._dropout_cv()

    def test_group_dropin_no_limit(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "dropin")
            self.assertTrue(value in slf.workflow.data.meta_data[slf.dropin_column].unique())
            self.assertListEqual((slf.workflow.data.meta_data[slf.dropin_column] == value).tolist(),
                                 mask_function().tolist())

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.dropin_column = "VAR"
        self.cv.dropin_max_size = None
        self.cv.dropin_seed = 50

        self.cv._dropin_cv()

    def test_group_dropin_limit(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "dropin")

            mask = mask_function()

            if value == "all":
                self.assertEqual(mask.sum(), slf.dropin_max_size)
            else:
                self.assertTrue(value in slf.workflow.data.meta_data[slf.dropin_column].unique())

                self.assertEqual(min((slf.workflow.data.meta_data[slf.dropin_column] == value).sum(),
                                     slf.dropin_max_size),
                                 mask.sum())

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.dropin_column = "VAR"
        self.cv.dropin_max_size = 25
        self.cv.dropin_seed = 50

        self.cv._dropin_cv()

    def test_size_sampling_no_strat(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "size")
            self.assertTrue(value == "0.5")

            self.assertEqual(max(int(slf.workflow.data.meta_data.shape[0] * float(value)), 1),
                             mask_function().sum())

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.size_sample_vector = [0.5]
        self.cv.size_sample_seed = 50

        self.cv._size_cv()

    def test_size_sampling_strat(self):

        def test_grid_search(slf, test=None, value=None, mask_function=None):
            self.assertEqual(test, "size")
            self.assertTrue(value == "0.5")

            mask = mask_function()
            for g in slf.workflow.data.meta_data[slf.size_sample_stratified_column].unique():
                is_group = slf.workflow.data.meta_data[slf.size_sample_stratified_column] == g
                self.assertEqual(max(int(is_group.sum() * float(value)), 1),
                                 mask[is_group].sum())

        self.cv._grid_search = types.MethodType(test_grid_search, self.cv)

        self.cv.size_sample_vector = [0.5]
        self.cv.size_sample_seed = 50
        self.cv.size_sample_stratified_column = "VAR"

        self.cv._size_cv()
