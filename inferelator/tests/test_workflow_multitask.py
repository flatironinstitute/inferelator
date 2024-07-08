import unittest
import os
import numpy as np
import pandas as pd
import pandas.testing as pdt
import copy

from inferelator import workflow
from inferelator.workflows.amusr_workflow import MultitaskLearningWorkflow
from inferelator.tests.artifacts.test_stubs import TaskDataStub

data_path = os.path.join(os.path.dirname(__file__), "../../data/dream4")

class TestAMuSRWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = workflow.inferelator_workflow(workflow="amusr", regression="amusr")
        self.workflow.create_output_dir = lambda *x: None
        self.workflow.gold_standard = TaskDataStub.priors_data.copy()

    def test_set_task_filters(self):
        self.assertEqual(self.workflow._regulator_expression_filter, "intersection")
        self.assertEqual(self.workflow._target_expression_filter, "union")

        self.workflow.set_task_filters(regulator_expression_filter="test1",
                                       target_expression_filter="test2")

        self.assertEqual(self.workflow._regulator_expression_filter, "test1")
        self.assertEqual(self.workflow._target_expression_filter, "test2")

    def test_create_task(self):
        self.assertIsNone(self.workflow._task_objects)

        with self.assertRaises(ValueError):
            self.workflow._load_tasks()

        self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                  meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                  priors_file="gold_standard.tsv", drd_driver=None)

        self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                    meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                    priors_file="gold_standard.tsv", gold_standard_file="gold_standard.tsv")

        with self.assertRaises(ValueError):
            self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                      meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                      priors_file="gold_standard.tsv", does_not_exist=True)

        self.assertEqual(len(self.workflow._task_objects), 2)

        new_task = self.workflow._task_objects[0]

        self.assertEqual(new_task.expression_matrix_file, "expression.tsv")
        self.assertIsNone(new_task.data)

        self.assertEqual(new_task.meta_data_file, "meta_data.tsv")
        self.assertIsNone(new_task.data)

        self.assertEqual(new_task.priors_file, "gold_standard.tsv")
        self.assertIsNone(new_task.priors_data)

        self.assertEqual(new_task.tf_names_file, "tf_names.tsv")
        self.assertIsNone(new_task.tf_names)

    def test_num_props(self):
        self.assertIsNone(self.workflow._num_obs)
        self.assertIsNone(self.workflow._num_tfs)
        self.assertIsNone(self.workflow._num_genes)

        task_obs = TaskDataStub().data.num_obs
        task_tfs = TaskDataStub().priors_data.shape[1]
        task_genes = TaskDataStub().data.num_genes

        self.workflow._task_objects = [TaskDataStub(), TaskDataStub(), TaskDataStub()]

        self.assertEqual(self.workflow._num_obs, task_obs * 3)
        self.assertEqual(self.workflow._num_genes, task_genes)
        self.assertEqual(self.workflow._num_tfs, task_tfs)

    def test_name_props(self):
        self.assertIsNone(self.workflow._gene_names)
        self.assertIsNone(self.workflow._tf_names)

        task_tfs = TaskDataStub().priors_data.columns.tolist()
        task_genes = TaskDataStub().data.gene_names.tolist()

        self.workflow._task_objects = [TaskDataStub(), TaskDataStub(), TaskDataStub()]

        self.assertListEqual(self.workflow._gene_names.tolist(), task_genes)
        self.assertListEqual(self.workflow._tf_names.tolist(), task_tfs)

        self.workflow.startup()

        self.assertListEqual(self.workflow._gene_names.tolist(), ["gene1", "gene2", "gene4", "gene6"])
        self.assertListEqual(self.workflow._tf_names.tolist(), task_tfs)        

    def test_align_parent_priors(self):
        self.workflow._process_default_priors()

        self.assertIsNone(self.workflow.priors_data)
        self.assertTupleEqual(self.workflow.gold_standard.shape, TaskDataStub.priors_data.shape)

        self.workflow.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
        self.workflow._process_default_priors()

        self.assertIsNone(self.workflow.priors_data)
        self.assertEqual(self.workflow.gold_standard.shape[1], TaskDataStub.priors_data.shape[1])
        self.assertEqual(max(int(TaskDataStub.priors_data.shape[0] * 0.2), 1), self.workflow.gold_standard.shape[0])

    def test_align_task_priors(self):
        self.workflow.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
        self.workflow._process_default_priors()

        self.workflow._task_objects = [copy.deepcopy(TaskDataStub()),
                                       copy.deepcopy(TaskDataStub()),
                                       copy.deepcopy(TaskDataStub())]

        self.assertTupleEqual(self.workflow._task_objects[0].priors_data.shape, TaskDataStub.priors_data.shape)

        self.workflow._process_task_priors()

        expect_size = 2
        prior_sizes = list(map(lambda x: x.priors_data.shape[0], self.workflow._task_objects))

        self.assertListEqual([expect_size] * 3, prior_sizes)

    def test_taskdata_loading(self):
        self.assertIsNone(self.workflow._task_objects)
        task1 = self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                          meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                          priors_file="gold_standard.tsv")
        task1.set_file_properties(expression_matrix_columns_are_genes=False)

        self.assertEqual(len(self.workflow._task_objects), 1)
        task2 = self.workflow.create_task(expression_matrix_file="expression.tsv", input_dir=None,
                                          meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                          priors_file="gold_standard.tsv")
        task2.set_file_properties(expression_matrix_columns_are_genes=False)

        self.assertEqual(len(self.workflow._task_objects), 2)
        self.workflow.input_dir = data_path
        self.workflow._load_tasks()

        self.assertEqual(task1.data.shape, (421, 100))
        np.testing.assert_allclose(np.sum(task1.data.expression_data), 13507.22145160)
        self.assertEqual(len(task1.tf_names), 100)
        self.assertListEqual(task1.tf_names, list(map(lambda x: "G" + str(x), list(range(1, 101)))))
        self.assertEqual(task1.priors_data.shape, (100, 100))
        self.assertEqual(task1.data.meta_data.shape, (421, 5))

        self.assertEqual(task1.input_dir, task2.input_dir)

    def test_taskdata_processing(self):
        self.workflow._task_objects = [TaskDataStub()]
        # Test the TaskData processing
        self.assertEqual(len(self.workflow._task_objects), 1)
        self.workflow._load_tasks()
        self.assertEqual(len(self.workflow._task_objects), 3)

        # Test processing the TaskData objects into data structures in MultitaskLearningWorkflow
        self.assertEqual(self.workflow._n_tasks, 3)
        self.assertEqual(list(map(lambda x: x.data.shape, self.workflow._task_objects)),
                         [(2, 6), (4, 6), (4, 6)])
        self.assertEqual(list(map(lambda x: x.data.meta_data.shape, self.workflow._task_objects)),
                         [(2, 2), (4, 2), (4, 2)])

    def test_task_processing(self):
        self.workflow._task_objects = [TaskDataStub()]
        self.workflow._load_tasks()
        self.workflow.startup_finish()
        self.assertEqual(self.workflow._regulators.tolist(), ["gene3", "gene6"])
        self.assertEqual(self.workflow._targets.tolist(), ["gene1", "gene2", "gene4", "gene6"])
        self.assertEqual(len(self.workflow._task_design), 3)
        self.assertEqual(len(self.workflow._task_response), 3)
        self.assertEqual(len(self.workflow._task_bootstraps), 3)
        pdt.assert_frame_equal(self.workflow._task_design[0].to_df(),
                               pd.DataFrame([[16., 5.], [15., 15.]], index=["gene3", "gene6"], columns=["0", "6"]).T,
                               check_dtype=False)
        pdt.assert_frame_equal(self.workflow._task_response[0].to_df(),
                               pd.DataFrame([[2, 3], [28, 27], [16, 5], [3, 4]],
                                            index=["gene1", "gene2", "gene4", "gene6"], columns=["0", "6"]).T,
                               check_dtype=False)

    @unittest.skip('Changes in numpy2, dunno why')
    def test_result_processor_random(self):
        self.workflow._task_objects = [TaskDataStub()]
        self.workflow._load_tasks()

        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0], [0, 1], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta3 = pd.DataFrame(np.array([[0.5, 0.2], [0.5, 0.1], [0.5, 0.2], [0.5, 0.2]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb1 = pd.DataFrame(np.array([[0.75, 0], [0.25, 0], [0.75, 0], [0.25, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb2 = pd.DataFrame(np.array([[0, 0], [1, 0], [0, 0], [1, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb3 = pd.DataFrame(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])

        self.workflow.startup_finish()
        self.workflow.gold_standard_filter_method = 'overlap'
        self.workflow.emit_results([[beta1, beta1], [beta2, beta2], [beta3, beta3]],
                                   [[rb1, rb1], [rb2, rb2], [rb3, rb3]],
                                   self.workflow.gold_standard,
                                   self.workflow.priors_data)
        self.assertAlmostEqual(self.workflow.results.score, 0.37777, places=4)

    def test_result_processor_perfect(self):
        self.workflow._task_objects = [TaskDataStub()]
        self.workflow._load_tasks()

        beta1 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta2 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        beta3 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0.2]]),
                             ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb1 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.25, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb2 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])
        rb3 = pd.DataFrame(np.array([[0, 1], [0, 1], [1, 0], [0.5, 0.5]]),
                           ["gene1", "gene2", "gene4", "gene6"], ["gene3", "gene6"])

        self.workflow.startup_finish()
        self.workflow.emit_results([[beta1, beta1], [beta2, beta2], [beta3, beta3]],
                                   [[rb1, rb1], [rb2, rb2], [rb3, rb3]],
                                   self.workflow.gold_standard,
                                   self.workflow.priors_data)
        self.assertAlmostEqual(self.workflow.results.score, 1)

class TestWorkflowFunctions(unittest.TestCase):
    data = None

    @classmethod
    def setUpClass(cls):

        cls.data = MultitaskLearningWorkflow()

        cls.data.set_file_paths(input_dir=data_path, gold_standard_file='gold_standard.tsv',
                                tf_names_file="tf_names.tsv")

        task1 = cls.data.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                     meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                     priors_file="gold_standard.tsv")

        task2 = cls.data.create_task(expression_matrix_file="expression.tsv", input_dir=data_path,
                                     meta_data_file="meta_data.tsv", tf_names_file="tf_names.tsv",
                                     priors_file="gold_standard.tsv")

        task1.set_file_properties(expression_matrix_columns_are_genes=False)
        task2.set_file_properties(expression_matrix_columns_are_genes=False)

        cls.data.get_data()


    def setUp(self):
        self.workflow = MultitaskLearningWorkflow()
        self.workflow._task_objects = copy.deepcopy(self.data._task_objects)
        self.workflow.gold_standard = self.data.gold_standard.copy()
        self.workflow.tf_names = copy.deepcopy(self.data.tf_names)


    def test_shuffle_priors_genes(self):

        self.workflow.shuffle_prior_axis = 0

        for tobj in self.workflow._task_objects:
            np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)
    
        self.workflow._process_default_priors()
        self.workflow._process_task_priors()

        for tobj in self.workflow._task_objects:

            self.assertTrue(all(tobj.priors_data.columns == self.workflow.gold_standard.columns))
            self.assertTrue(all(tobj.priors_data.index == self.workflow.gold_standard.index))
            self.assertTrue(all(tobj.priors_data.sum(axis=0) == self.workflow.gold_standard.sum(axis=0)))
            
            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

    def test_noise_prior(self):
        self.workflow.add_prior_noise = 0.5

        for tobj in self.workflow._task_objects:
            np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

        self.workflow._process_default_priors()
        self.workflow._process_task_priors()

        for tobj in self.workflow._task_objects:

            self.assertTrue(all(tobj.priors_data.columns == self.workflow.gold_standard.columns))
            self.assertTrue(all(tobj.priors_data.index == self.workflow.gold_standard.index))

            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

        np.testing.assert_array_almost_equal_nulp(self.workflow._task_objects[0].priors_data.values,
                                                  self.workflow._task_objects[1].priors_data.values)

    def test_noise_prior_different(self):
        self.workflow.add_prior_noise = 0.5

        for tobj in self.workflow._task_objects:
            np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

        self.workflow._task_objects[0].random_seed = 1000

        self.workflow._process_default_priors()
        self.workflow._process_task_priors()

        for tobj in self.workflow._task_objects:

            self.assertTrue(all(tobj.priors_data.columns == self.workflow.gold_standard.columns))
            self.assertTrue(all(tobj.priors_data.index == self.workflow.gold_standard.index))

            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

                
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal_nulp(self.workflow._task_objects[0].priors_data.values,
                                                     self.workflow._task_objects[1].priors_data.values)

    def test_noise_prior_base(self):
        self.workflow.add_prior_noise = 0.5
        self.workflow.add_prior_noise_to_task_priors = False

        self.workflow._task_objects[0].random_seed = 1000

        for tobj in self.workflow._task_objects:
            np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

        self.workflow._process_default_priors()
        self.workflow._process_task_priors()

        for tobj in self.workflow._task_objects:

            self.assertTrue(all(tobj.priors_data.columns == self.workflow.gold_standard.columns))
            self.assertTrue(all(tobj.priors_data.index == self.workflow.gold_standard.index))

            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal_nulp(tobj.priors_data.values, self.workflow.gold_standard.values)

        np.testing.assert_array_almost_equal_nulp(self.workflow._task_objects[0].priors_data.values,
                                                  self.workflow._task_objects[1].priors_data.values)