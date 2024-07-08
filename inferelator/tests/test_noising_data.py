import unittest
from inferelator.tests.artifacts.test_stubs import TEST_DATA
from inferelator.preprocessing import simulate_data
from inferelator import MPControl, inferelator_workflow
from inferelator.tests.artifacts.test_stubs import FakeRegressionMixin
from inferelator.utils import todense
import os
import numpy.testing as npt
from scipy import sparse as _sparse

my_dir = os.path.dirname(__file__)


class NoiseData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()

    def setUp(self):
        self.data = TEST_DATA.copy()

    def test_noise_int_data(self):
        noise_data = self.data.copy()
        simulate_data.make_data_noisy(noise_data, random_seed=100)

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(self.data.expression_data, noise_data.expression_data)

        npt.assert_array_equal(self.data.sample_counts, noise_data.sample_counts)

    def test_noise_float_data(self):
        float_data = self.data.copy()
        float_data.expression_data = float_data.expression_data.astype(float)
        noise_data = float_data.copy()
        simulate_data.make_data_noisy(noise_data, random_seed=100)

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(float_data.expression_data, noise_data.expression_data)

    def test_noise_int_data_sparse(self):
        noise_data = self.data.copy()
        noise_data._adata.X = _sparse.csr_matrix(noise_data._adata.X)
        simulate_data.make_data_noisy(noise_data, random_seed=100)

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                self.data.expression_data,
                todense(noise_data.expression_data)
            )

        self.assertTrue(noise_data.is_sparse)

        npt.assert_array_equal(self.data.sample_counts, noise_data.sample_counts)

    def test_noise_float_data_sparse(self):
        float_data = self.data.copy()
        float_data.expression_data = _sparse.csr_matrix(float_data.expression_data.astype(float))
        noise_data = float_data.copy()
        simulate_data.make_data_noisy(noise_data, random_seed=100)

        self.assertFalse(noise_data.is_sparse)

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                todense(float_data.expression_data),
                noise_data.expression_data
            )


class NoiseWorkflowData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wkf = inferelator_workflow(regression=FakeRegressionMixin, workflow="tfa")


        wkf.set_file_paths(input_dir=os.path.join(my_dir, "../../data/dream4"),
                           expression_matrix_file="expression.tsv",
                           meta_data_file="meta_data.tsv",
                           priors_file="gold_standard.tsv",
                           gold_standard_file="gold_standard.tsv")
        wkf.set_file_properties(expression_matrix_columns_are_genes=False)
        wkf.get_data()

        cls.normal_data = wkf.data

    def test_noise_tfa(self):
        wkf = inferelator_workflow(regression=FakeRegressionMixin, workflow="tfa")

        wkf.set_file_paths(input_dir=os.path.join(my_dir, "../../data/dream4"),
                           expression_matrix_file="expression.tsv",
                           meta_data_file="meta_data.tsv",
                           priors_file="gold_standard.tsv",
                           gold_standard_file="gold_standard.tsv")
        wkf.set_file_properties(expression_matrix_columns_are_genes=False)
        wkf.get_data()
        wkf.align_priors_and_expression()

        npt.assert_array_almost_equal(wkf.data.expression_data, self.normal_data.expression_data)

        wkf.set_shuffle_parameters(make_data_noise=True)
        wkf.align_priors_and_expression()

        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(wkf.data.expression_data, self.normal_data.expression_data)

    def test_noise_multitask(self):

        def _set_worker(wkf):
            wkf.set_file_paths(input_dir=os.path.join(my_dir, "../../data/dream4"),
                               gold_standard_file="gold_standard.tsv")

            task1 = wkf.create_task(input_dir=os.path.join(my_dir, "../../data/dream4"),
                                    expression_matrix_file="expression.tsv",
                                    meta_data_file="meta_data.tsv",
                                    priors_file="gold_standard.tsv")
            task1.set_file_properties(expression_matrix_columns_are_genes=False)

            task2 = wkf.create_task(input_dir=os.path.join(my_dir, "../../data/dream4"),
                                    expression_matrix_file="expression.tsv",
                                    meta_data_file="meta_data.tsv",
                                    priors_file="gold_standard.tsv")
            task2.set_file_properties(expression_matrix_columns_are_genes=False)

        wk = inferelator_workflow(regression=FakeRegressionMixin, workflow="multitask")
        _set_worker(wk)
        wk.get_data()

        for t in wk._task_objects:
            t.align_priors_and_expression()

            npt.assert_array_almost_equal(t.data.expression_data, self.normal_data.expression_data)

        wk = inferelator_workflow(regression=FakeRegressionMixin, workflow="multitask")
        wk.set_shuffle_parameters(make_data_noise=True)
        _set_worker(wk)
        wk.get_data()

        for t in wk._task_objects:
            t.align_priors_and_expression()

            with self.assertRaises(AssertionError):
                npt.assert_array_almost_equal(t.data.expression_data, self.normal_data.expression_data)


class NoiseDataMultiprocessing(NoiseData):

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("multiprocessing")
        MPControl.set_processes(1)
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()


@unittest.skip
class NoiseDataDask(NoiseData):

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("dask-local")
        MPControl.set_processes(1)
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()
