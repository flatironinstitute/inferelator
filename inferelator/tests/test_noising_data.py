import unittest
from inferelator.tests.artifacts.test_stubs import TEST_DATA
from inferelator.preprocessing import simulate_data
from inferelator import MPControl
import os
import numpy.testing as npt

try:
    from dask import distributed
    from inferelator.distributed import dask_local_controller

    TEST_DASK_LOCAL = True
except ImportError:
    TEST_DASK_LOCAL = False

my_dir = os.path.dirname(__file__)


class NoiseIntData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if MPControl.is_initialized:
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
            npt.assert_array_almost_equal(self.data.expression_data, noise_data.expression_data)


class NoiseIntDataMultiprocessing(NoiseIntData):

    @classmethod
    def setUpClass(cls):
        if MPControl.is_initialized:
            MPControl.shutdown()

        MPControl.set_multiprocess_engine("multiprocessing")
        MPControl.set_processes(1)
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        if MPControl.is_initialized:
            MPControl.shutdown()

        MPControl.set_multiprocess_engine("local")
        MPControl.connect()


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
class NoiseIntDataDask(NoiseIntData):

    @classmethod
    def setUpClass(cls):
        if MPControl.is_initialized:
            MPControl.shutdown()

        MPControl.set_multiprocess_engine("dask-local")
        MPControl.set_processes(1)
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        if MPControl.is_initialized:
            MPControl.shutdown()

        MPControl.set_multiprocess_engine("local")
        MPControl.connect()