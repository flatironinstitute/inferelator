import unittest
import tempfile
import shutil
import os

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.distributed import dask_cluster_controller


def math_function(x, y, z):
    return x + y ** 2 - z


class TestMPControl(unittest.TestCase):
    name = "local"
    map_test_data = [[1] * 3, list(range(3)), [0, 2, 4]]
    map_test_expect = [1, 0, 1]

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()


class TestNoController(TestMPControl):
    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.client = None

    @classmethod
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()

    def test_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_bad_engine(self):
        with self.assertRaises(ValueError):
            MPControl.set_multiprocess_engine("V8")

    def test_bad_engine_II(self):
        with self.assertRaises(ValueError):
            MPControl.set_multiprocess_engine(unittest.TestCase)


class TestLocalController(TestMPControl):
    name = "local"

    def test_local_connect(self):
        self.assertTrue(MPControl.status())

    def test_can_change(self):
        MPControl.set_multiprocess_engine("local")

    def test_local_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_local_name(self):
        self.assertEqual(MPControl.name(), self.name)

class TestMultiprocessingMPController(TestMPControl):
    name = "joblib"

    @classmethod
    def setUpClass(cls):
        super(TestMultiprocessingMPController, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestMultiprocessingMPController, cls).tearDownClass()

    def test_mp_connect(self):
        self.assertTrue(MPControl.status())

    def test_mp_name(self):
        self.assertEqual(MPControl.name(), self.name)

    def test_mp_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)


class TestDaskLocalMPController(TestMPControl):
    name = "dask-local"
    client_name = "dask-local"
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(local_dir=cls.tempdir, n_workers=1)

    @classmethod
    def tearDownClass(cls):
        super(TestDaskLocalMPController, cls).tearDownClass()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)

    def test_dask_local_connect(self):
        self.assertTrue(MPControl.status())

    def test_dask_local_name(self):
        self.assertEqual(MPControl.name(), self.client_name)

    def test_dask_local_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)


class TestDaskHPCMPController(TestMPControl):
    name = "dask-cluster"
    client_name = "dask-cluster"
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.client.use_default_configuration('greene', 0)
        MPControl.client.set_job_size_params(n_jobs=0, n_threads_per_worker=1)
        MPControl.client.set_cluster_params(local_workers=2)
        MPControl.client.add_worker_conda()
        MPControl.client._interface = None
        MPControl.client._log_directory = cls.tempdir
        MPControl.connect()
        MPControl.client._scale_jobs()

    @classmethod
    def tearDownClass(cls):
        super(TestDaskHPCMPController, cls).tearDownClass()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)

    def test_dask_cluster_connect(self):
        self.assertTrue(MPControl.status())

    def test_dask_cluster_name(self):
        self.assertEqual(MPControl.name(), self.client_name)

    def test_bad_default_config(self):
        with self.assertRaises(ValueError):
            MPControl.client.use_default_configuration("no")

    @unittest.skipIf('CI' in os.environ, "workers are weird for this on CI")
    def test_dask_cluster_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_memory_0_hack(self):
        old_command = "dask-worker tcp://scheduler:port --memory-limit=4e9 --nthreads 1 --nprocs 20"
        new_command = "dask-worker tcp://scheduler:port --memory-limit 0 --nthreads 1 --nprocs 20"
        self.assertEqual(new_command, dask_cluster_controller.memory_limit_0(old_command))
        old_command_2 = "dask-worker tcp://scheduler:port --nthreads 1 --nprocs 20 --memory-limit=4e9"
        new_command_2 = "dask-worker tcp://scheduler:port --nthreads 1 --nprocs 20 --memory-limit 0 "
        self.assertEqual(new_command_2, dask_cluster_controller.memory_limit_0(old_command_2))
