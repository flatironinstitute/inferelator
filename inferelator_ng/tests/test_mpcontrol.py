import unittest
import tempfile
import os
import shutil
from inferelator_ng.distributed.inferelator_mp import MPControl

try:
    from kvsstcp import kvsstcp
    from inferelator_ng.distributed import kvs_controller

    TEST_KVS = True
except ImportError:
    TEST_KVS = False

try:
    import dask.distributed as distributed
    from inferelator_ng.distributed import dask_cluster_controller, dask_local_controller

    TEST_DASK = True
except ImportError:
    TEST_DASK = False

try:
    import pathos
    from inferelator_ng.distributed import multiprocessing_controller

    TEST_PATHOS = True
except ImportError:
    TEST_PATHOS = False


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


class TestLocalController(TestMPControl):
    name = "local"

    def test_local_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_local_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_local_sync(self):
        self.assertTrue(MPControl.sync_processes())

    def test_local_name(self):
        self.assertEqual(MPControl.name(), self.name)


@unittest.skipIf(not TEST_KVS, "KVS not installed")
class TestKVSMPController(TestMPControl):
    server = None
    name = "kvs"

    @classmethod
    def setUpClass(cls):
        cls.server = kvsstcp.KVSServer("", 0)
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(host=cls.server.cinfo[0], port=cls.server.cinfo[1])

    @classmethod
    def tearDownClass(cls):
        super(TestKVSMPController, cls).tearDownClass()
        if cls.server is not None:
            cls.server.shutdown()

    def test_kvs_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_kvs_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_kvs_sync(self):
        MPControl.sync_processes()

    def test_kvs_name(self):
        self.assertEqual(MPControl.name(), self.name)


@unittest.skipIf(not TEST_PATHOS, "Pathos not installed")
class TestMultiprocessingMPController(TestMPControl):
    name = "multiprocessing"

    def test_mp_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_mp_name(self):
        self.assertEqual(MPControl.name(), self.name)

    def test_mp_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_mp_sync(self):
        self.assertTrue(MPControl.sync_processes())


@unittest.skipIf(not TEST_DASK, "Dask not installed")
class TestDaskLocalMPController(TestMPControl):
    name = "dask-local"
    client_name = "dask"
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
        self.assertTrue(MPControl.is_initialized)

    def test_dask_local_name(self):
        self.assertEqual(MPControl.name(), self.client_name)

    def test_dask_local_map(self):
        with self.assertRaises(NotImplementedError):
            MPControl.map(math_function, *self.map_test_data)

    def test_dask_local_sync(self):
        self.assertTrue(MPControl.sync_processes())

    def test_dask_local_workers_running(self):
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, "global.lock")))
