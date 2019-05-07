import unittest
import tempfile
import shutil
import types
from inferelator.distributed.inferelator_mp import MPControl

# Run tests only when the associated packages are installed
try:
    from kvsstcp import kvsstcp
    from inferelator.distributed import kvs_controller

    TEST_KVS = True
except ImportError:
    TEST_KVS = False

try:
    from dask import distributed
    from inferelator.distributed import dask_local_controller

    TEST_DASK_LOCAL = True
except ImportError:
    TEST_DASK_LOCAL = False

try:
    from dask import distributed
    import dask_jobqueue
    from inferelator.distributed import dask_cluster_controller

    TEST_DASK_CLUSTER = True
except ImportError:
    TEST_DASK_CLUSTER = False

try:
    import pathos
    from inferelator.distributed import multiprocessing_controller

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
        with self.assertRaises(RuntimeError):
            MPControl.map(math_function, *self.map_test_data)

    def test_sync(self):
        with self.assertRaises(RuntimeError):
            MPControl.sync_processes()

    def test_name(self):
        with self.assertRaises(NameError):
            MPControl.name()

    def test_bad_engine(self):
        with self.assertRaises(ValueError):
            MPControl.set_multiprocess_engine("V8")

    def test_bad_engine_II(self):
        with self.assertRaises(ValueError):
            MPControl.set_multiprocess_engine(unittest.TestCase)


class TestLocalController(TestMPControl):
    name = "local"

    def test_local_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_cant_change(self):
        with self.assertRaises(RuntimeError):
            MPControl.set_multiprocess_engine("local")

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
    temp_dir = None

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.server = kvsstcp.KVSServer("", 0)
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(host=cls.server.cinfo[0], port=cls.server.cinfo[1])

    @classmethod
    def tearDownClass(cls):
        super(TestKVSMPController, cls).tearDownClass()
        if cls.server is not None:
            cls.server.shutdown()
        if cls.temp_dir is not None:
            shutil.rmtree(cls.temp_dir)

    def test_kvs_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_kvs_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_kvs_map_distribute(self):
        test_result = MPControl.map(math_function, *self.map_test_data, tell_children=True)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_kvs_map_by_file(self):
        test_result = MPControl.map(math_function, *self.map_test_data, tell_children=True, tmp_file_path=self.temp_dir)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_kvs_sync(self):
        self.assertEqual(MPControl.sync_processes(), None)

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


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
class TestDaskLocalMPController(TestMPControl):
    name = "dask-local"
    client_name = "dask"
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(local_dir=cls.tempdir, n_workers=0)

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


@unittest.skipIf(not TEST_DASK_CLUSTER, "Dask not installed")
class TestDaskHPCMPController(TestMPControl):
    name = "dask-cluster"
    client_name = "dask"
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)

        # Create a wrapper for LocalCluster so that the HPC controller can be tested locally
        # And then bind it so that it works in py27 right
        def fake_cluster(*args, **kwargs):
            replace_args = dict()
            replace_args["n_workers"] = kwargs.pop("n_workers", 0)
            replace_args["threads_per_worker"] = kwargs.pop("threads_per_worker", 1)
            replace_args["processes"] = kwargs.pop("processes", True)
            replace_args["local_dir"] = kwargs.pop("local_directory", None)

            clust = distributed.LocalCluster(**replace_args)
            clust._active_worker_n = 0

            def _count_active_workers(self):
                val = self._active_worker_n
                self._active_worker_n += 1
                return val

            clust._count_active_workers = types.MethodType(_count_active_workers, clust)
            return clust

        MPControl.client.cluster_controller_class = types.MethodType(fake_cluster, MPControl.client)
        MPControl.client.worker_memory_limit = '1gb'
        MPControl.client.minimum_cores = 0
        MPControl.client.maximum_cores = 0
        MPControl.client.hack_cluster_controller_for_NYU = False
        MPControl.client.local_directory = cls.tempdir
        MPControl.connect()

    @classmethod
    def tearDownClass(cls):
        super(TestDaskHPCMPController, cls).tearDownClass()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)

    def test_dask_cluster_connect(self):
        self.assertTrue(MPControl.is_initialized)

    def test_dask_cluster_name(self):
        self.assertEqual(MPControl.name(), self.client_name)

    def test_dask_cluster_map(self):
        with self.assertRaises(NotImplementedError):
            MPControl.map(math_function, *self.map_test_data)

    def test_dask_cluster_sync(self):
        self.assertTrue(MPControl.sync_processes())

    def test_memory_0_hack(self):
        old_command = "dask-worker tcp://scheduler:port --memory-limit=4e9 --nthreads 1 --nprocs 20"
        new_command = "dask-worker tcp://scheduler:port --memory-limit 0 --nthreads 1 --nprocs 20"
        self.assertEqual(new_command, dask_cluster_controller.memory_limit_0(old_command))
        old_command_2 = "dask-worker tcp://scheduler:port --nthreads 1 --nprocs 20 --memory-limit=4e9"
        new_command_2 = "dask-worker tcp://scheduler:port --nthreads 1 --nprocs 20 --memory-limit 0 "
        self.assertEqual(new_command_2, dask_cluster_controller.memory_limit_0(old_command_2))

    def test_nyu_hack(self):
        old_header = "\n".join(['export MKL_NUM_THREADS=1', 'export OPENBLAS_NUM_THREADS=1',
                                'export NUMEXPR_NUM_THREADS=1', '#SBATCH --cpus-per-task=20'])
        new_header = "\n".join(['export MKL_NUM_THREADS=1', 'export OPENBLAS_NUM_THREADS=1',
                                'export NUMEXPR_NUM_THREADS=1', '#SBATCH --nodes=1', '#SBATCH --ntasks-per-node=1',
                                '#SBATCH --cpus-per-task=20'])
        self.assertEqual(new_header,
                         dask_cluster_controller.fix_header_for_nyu(old_header,
                                                                    dask_cluster_controller.DEFAULT_NYU_HEADER))
