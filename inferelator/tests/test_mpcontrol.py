import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
import numpy.testing as npt

from distributed import Future

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.distributed import dask_cluster_controller

from inferelator.distributed.dask import (
    _scatter_wrapper_args,
    process_futures_into_list,
    make_scatter_map
)

if sys.version_info[1] > 10:
    DASK_SKIP = True

else:
    DASK_SKIP = True

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


@unittest.skipIf(DASK_SKIP, "No dask")
class TestDaskLocalMPControllerJoblib(TestMPControl):
    name = "dask-local"
    client_name = "dask-local"

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(n_workers=1)
        MPControl.client.set_task_parameters(batch_size=2)

    @classmethod
    def tearDownClass(cls):
        MPControl.client.set_task_parameters(batch_size=1)
        super().tearDownClass()

    def test_dask_local_connect(self):
        self.assertTrue(MPControl.status())

    def test_dask_local_name(self):
        self.assertEqual(MPControl.name(), self.client_name)

    def test_dask_local_map(self):
        test_result = MPControl.map(math_function, *self.map_test_data)
        self.assertListEqual(test_result, self.map_test_expect)

    def test_scatter(self):

        a, b, c = np.random.rand(100), np.random.rand(50), np.random.rand(10)

        def _not_scattered(x, y, z):

            return (
                id(a) == id(x),
                id(b) == id(y),
                id(c) == id(z)
            )

        not_scattered = MPControl.map(_not_scattered,
            [a], [b], [c],
            scatter=[a, b, c]
        )

        self.assertEqual(sum(not_scattered[0]), 0)

        not_scattered = MPControl.map(_not_scattered,
            [a], [b], [c],
            scatter=[a, c]
        )

        self.assertEqual(sum(not_scattered[0]), 1)

        not_scattered = MPControl.map(_not_scattered,
            [a], [b], [c],
            scatter=[a]
        )

        self.assertEqual(sum(not_scattered[0]), 2)

        not_scattered = MPControl.map(_not_scattered,
            [a], [b], [c],
        )

        self.assertEqual(sum(not_scattered[0]), 3)


@unittest.skipIf(DASK_SKIP, "No dask")
class TestDaskAccessories(TestDaskLocalMPControllerJoblib):

    def setUp(self) -> None:
        self.a, self.b, self.c = (
            np.random.rand(100),
            np.random.rand(100),
            np.random.rand(100)
        )

        self.futures = [
            MPControl.client.client.submit(
                math_function,
                *a
            )
            for a in zip(
                [self.a, self.b],
                [self.b, self.a],
                [self.c, self.c]
            )
        ]

        return super().setUp()

    def test_scatter_replace(self):

        a, b, c = self.a, self.b, self.c

        scatter_map = make_scatter_map(
            [a, b],
            MPControl.client.client
        )

        self.assertEqual(len(scatter_map), 2)

        self.assertListEqual(
            [id(a), id(b)],
            list(scatter_map.keys())
        )

        for v in scatter_map.values():
            self.assertTrue(isinstance(v, Future))

        submit_stuff = _scatter_wrapper_args(
            a, b, c,
            scatter_map=scatter_map
        )

        self.assertTrue(isinstance(submit_stuff[0], Future))
        self.assertTrue(isinstance(submit_stuff[1], Future))
        self.assertFalse(isinstance(submit_stuff[2], Future))
        self.assertEqual(id(submit_stuff[2]), id(c))

        MPControl.client.client.cancel(scatter_map.values())

    def test_get_results(self):

        a, b, c = self.a, self.b, self.c

        res = process_futures_into_list(
            self.futures,
            MPControl.client.client
        )

        npt.assert_array_almost_equal(
            res[0],
            math_function(a, b, c)
        )

        npt.assert_array_almost_equal(
            res[1],
            math_function(b, a, c)
        )

    def test_get_results_missing(self):

        a, b, c = self.a, self.b, self.c

        self.futures[1].cancel()

        res = process_futures_into_list(
            self.futures,
            MPControl.client.client,
            raise_on_error=False
        )

        npt.assert_array_almost_equal(
            res[0],
            math_function(a, b, c)
        )

        self.assertIsNone(res[1])

    def test_get_results_error(self):

        self.futures[1].cancel()

        with self.assertRaises(KeyError):
            _ = process_futures_into_list(
                self.futures,
                MPControl.client.client
            )


@unittest.skipIf(DASK_SKIP, "No dask")
class TestDaskLocalMPControllerSubmit(TestDaskLocalMPControllerJoblib):

    @classmethod
    def setUpClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(n_workers=1)
        MPControl.client.set_task_parameters(batch_size=1)

    @unittest.skip("Doesnt work for submit because its not a coroutine")
    def test_scatter(self):
        pass


@unittest.skipIf(DASK_SKIP, "No dask")
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
        MPControl.client._interface = None
        MPControl.client._log_directory = cls.tempdir
        MPControl.client._local_directory = cls.tempdir
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
