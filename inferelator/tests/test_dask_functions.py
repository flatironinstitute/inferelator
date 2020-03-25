import unittest
import tempfile
import pandas as pd
import os
import shutil
import numpy as np

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import InferelatorData

try:
    from dask import distributed
    from inferelator.distributed import dask_local_controller
    from inferelator.distributed import dask_functions

    TEST_DASK_LOCAL = True
except ImportError:
    TEST_DASK_LOCAL = False


@unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
@unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skip Travis for Dask")
class TestDaskLocalMPController(unittest.TestCase):
    name = "dask-local"
    tempdir = None

    @classmethod
    @unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skip Travis for Dask")
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        MPControl.shutdown()
        MPControl.set_multiprocess_engine(cls.name)
        MPControl.connect(local_dir=cls.tempdir, n_workers=1)

    @classmethod
    @unittest.skipIf(not TEST_DASK_LOCAL, "Dask not installed")
    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skip Travis for Dask")
    def tearDownClass(cls):
        MPControl.shutdown()
        MPControl.set_multiprocess_engine("local")
        MPControl.connect()
        if cls.tempdir is not None:
            shutil.rmtree(cls.tempdir)

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skip Travis for Dask")
    def test_dask_function_mi(self):
        """Compute mi for identical arrays [[1, 2, 1], [2, 4, 6]]."""

        L = [[0, 0], [9, 3], [0, 9]]
        x_dataframe = InferelatorData(pd.DataFrame(L))
        y_dataframe = InferelatorData(pd.DataFrame(L))
        mi = dask_functions.build_mi_array_dask(x_dataframe.values, y_dataframe.values, 10, np.log)
        expected = np.array([[0.63651417, 0.63651417], [0.63651417, 1.09861229]])
        np.testing.assert_almost_equal(mi, expected)

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skip Travis for Dask")
    def test_dask_function_bbsr(self):
        pp_mat = pd.DataFrame([[1, 1], [1, 1]], index=['gene1', 'gene2'], columns=['gene1', 'gene2'])
        X = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']).T)
        Y = InferelatorData(pd.DataFrame([1, 2], index=['gene1', 'gene2'], columns=['ss']).T)
        results = dask_functions.bbsr_regress_dask(X, Y, pp_mat, pp_mat.copy(), 2, ['gene1', 'gene2'], 10)
        expected_beta_0 = np.array([0, 0])
        expected_beta_1 = np.array([0, 0])
        np.testing.assert_almost_equal(results[0]['betas'], expected_beta_0)
        np.testing.assert_almost_equal(results[1]['betas'], expected_beta_1)
