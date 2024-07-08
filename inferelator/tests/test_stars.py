import unittest
import numpy as np
import numpy.testing as npt

from .artifacts.test_data import (
    TEST_DATA
)

from inferelator.utils import InferelatorData

from inferelator.regression.stability_selection import (
    StARS,
    _make_subsample_idx,
    _regress_all_alphas,
    stars_model_select
)

A = np.random.default_rng(101).uniform(size=TEST_DATA.shape[1]).reshape(-1, 1)
A_comp = A.copy()
A_comp[2] = 0


class TestSTARSLasso(unittest.TestCase):

    def setUp(self) -> None:

        self.X = TEST_DATA.copy()
        self.y = InferelatorData((TEST_DATA.values @ A).reshape(-1, 1))
        self.alphas = np.array([0, 0.1, 0.25, 0.5, 1, 2, 5, 1000])

        self.num_subs = 2
        self.b = 5

    def test_STARS(self):

        out = StARS(
            self.X,
            self.y,
            40,
            alphas=self.alphas,
            num_subsamples=2
        ).regress()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['ind'], 0)

    def test_modeler(self):

        out = stars_model_select(
            self.X.values,
            self.y.values.ravel(),
            self.alphas,
            num_subsamples=2,
            random_seed=50
        )

        self.assertEqual(len(out), 4)
        self.assertEqual(len(out['betas']), 3)
        self.assertEqual(out['pp'].sum(), 3)
        self.assertEqual(out['selected_alpha'], 0.5)

    def test_subsample_idx(self):

        idx = _make_subsample_idx(
            10, self.b, self.num_subs, 50
        )

        npt.assert_equal(
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            idx
        )

    def test_regression_path(self):

        alphas, coefs = _regress_all_alphas(
            self.X.values,
            self.y.values,
            self.alphas,
            'lasso',
            random_state=40
        )

        npt.assert_equal(
            alphas, self.alphas[::-1]
        )

        npt.assert_almost_equal(
            np.zeros_like(coefs[0]),
            coefs[0]
        )

        npt.assert_almost_equal(
            A_comp.ravel(),
            coefs[-1]
        )
