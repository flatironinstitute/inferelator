import unittest
from inferelator.regression import elasticnet_python
from inferelator.regression import sklearn_regression
import numpy as np

TEST_SEED = 50

PREDICT_ARRAY = np.random.RandomState(TEST_SEED).randn(100, 5)
RESPONSE_ARRAY = np.random.RandomState(TEST_SEED).randn(100, 1)
RESPONSE_ARRAY[:, 0] = np.sort(RESPONSE_ARRAY[:, 0]).ravel()

PARAMS = {"l1_ratio": [0.5, 0.7, 0.9],
          'eps': 0.001,
          'n_alphas': 50,
          'alphas': None,
          'fit_intercept': True,
          'precompute': 'auto',
          'max_iter': 1000,
          'tol': 0.001,
          'cv': 3,
          'copy_X': True,
          'verbose': 0,
          'n_jobs': 1,
          'positive': False,
          'random_state': 99,
          'selection': 'random'}

MIN_COEF = 0.1


class TestElasticNet(unittest.TestCase):

    def test_elastic_net_sklearn_zeros(self):
        x = PREDICT_ARRAY.copy()
        y = RESPONSE_ARRAY.copy()

        raw_result = elasticnet_python.ElasticNetCV(
            **PARAMS
        ).fit(x, y.flatten())
        np.testing.assert_array_equal(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            raw_result.coef_
        )

    def test_elastic_net_sklearn_nz(self):
        x = PREDICT_ARRAY.copy()
        y = RESPONSE_ARRAY.copy()

        x[:, 2] = np.sort(x[:, 2])

        raw_result = elasticnet_python.ElasticNetCV(
            **PARAMS
        ).fit(x, y.flatten())

        np.testing.assert_array_equal(
            [False, False, True, False, False],
            abs(raw_result.coef_) > 0.1
        )

    def test_elastic_net_zeros(self):
        x = PREDICT_ARRAY.copy()
        y = RESPONSE_ARRAY.copy()

        result = sklearn_regression.sklearn_gene(
            x,
            y.flatten(),
            elasticnet_python.ElasticNetCV(**PARAMS),
            min_coef=MIN_COEF
        )

        self.assertEqual(len(result["pp"]), 5)
        self.assertEqual(len(result["betas"]), 5)
        self.assertEqual(len(result["betas_resc"]), 5)

        pp = np.array([True, True, True, True, True])
        betas = ([0.0, 0.0, 0.0, 0.0, 0.0])
        betas_resc = ([0.0, 0.0, 0.0, 0.0, 0.0])
        check = {'pp': pp, 'betas': betas, 'betas_resc': betas_resc}
        for component in check.keys():
            for idx in range(0, len(check[component])):
                np.testing.assert_array_almost_equal(
                    result[component][idx],
                    check[component][idx], 2
                )

    def test_elastic_net_nz(self):
        x = PREDICT_ARRAY.copy()
        y = RESPONSE_ARRAY.copy()

        x[:, 2] = np.sort(x[:, 2])

        result = sklearn_regression.sklearn_gene(
            x,
            y.flatten(),
            elasticnet_python.ElasticNetCV(**PARAMS),
            min_coef=MIN_COEF
        )

        self.assertEqual(len(result["pp"]), 5)
        self.assertEqual(len(result["betas"]), 1)
        self.assertEqual(len(result["betas_resc"]), 1)

        pp = np.array([False, False, True, False, False])
        betas = ([1.05])
        betas_resc = ([0.97])
        check = {'pp': pp, 'betas': betas, 'betas_resc': betas_resc}
        for component in check.keys():
            for idx in range(0, len(check[component])):
                np.testing.assert_array_almost_equal(
                    result[component][idx],
                    check[component][idx], 2
                )
