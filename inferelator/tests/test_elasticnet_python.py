import unittest
from inferelator.regression import elasticnet_python
import pandas as pd
import numpy as np
import os

class TestElasticNet(unittest.TestCase):

    def test_elastic_net(self):
        X = np.array([[0, 1, 0, 1, 0, 1], [2, 3, 2, 3, 2, 3], [4, 5, 4, 5, 4, 5], [6, 7, 6, 7, 6, 7],
                      [8, 9, 8, 9, 8, 9]])
        Y = np.array([[1, 2, 3, 4, 5, 6]])
        params = {"l1_ratio":[0.5, 0.7, 0.9],
                             'eps': 0.001,
                             'n_alphas': 50,
                             'alphas': None,
                             'fit_intercept': True,
                             'normalize': False,
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

        raw_result = elasticnet_python.ElasticNetCV(**params).fit(X.T, Y.flatten())
        np.testing.assert_array_equal([0.0, 0.0, 0.0, 0.0, 0.0], raw_result.coef_)

        result = elasticnet_python.elastic_net(X, Y, params)
        print(result)
        pp = np.array([True, True, True, True, True])
        betas = ([0.0, 0.0, 0.0, 0.0, 0.0])
        betas_resc = ([0.0, 0.0, 0.0, 0.0, 0.0])
        check = {'pp': pp, 'betas': betas, 'betas_resc': betas_resc}
        for component in check.keys():
            for idx in range(0, len(check[component])):
                np.testing.assert_array_almost_equal(result[component][idx], check[component][idx], 2)

    def test_elastic_net_result(self):
        X = np.array([[10, 1, 5, 2, 7, 1], [15, 3, 12, 3, 11, 3], [4, 5, 0, 5, 4, 10], [6, 0, 3, 7, 6, 11],
                      [8, 15, 8, 10, 0, 9]])
        Y = np.array([[10, 2, 12, 4, 5, 6]])
        params = {"l1_ratio":[0.5, 0.7, 0.9],
                             'eps': 0.001,
                             'n_alphas': 50,
                             'alphas': None,
                             'fit_intercept': True,
                             'normalize': False,
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

        raw_result = elasticnet_python.ElasticNetCV(**params).fit(X.T, Y.flatten())
        np.testing.assert_almost_equal([0.0, 0.354414, 0.0, 0.0, 0.0], raw_result.coef_, 3)

        result = elasticnet_python.elastic_net(X, Y, params)
        print(result)
        pp = np.array([False, True, False, False, False])
        betas = ([0.74468085])
        betas_resc = ([0.50166146])
        check = {'pp': pp, 'betas': betas, 'betas_resc': betas_resc}
        for component in check.keys():
            for idx in range(0, len(check[component])):
                np.testing.assert_array_almost_equal(result[component][idx], check[component][idx], 2)
