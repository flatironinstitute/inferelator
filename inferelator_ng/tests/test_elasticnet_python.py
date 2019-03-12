import unittest
from inferelator_ng.regression import elasticnet_python
import pandas as pd
import numpy as np
import os

class TestElasticNet(unittest.TestCase):

    def test_elastic_net(self):
        X = np.array([[0, 1, 0, 1, 0, 1], [2, 3, 2, 3, 2, 3], [4, 5, 4, 5, 4, 5], [6, 7, 6, 7, 6, 7],
                      [8, 9, 8, 9, 8, 9]])
        Y = np.array([1, 2, 3, 4, 5, 6])
        params = dict(l1_ratio=[0.5, 0.7, 0.9],
                             eps=0.001,
                             n_alphas=50,
                             alphas=None,
                             fit_intercept=True,
                             normalize=False,
                             precompute='auto',
                             max_iter=1000,
                             tol=0.001,
                             cv=3,
                             copy_X=True,
                             verbose=0,
                             n_jobs=1,
                             positive=False,
                             random_state=99,
                             selection='random')
        elasticnet_python.elastic_net(X, Y, params)
