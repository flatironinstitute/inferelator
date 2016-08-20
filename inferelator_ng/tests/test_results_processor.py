import unittest
from .. import results_processor
import pandas as pd
import numpy as np

class TestResultsProcessor(unittest.TestCase):

    def test_combining_confidences_one_beta(self):
        # rescaled betas are only in the 
        betas = pd.DataFrame(np.array([[1, 0], [1, 2]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([betas], [betas])
        confidences = rp.compute_combined_confidences()
        np.testing.assert_equal(confidences.values,
            np.array([[0.5,  0.0], [0.5,  1.0]]))

    def test_combining_confidences_two_betas(self):
        # rescaled betas are only in the 
        beta1 = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta2 = pd.DataFrame(np.array([[1, 0], [1, 2]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1, beta2], [beta1, beta2])
        confidences = rp.compute_combined_confidences()
        np.testing.assert_equal(confidences.values,
            np.array([[0.63636363636363635,0],[0.63636363636363635,0.54545454545454541]]))