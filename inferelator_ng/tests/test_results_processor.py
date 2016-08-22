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
            np.array([[0.5,  0.0], [0.5,      1.0]]))

    def test_combining_confidences_two_betas(self):
        # rescaled betas are only in the 
        beta1 = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta2 = pd.DataFrame(np.array([[1, 0], [1, 2]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1, beta2], [beta1, beta2])
        confidences = rp.compute_combined_confidences()
        np.testing.assert_equal(confidences.values,
            np.array([[0.63636363636363635,0],[0.63636363636363635,0.54545454545454541]]))

    def test_threshold_and_summarize_one_beta(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1], [beta1])
        thresholded_mat = rp.threshold_and_summarize()
        np.testing.assert_equal(thresholded_mat.values,
            np.array([[1,0],[1,0]]))

    def test_threshold_and_summarize_two_betas(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 1]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1, beta2], [beta1, beta2])
        thresholded_mat = rp.threshold_and_summarize()
        np.testing.assert_equal(thresholded_mat.values,
            np.array([[1,0],[1,1]]))

    def test_threshold_and_summarize_three_betas(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta3 = pd.DataFrame(np.array([[0.5, 0.2], [0.5, 0.1]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1, beta2, beta3], [beta1, beta2, beta3])
        thresholded_mat = rp.threshold_and_summarize()
        np.testing.assert_equal(thresholded_mat.values,
            np.array([[1,0],[1,0]]))

    def test_threshold_and_summarize_three_betas_negative_values(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [-0.5, 0]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [-0.5, 1]]), ['gene1', 'gene2'], ['tf1','tf2'])
        beta3 = pd.DataFrame(np.array([[-0.5, 0.2], [-0.5, 0.1]]), ['gene1', 'gene2'], ['tf1','tf2'])
        rp = results_processor.ResultsProcessor([beta1, beta2, beta3], [beta1, beta2, beta3])
        thresholded_mat = rp.threshold_and_summarize()
        np.testing.assert_equal(thresholded_mat.values,
            np.array([[1,0],[1,1]]))

