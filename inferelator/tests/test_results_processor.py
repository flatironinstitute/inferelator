from __future__ import division

import unittest
from inferelator.postprocessing import results_processor
from inferelator.postprocessing import model_performance
import pandas as pd
import numpy as np
import os
import tempfile
import shutil


class TestResults(unittest.TestCase):

    def setUp(self):
        # Data was taken from a subset of row 42 of Bacillus subtilis run results
        self.beta1 = pd.DataFrame(np.array([[-0.2841755, 0, 0.2280624, -0.3852462, 0.2545609]]), ['gene1'],
                                  ['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])
        self.rescaled_beta1 = pd.DataFrame(np.array([[0.09488207, 0, 0.07380172, 0.15597205, 0.07595131]]), ['gene1'],
                                           ['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])
        self.beta2 = pd.DataFrame(np.array([[0, 0.2612011, 0.1922999, 0.00000000, 0.19183277]]), ['gene1'],
                                  ['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])
        self.rescaled_beta2 = pd.DataFrame(np.array([[0, 0.09109101, 0.05830292, 0.00000000, 0.3675702]]), ['gene1'],
                                           ['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])

        # Toy data
        self.beta = pd.DataFrame(np.array([[0, 1], [0.5, 0.05]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        self.beta_resc = pd.DataFrame(np.array([[0, 1], [1, 0.05]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        self.prior = pd.DataFrame([[0, 1], [1, 0]], ['gene1', 'gene2'], ['tf1', 'tf2'])
        self.gold_standard = pd.DataFrame([[0, 1], [1, 0]], ['gene1', 'gene2'], ['tf1', 'tf2'])


class TestResultsProcessor(TestResults):

    def test_full_stack(self):
        rp = results_processor.ResultsProcessor([self.beta], [self.beta_resc])
        aupr = rp.summarize_network(None, self.gold_standard, self.prior)
        self.assertEqual(aupr, 1)

    def test_combining_confidences_two_betas_negative_values_assert_nonzero_betas(self):
        _, _, betas_non_zero = results_processor.ResultsProcessor.threshold_and_summarize([self.beta1, self.beta2], 0.5)
        np.testing.assert_equal(betas_non_zero, np.array([[1, 1, 2, 1, 2]]))

    def test_combining_confidences_two_betas_negative_values_assert_sign_betas(self):
        _, betas_sign, _ = results_processor.ResultsProcessor.threshold_and_summarize([self.beta1, self.beta2], 0.5)
        np.testing.assert_equal(betas_sign, np.array([[-1, 1, 2, -1, 2]]))

    def test_threshold_and_summarize_one_beta(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        thresholded_mat, _, _ = results_processor.ResultsProcessor.threshold_and_summarize([beta1], 0.5)
        np.testing.assert_equal(thresholded_mat.values, np.array([[1, 0], [1, 0]]))

    def test_threshold_and_summarize_two_betas(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        thresholded_mat, _, _ = results_processor.ResultsProcessor.threshold_and_summarize([beta1, beta2], 0.5)
        np.testing.assert_equal(thresholded_mat.values,
                                np.array([[1, 0], [1, 1]]))

    def test_threshold_and_summarize_three_betas(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta3 = pd.DataFrame(np.array([[0.5, 0.2], [0.5, 0.1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        thresholded_mat, _, _ = results_processor.ResultsProcessor.threshold_and_summarize([beta1, beta2, beta3], 0.5)
        np.testing.assert_equal(thresholded_mat.values,
                                np.array([[1, 0], [1, 0]]))

    def test_threshold_and_summarize_three_betas_negative_values(self):
        beta1 = pd.DataFrame(np.array([[1, 0], [-0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta2 = pd.DataFrame(np.array([[0, 0], [-0.5, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta3 = pd.DataFrame(np.array([[-0.5, 0.2], [-0.5, 0.1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        thresholded_mat, _, _ = results_processor.ResultsProcessor.threshold_and_summarize([beta1, beta2, beta3], 0.5)
        np.testing.assert_equal(thresholded_mat.values,
                                np.array([[1, 0], [1, 1]]))

    def test_mean_and_median(self):
        beta1 = pd.DataFrame(np.array([[1, 1], [1, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        beta2 = pd.DataFrame(np.array([[2, 2], [2, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        mean, median = results_processor.ResultsProcessor.mean_and_median([beta1, beta2])
        np.testing.assert_equal(mean, np.array([[1.5, 1.5], [1.5, 1.5]]))
        np.testing.assert_equal(median, np.array([[1.5, 1.5], [1.5, 1.5]]))


class TestNetworkCreator(TestResults):

    def setUp(self):
        super(TestNetworkCreator, self).setUp()
        self.pr_calc = model_performance.RankSummaryPR([self.rescaled_beta1, self.rescaled_beta2], self.gold_standard,
                                                       "keep_all_gold_standard")
        self.beta_sign, self.beta_nonzero = results_processor.ResultsProcessor.summarize([self.beta1, self.beta2])
        self.beta_threshold = results_processor.ResultsProcessor.passes_threshold(self.beta_nonzero, 2, 0.5)

    def test_process_network(self):
        net = results_processor.ResultsProcessor.process_network(self.pr_calc, self.prior,
                                                                 beta_threshold=self.beta_threshold)
        self.assertListEqual(net['regulator'].tolist(), ['tf5', 'tf4', 'tf1'])
        self.assertListEqual(net['target'].tolist(), ['gene1'] * 3)
        self.assertListEqual(net['combined_confidences'].tolist(), [0.6, 0.3, 0.1])

    def test_network_summary(self):
        temp_dir = tempfile.mkdtemp()
        net = results_processor.ResultsProcessor.process_network(self.pr_calc, self.prior,
                                                                 beta_threshold=self.beta_threshold)
        results_processor.ResultsProcessor.save_network_to_tsv(net, temp_dir)
        processed_data = pd.read_csv(os.path.join(temp_dir, "network.tsv"), sep="\t", index_col=None, header=0)
        self.assertEqual(processed_data.shape[0], 3)
        self.assertListEqual(processed_data['regulator'].tolist(), ['tf5', 'tf4', 'tf1'])
        self.assertListEqual(processed_data['target'].tolist(), ['gene1'] * 3)
        self.assertListEqual(processed_data['combined_confidences'].tolist(), [0.6, 0.3, 0.1])
        shutil.rmtree(temp_dir)


class TestPRProcessor(TestResults):

    def test_combining_confidences_one_beta(self):
        # rescaled betas are only in the 
        beta = pd.DataFrame(np.array([[0.5, 0], [0.5, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = model_performance.RankSummaryPR.compute_combined_confidences([beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.5, 0.0], [0.5, 1.0]]))

    def test_combining_confidences_one_beta_invariant_to_rescale_division(self):
        # rescaled betas are only in the 
        beta = pd.DataFrame(np.array([[1, 0], [1, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        rescaled_beta = pd.DataFrame((beta / 3.0), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = model_performance.RankSummaryPR.compute_combined_confidences([rescaled_beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.5, 0.0], [0.5, 1.0]]))

    def test_combining_confidences_one_beta_all_negative_values(self):
        # rescaled betas are only in the 
        beta = pd.DataFrame(np.array([[-1, -.5, -3], [-1, -2, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2', 'tf3'])
        rescaled_beta = pd.DataFrame([[0.2, 0.1, 0.4], [0.3, 0.5, 0]], ['gene1', 'gene2'], ['tf1', 'tf2', 'tf3'])
        confidences = model_performance.RankSummaryPR.compute_combined_confidences([rescaled_beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.4, 0.2, 0.8], [0.6, 1.0, 0]]))

    def test_combining_confidences_one_beta_with_negative_values(self):
        confidences = model_performance.RankSummaryPR.compute_combined_confidences([self.rescaled_beta1])
        np.testing.assert_equal(confidences.values, np.array([[0.75, 0, 0.25, 1, 0.5]]))

    def test_combining_confidences_two_betas_negative_values(self):
        confidences = model_performance.RankSummaryPR.compute_combined_confidences([self.rescaled_beta1,
                                                                                    self.rescaled_beta2])
        np.testing.assert_equal(confidences.values, np.array([[0.1, 0., 0., 0.3, 0.6]]))

    ####################

    # TODO: Fix the following three tests so that they have unique and correct precision recall values

    ####################

    def test_precision_recall_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        recall, precision = model_performance.RankSummaryPR.modify_pr(recall, precision)
        np.testing.assert_equal(recall, [0., 0.5, 1., 1., 1.])
        np.testing.assert_equal(precision, [1., 1., 1., 2. / 3, 0.5])

    def test_precision_recall_prediction_off(self):
        gs = pd.DataFrame(np.array([[1, 0], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        recall, precision = model_performance.RankSummaryPR.modify_pr(recall, precision)
        np.testing.assert_equal(recall, [0., 0.5, 0.5, 1., 1.])
        np.testing.assert_equal(precision, [1., 1., 0.5, 2. / 3, 0.5])

    def test_precision_recall_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        recall, precision = model_performance.RankSummaryPR.modify_pr(recall, precision)
        np.testing.assert_equal(recall, [0., 0., 0., 0.5, 1.])
        np.testing.assert_equal(precision, [0., 0., 0., 1. / 3, 0.5, ])

    def test_aupr_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        aupr = model_performance.RankSummaryPR.calculate_aupr(recall, precision)
        np.testing.assert_equal(aupr, 1.0)

    def test_negative_gs_aupr_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[-1, 0], [-1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        aupr = model_performance.RankSummaryPR.calculate_aupr(recall, precision)
        np.testing.assert_equal(aupr, 1.0)

    def test_negative_gs_precision_recall_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, -1], [-1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        recall, precision = model_performance.RankSummaryPR.modify_pr(recall, precision)
        np.testing.assert_equal(recall, [0., 0., 0., 0.5, 1.])
        np.testing.assert_equal(precision, [0., 0., 0., 1. / 3, 0.5, ])

    def test_aupr_prediction_off(self):
        gs = pd.DataFrame(np.array([[1, 0], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        aupr = model_performance.RankSummaryPR.calculate_aupr(recall, precision)
        np.testing.assert_equal(aupr, 19. / 24)

    def test_aupr_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        recall, precision, _ = model_performance.RankSummaryPR.calculate_precision_recall(confidences, gs)
        aupr = model_performance.RankSummaryPR.calculate_aupr(recall, precision)
        np.testing.assert_approx_equal(aupr, 7. / 24)

    def test_compute_combined_confidences_rank_method_sum(self):
        rankable_data = [pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         pd.DataFrame(np.array([[5.0, 6.0], [7.0, 8.0]]))]
        kwargs = {"rank_method": "sum"}
        rankable_data = model_performance.RankSummaryPR.compute_combined_confidences(rankable_data, **kwargs)

    def test_compute_combined_confidences_rank_method_sum_threshold(self):
        rankable_data = [pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         pd.DataFrame(np.array([[5.0, 6.0], [7.0, 8.0]]))]
        kwargs = {"rank_method": "threshold_sum", "data_threshold": 0.9}
        rankable_data = model_performance.RankSummaryPR.compute_combined_confidences(rankable_data, **kwargs)

    def test_compute_combined_confidences_rank_method_max_value(self):
        rankable_data = [pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         pd.DataFrame(np.array([[5.0, 6.0], [7.0, 8.0]]))]
        kwargs = {"rank_method": "max"}
        rankable_data = model_performance.RankSummaryPR.compute_combined_confidences(rankable_data, **kwargs)

    def test_compute_combined_confidences_rank_method_geo_mean(self):
        rankable_data = [pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])),
                         pd.DataFrame(np.array([[5.0, 6.0], [7.0, 8.0]]))]
        kwargs = {"rank_method": "geo_mean"}
        rankable_data = model_performance.RankSummaryPR.compute_combined_confidences(rankable_data, **kwargs)

    def test_rank_sum_increasing(self):
        rankable_data = [pd.DataFrame(np.array([[2.0, 4.0], [6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_sum(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.333333], [0.666667, 1.000000]]), 5)

    def test_rank_sum_decreasing(self):
        rankable_data = [pd.DataFrame(np.array([[8.0, 6.0], [4.0, 2.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_sum(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[1.0, 0.666667], [0.333333, 0.0]]), 5)

    def test_rank_sum_random(self):
        rankable_data = [pd.DataFrame(np.array([[3.0, 2.0], [1.0, 4.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_sum(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.666667, 0.333333], [0.0, 1.0]]), 5)

    def test_rank_sum_negative(self):
        rankable_data = [pd.DataFrame(np.array([[-2.0, 4.0], [-6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_sum(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.333333, 0.666667], [0.0, 1.0]]), 5)

    def test_rank_sum_zeros(self):
        rankable_data = [pd.DataFrame(np.array([[0, 0], [0, 0]]))]
        combine_conf = results_processor.RankSumming.rank_sum(rankable_data)
        np.testing.assert_array_equal(combine_conf, np.array([[0, 0], [0, 0]]))

    def test_rank_sum_threshold_increasing(self):
        rankable_data = [pd.DataFrame(np.array([[2.0, 4.0], [6.0, 8.0]]))]
        # pd.set_option('precision',16)
        combine_conf = model_performance.RankSummaryPR.rank_sum_threshold(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.333333], [0.666667, 1.000000]]), 5)
        # | computed_solution - true_solution | < \epsilon = O(1e-6)

    def test_rank_sum_threshold_decreasing(self):
        rankable_data = [pd.DataFrame(np.array([[8.0, 6.0], [4.0, 2.0]]))]
        # pd.set_option('precision',16)
        combine_conf = model_performance.RankSummaryPR.rank_sum_threshold(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[1.0, 0.666667], [0.333333, 0.0]]), 5)
        # | computed_solution - true_solution | < \epsilon = O(1e-6)

    def test_rank_sum_threshold_random(self):
        rankable_data = [pd.DataFrame(np.array([[3.0, 2.0], [1.0, 4.0]]))]
        # pd.set_option('precision',16)
        combine_conf = model_performance.RankSummaryPR.rank_sum_threshold(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.666667, 0.333333], [0.0, 1.0]]), 5)
        # | computed_solution - true_solution | < \epsilon = O(1e-6)

    def test_rank_sum_threshold_negative(self):
        rankable_data = [pd.DataFrame(np.array([[-2.0, 4.0], [-6.0, 8.0]]))]
        # pd.set_option('precision',16)
        combine_conf = model_performance.RankSummaryPR.rank_sum_threshold(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.75], [0.0, 1.0]]), 5)
        # | computed_solution - true_solution | < \epsilon = O(1e-6)

    def test_rank_sum_threshold_zeros(self):
        rankable_data = [pd.DataFrame(np.array([[0, 0], [0, 0]]))]
        combine_conf = results_processor.RankSumming.rank_sum_threshold(rankable_data)
        with self.assertRaises(ValueError):
            if any(np.isnan(combine_conf)):
                raise ValueError("combined_conf contains NaNs")

    def test_rank_max_value_increasing(self):
        rankable_data = [pd.DataFrame(np.array([[2.0, 4.0], [6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_max_value(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.333333], [0.666667, 1.000000]]), 5)

    def test_rank_max_value_decreasing(self):
        rankable_data = [pd.DataFrame(np.array([[8.0, 6.0], [4.0, 2.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_max_value(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[1.0, 0.666667], [0.333333, 0.0]]), 5)

    def test_rank_max_value_random(self):
        rankable_data = [pd.DataFrame(np.array([[3.0, 2.0], [1.0, 4.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_max_value(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.666667, 0.333333], [0.0, 1.0]]), 5)

    def test_rank_max_value_negative(self):
        rankable_data = [pd.DataFrame(np.array([[-2.0, 4.0], [-6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_max_value(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.6], [0.0, 1.0]]), 5)

    def test_rank_max_value_zero(self):
        rankable_data = [pd.DataFrame(np.array([[0, 0], [0, 0]]))]
        combine_conf = model_performance.RankSumming.rank_max_value(rankable_data)
        with self.assertRaises(ValueError):
            if any(np.isnan(combine_conf)):
                raise ValueError("combined_conf contains NaNs")

    def test_rank_geo_mean_increasing(self):
        rankable_data = [pd.DataFrame(np.array([[2.0, 4.0], [6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_geo_mean(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.333333], [0.666667, 1.000000]]), 5)

    def test_rank_geo_mean_decreasing(self):
        rankable_data = [pd.DataFrame(np.array([[8.0, 6.0], [4.0, 2.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_geo_mean(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[1.0, 0.666667], [0.333333, 0.0]]), 5)

    def test_rank_geo_mean_random(self):
        rankable_data = [pd.DataFrame(np.array([[3.0, 2.0], [1.0, 4.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_geo_mean(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.666667, 0.333333], [0.0, 1.0]]), 5)

    def test_rank_geo_mean_negative(self):
        rankable_data = [pd.DataFrame(np.array([[-2.0, 4.0], [-6.0, 8.0]]))]
        combine_conf = model_performance.RankSummaryPR.rank_geo_mean(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.333333, 0.666667], [0.0, 1.0]]), 5)

    def test_rank_geo_mean_zeros(self):
        rankable_data = [pd.DataFrame(np.array([[0, 0], [0, 0]]))]
        combine_conf = model_performance.RankSumming.rank_geo_mean(rankable_data)
        with self.assertRaises(ValueError):
            if any(np.isnan(combine_conf)):
                raise ValueError("combined_conf contains NaNs")

    def test_filter_to_left_size(self):
        left = pd.DataFrame(np.array([[1, 1], [2, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        right = pd.DataFrame(np.array([[0, 0], [2, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        model_performance.RankSummaryPR.filter_to_left_size(left, right)

    def test_plot_pr_curve(self):
        temp_dir = tempfile.mkdtemp()
        file_name = os.path.join(temp_dir, "pr_curve.pdf")
        model_performance.RankSummaryPR.plot_pr_curve([0, 1], [1, 0], "x", temp_dir, "pr_curve.pdf")
        self.assertTrue(os.path.exists(file_name))
        os.remove(file_name)

        model_performance.RankSummaryPR.plot_pr_curve(recall=[0, 1], precision=[1, 0], aupr=0.9, output_dir=temp_dir,
                                                      file_name="pr_curve.pdf")
        self.assertTrue(os.path.exists(file_name))
        os.remove(file_name)

        model_performance.RankSummaryPR.plot_pr_curve(recall=[0, 1], precision=[1, 0], aupr=0.9, output_dir=temp_dir,
                                                      file_name=None)
        self.assertFalse(os.path.exists(file_name))
        shutil.rmtree(temp_dir)
