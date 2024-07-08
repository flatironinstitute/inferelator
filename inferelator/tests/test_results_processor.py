import unittest
from inferelator import utils
from inferelator.postprocessing import (
    GOLD_STANDARD_COLUMN,
    CONFIDENCE_COLUMN,
    TARGET_COLUMN,
    REGULATOR_COLUMN
)

from inferelator.postprocessing import results_processor
from inferelator.postprocessing import results_processor_mtl
from inferelator.postprocessing import MetricHandler, RankSummingMetric
from inferelator.postprocessing.results_processor import ResultsProcessor

import pandas as pd
import pandas.testing as pdt
import numpy as np
import os
import tempfile
import shutil
import anndata as ad

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt


class TestResults(unittest.TestCase):

    def setUp(self):

        # Data was taken from a subset of row 42 of
        # Bacillus subtilis run results
        self.beta1 = pd.DataFrame(
            np.array([[-0.2841755, 0, 0.2280624, -0.3852462, 0.2545609]]),
            index=['gene1'],
            columns=['tf1', 'tf2', 'tf3', 'tf4', 'tf5']
        )

        self.rescaled_beta1 = pd.DataFrame(
            np.array([[0.09488207, 0, 0.07380172, 0.15597205, 0.07595131]]),
            index=['gene1'],
            columns=['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])

        self.beta2 = pd.DataFrame(
            np.array([[0, 0.2612011, 0.1922999, 0.00000000, 0.19183277]]),
            index=['gene1'],
            columns=['tf1', 'tf2', 'tf3', 'tf4', 'tf5'])

        self.rescaled_beta2 = pd.DataFrame(
            np.array([[0, 0.09109101, 0.05830292, 0.00000000, 0.3675702]]),
            index=['gene1'],
            columns=['tf1', 'tf2', 'tf3', 'tf4', 'tf5']
        )

        # Toy data
        self.beta = pd.DataFrame(
            np.array([[0, 1], [0.5, 0.05]]),
            index=['gene1', 'gene2'],
            columns=['tf1', 'tf2']
        )

        self.beta_resc = pd.DataFrame(
            np.array([[0, 1.1], [1, 0.05]]),
            index=['gene1', 'gene2'],
            columns=['tf1', 'tf2']
        )

        self.prior = pd.DataFrame(
            [[0, 1], [1, 0]],
            index=['gene1', 'gene2'],
            columns=['tf1', 'tf2']
        )

        self.gold_standard = pd.DataFrame(
            [[0, 1], [1, 0]],
            index=['gene1', 'gene2'],
            columns=['tf1', 'tf2']
        )

        self.gold_standard_unaligned = pd.DataFrame(
            [[0, 1], [0, 0]],
            index=['gene1', 'gene3'],
            columns=['tf1', 'tf2']
        )

        self.metric = MetricHandler.get_metric("combined")

    def test_output_files(self):

        with tempfile.TemporaryDirectory() as td:
            rp = ResultsProcessor(
                [self.beta],
                [self.beta_resc],
                metric=self.metric
            )

            result = rp.summarize_network(
                td,
                self.gold_standard,
                self.prior,
                full_model_betas=self.beta,
                full_model_var_exp=self.beta_resc
            )

            if result.model_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.model_file_name))
                )

            if result.curve_data_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.curve_data_file_name))
                )

            if result.curve_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.curve_file_name))
                )

            if result.network_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.network_file_name))
                )

            if result.confidence_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.confidence_file_name))
                )

            if result.threshold_file_name is not None:
                self.assertTrue(
                    os.path.exists(os.path.join(td, result.threshold_file_name))
                )

    def test_model_h5_file(self):

        with tempfile.TemporaryDirectory() as td:
            rp = ResultsProcessor(
                [self.beta],
                [self.beta_resc],
                metric=self.metric
            )

            result = rp.summarize_network(
                td,
                self.gold_standard,
                self.prior,
                full_model_betas=self.beta,
                full_model_var_exp=self.beta_resc
            )

            adata = ad.read_h5ad(os.path.join(
                td, result.model_file_name
            ))

            self.assertEqual(adata.shape, self.beta.shape)
            self.assertTrue('prior' in adata.layers)
            self.assertTrue('gold_standard' in adata.layers)
            self.assertTrue('preprocessing' in adata.uns)
            self.assertTrue('scoring' in adata.uns)
            self.assertTrue('network' in adata.uns)

    @staticmethod
    def make_PR_data(gs, confidences):
        data = utils.melt_and_reindex_dataframe(
            confidences,
            value_name=CONFIDENCE_COLUMN
        ).reset_index()

        data = data.join(
            utils.melt_and_reindex_dataframe(
                gs,
                value_name=GOLD_STANDARD_COLUMN
            ),
            on=[TARGET_COLUMN, REGULATOR_COLUMN],
            how='outer'
        )

        return data


class TestResultsProcessor(TestResults):

    def test_full_stack(self):
        rp = ResultsProcessor([self.beta], [self.beta_resc])
        result = rp.summarize_network(None, self.gold_standard, self.prior)
        self.assertEqual(result.score, 1)

    def test_combining_confidences_two_betas_negative_values_assert_nonzero_betas(self):
        _, betas_non_zero = ResultsProcessor.summarize([self.beta1, self.beta2])
        np.testing.assert_equal(betas_non_zero, np.array([[1, 1, 2, 1, 2]]))

    def test_combining_confidences_two_betas_negative_values_assert_sign_betas(self):
        betas_sign, _ = ResultsProcessor.summarize([self.beta1, self.beta2])
        np.testing.assert_equal(betas_sign, np.array([[-1, 1, 2, -1, 2]]))

    def test_mean_and_median(self):
        beta = [
            pd.DataFrame(
                np.array([[1, 1], [1, 1]]),
                index=['gene1', 'gene2'],
                columns=['tf1', 'tf2']
            ),
            pd.DataFrame(
                np.array([[2, 2], [2, 2]]),
                index=['gene1', 'gene2'],
                columns=['tf1', 'tf2']
            )
        ]
        mean, median = ResultsProcessor.mean_and_median(beta)
        np.testing.assert_equal(mean, np.array([[1.5, 1.5], [1.5, 1.5]]))
        np.testing.assert_equal(median, np.array([[1.5, 1.5], [1.5, 1.5]]))


class TestNetworkCreator(TestResults):

    def setUp(self):
        super(TestNetworkCreator, self).setUp()
        self.metric = MetricHandler.get_metric("aupr")
        self.pr_calc = self.metric([self.rescaled_beta1, self.rescaled_beta2], self.gold_standard,
                                   "keep_all_gold_standard")
        self.beta_sign, self.beta_nonzero = ResultsProcessor.summarize([self.beta1, self.beta2])

    def test_process_network(self):
        net = ResultsProcessor.process_network(
            self.pr_calc,
            self.prior
        )

        self.assertListEqual(net['regulator'].tolist(), ['tf5', 'tf4', 'tf1'])
        self.assertListEqual(net['target'].tolist(), ['gene1'] * 3)
        self.assertListEqual(net['combined_confidences'].tolist(), [0.6, 0.3, 0.1])

    def test_network_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            net = ResultsProcessor.process_network(
                self.pr_calc,
                self.prior,
            )

            result = results_processor.InferelatorResults(
                net, self.beta,
                self.pr_calc.all_confidences,
                self.pr_calc
            )

            result.write_result_files(temp_dir)

            processed_data = pd.read_csv(
                os.path.join(temp_dir, "network.tsv.gz"),
                sep="\t",
                index_col=None,
                header=0
            )

            self.assertEqual(processed_data.shape[0], 3)
            self.assertListEqual(
                processed_data['regulator'].tolist(),
                ['tf5', 'tf4', 'tf1']
            )
            self.assertListEqual(
                processed_data['target'].tolist(),
                ['gene1'] * 3
            )
            self.assertListEqual(
                processed_data['combined_confidences'].tolist(),
                [0.6, 0.3, 0.1]
            )


class TestRankSummary(TestResults):

    def setUp(self):
        super(TestRankSummary, self).setUp()
        self.metric = RankSummingMetric

    def test_making_network_dataframe(self):
        calc = self.metric([self.beta_resc, self.beta_resc], self.gold_standard_unaligned)
        pdt.assert_frame_equal(calc.gold_standard, self.gold_standard_unaligned)
        self.assertEqual(calc.confidence_data.shape[0], 6)
        self.assertEqual(pd.isnull(calc.confidence_data[CONFIDENCE_COLUMN]).sum(), 2)
        self.assertEqual(pd.isnull(calc.confidence_data[GOLD_STANDARD_COLUMN]).sum(), 2)

    def test_combining_confidences_one_beta(self):
        # rescaled betas are only in the
        beta = pd.DataFrame(np.array([[0.5, 0], [0.5, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = self.metric.compute_combined_confidences([beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.5, 0.0], [0.5, 1.0]]))

    def test_combining_confidences_one_beta_invariant_to_rescale_division(self):
        # rescaled betas are only in the
        beta = pd.DataFrame(np.array([[1, 0], [1, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        rescaled_beta = beta / 3.0
        confidences = self.metric.compute_combined_confidences([rescaled_beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.5, 0.0], [0.5, 1.0]]))

    def test_combining_confidences_one_beta_all_negative_values(self):
        # rescaled betas are only in the
        beta = pd.DataFrame(np.array([[-1, -.5, -3], [-1, -2, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2', 'tf3'])
        rescaled_beta = pd.DataFrame([[0.2, 0.1, 0.4], [0.3, 0.5, 0]], ['gene1', 'gene2'], ['tf1', 'tf2', 'tf3'])
        confidences = self.metric.compute_combined_confidences([rescaled_beta])
        np.testing.assert_equal(confidences.values,
                                np.array([[0.4, 0.2, 0.8], [0.6, 1.0, 0]]))

    def test_combining_confidences_one_beta_with_negative_values(self):
        confidences = self.metric.compute_combined_confidences([self.rescaled_beta1])
        np.testing.assert_equal(confidences.values, np.array([[0.75, 0, 0.25, 1, 0.5]]))

    def test_combining_confidences_two_betas_negative_values(self):
        confidences = self.metric.compute_combined_confidences([self.rescaled_beta1, self.rescaled_beta2])
        np.testing.assert_equal(confidences.values, np.array([[0.1, 0., 0., 0.3, 0.6]]))

    def test_filter_to_left_size_equal(self):
        left = pd.DataFrame(np.array([[1, 1], [2, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        right = pd.DataFrame(np.array([[0, 0], [2, 2]]), ['gene1', 'gene2'], ['tf1', 'tf2'])

        data = self.make_PR_data(left, right)
        filter_data = self.metric.filter_to_left_size(GOLD_STANDARD_COLUMN, CONFIDENCE_COLUMN, data)
        self.assertEqual(data.shape, filter_data.shape)

    def test_output_files(self):
        pass

    def test_model_h5_file(self):
        pass


class TestPrecisionRecallMetric(TestResults):

    def setUp(self):
        super(TestPrecisionRecallMetric, self).setUp()
        self.metric = MetricHandler.get_metric("aupr")

    ####################

    # TODO: Fix the following three tests so that they have unique and correct precision recall values

    ####################

    def test_precision_recall_perfect_prediction(self):
        gs = self.gold_standard.copy()
        confidences = pd.DataFrame(np.array([[0, 1], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        recall, precision = self.metric.modify_pr(data)
        np.testing.assert_equal(recall, [0., 0.5, 1., 1., 1.])
        np.testing.assert_array_almost_equal(precision, [1., 1., 1., 7. / 12, 7. / 12])

    def test_precision_recall_unaligned_prediction(self):
        gs = self.gold_standard_unaligned.copy()
        confidences = pd.DataFrame(np.array([[0, 1], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        recall, precision = self.metric.modify_pr(data)
        np.testing.assert_equal(recall, [0., 1., 1., 1., 1.])
        np.testing.assert_equal(precision, [1., 1., 0.5, 1. / 3, 0.25])

    def test_precision_recall_prediction_off(self):
        gs = pd.DataFrame(np.array([[1, 0], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0.1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        recall, precision = self.metric.modify_pr(data)
        np.testing.assert_equal(recall, [0., 0.5, 0.5, 1., 1.])
        np.testing.assert_equal(precision, [1., 1., 0.5, 2. / 3, 0.5])

    def test_precision_recall_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data, transform_ties='mean')
        recall, precision = self.metric.modify_pr(data)
        np.testing.assert_equal(recall, [0., 0., 0., 0.75, 0.75])
        np.testing.assert_array_almost_equal(precision, [0., 0., 0., 5. / 12, 5. / 12])

    def test_aupr_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        aupr = self.metric.calculate_aupr(data)
        np.testing.assert_equal(aupr, 1.0)

    def test_negative_gs_aupr_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[-1, 0], [-1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        aupr = self.metric.calculate_aupr(data)
        np.testing.assert_equal(aupr, 1.0)

    def test_negative_gs_precision_recallbeta_resc_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, -1], [-1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data, transform_ties='mean')
        recall, precision = self.metric.modify_pr(data)
        np.testing.assert_equal(recall, [0., 0., 0., 0.75, 0.75])
        np.testing.assert_array_almost_equal(precision, [0., 0., 0., 5. / 12, 5. / 12])

    def test_aupr_prediction_off(self):
        gs = pd.DataFrame(np.array([[1, 0], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0.1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        aupr = self.metric.calculate_aupr(data)
        np.testing.assert_equal(aupr, 19. / 24)

    def test_aupr_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        data = self.make_PR_data(gs, confidences)
        data = self.metric.calculate_precision_recall(data)
        aupr = self.metric.calculate_aupr(data)
        np.testing.assert_approx_equal(aupr, 5. / 16)

    def test_rank_sum_increasing(self):
        rankable_data = [pd.DataFrame(np.array([[2.0, 4.0], [6.0, 8.0]]))]
        combine_conf = self.metric.compute_combined_confidences(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.0, 0.333333], [0.666667, 1.000000]]), 5)

    def test_rank_sum_decreasing(self):
        rankable_data = [pd.DataFrame(np.array([[8.0, 6.0], [4.0, 2.0]]))]
        combine_conf = self.metric.compute_combined_confidences(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[1.0, 0.666667], [0.333333, 0.0]]), 5)

    def test_rank_sum_random(self):
        rankable_data = [pd.DataFrame(np.array([[3.0, 2.0], [1.0, 4.0]]))]
        combine_conf = self.metric.compute_combined_confidences(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.666667, 0.333333], [0.0, 1.0]]), 5)

    def test_rank_sum_negative(self):
        rankable_data = [pd.DataFrame(np.array([[-2.0, 4.0], [-6.0, 8.0]]))]
        combine_conf = self.metric.compute_combined_confidences(rankable_data)
        np.testing.assert_array_almost_equal(combine_conf, np.array([[0.333333, 0.666667], [0.0, 1.0]]), 5)

    def test_rank_sum_zeros(self):
        rankable_data = [pd.DataFrame(np.array([[0, 0], [0, 0]]))]
        combine_conf = self.metric.compute_combined_confidences(rankable_data)
        np.testing.assert_array_equal(combine_conf, np.array([[0, 0], [0, 0]]))

    def test_plot_pr_curve(self):
        gs = pd.DataFrame(np.array([[-1, 0], [-1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])

        temp_dir = tempfile.mkdtemp()
        file_name = os.path.join(temp_dir, "pr_curve.pdf")
        self.metric = self.metric([confidences, confidences], gs)
        fig, ax = self.metric.output_curve_pdf(temp_dir, "pr_curve.pdf")
        self.assertTrue(os.path.exists(file_name))
        plt.close(fig)
        
        os.remove(file_name)
        self.assertFalse(os.path.exists(file_name))
        fig, ax = self.metric.output_curve_pdf(output_dir=temp_dir, file_name="pr_curve.pdf")
        self.assertTrue(os.path.exists(file_name))
        plt.close(fig)

        os.remove(file_name)
        self.assertFalse(os.path.exists(file_name))
        self.metric.curve_file_name = "pr_curve.pdf"
        fig, ax = self.metric.output_curve_pdf(output_dir=temp_dir, file_name=None)
        self.assertTrue(os.path.exists(file_name))
        plt.close(fig)

        os.remove(file_name)
        self.metric.curve_file_name = None
        self.assertFalse(os.path.exists(file_name))
        fig, ax = self.metric.output_curve_pdf(output_dir=temp_dir, file_name=None)
        self.assertFalse(os.path.exists(file_name))
        plt.close(fig)

        fig, ax = self.metric.output_curve_pdf(output_dir=None, file_name="pr_curve.pdf")
        self.assertFalse(os.path.exists(file_name))
        plt.close(fig)

        shutil.rmtree(temp_dir)


class TestMCCMetric(TestResults):

    def setUp(self):
        super(TestMCCMetric, self).setUp()
        self.metric = MetricHandler.get_metric("mcc")

    def test_mcc_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        mcc = self.metric([confidences, confidences], gs)
        np.testing.assert_approx_equal(mcc.score()[1], 1.0)

    def test_mcc_perfect_inverse_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        mcc = self.metric([confidences, confidences], gs)
        np.testing.assert_approx_equal(mcc.score()[1], -1)

    def test_mcc_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[0, 0], [0, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        mcc = self.metric([confidences, confidences], gs)
        np.testing.assert_approx_equal(mcc.score()[1], 0)


class TestF1Metric(TestResults):

    def setUp(self):
        super(TestF1Metric, self).setUp()
        self.metric = MetricHandler.get_metric("f1")

    def test_f1_perfect_prediction(self):
        gs = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0.5, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        f1 = self.metric([confidences, confidences], gs)
        np.testing.assert_equal(f1.score()[1], 1.0)

    @unittest.skip
    def test_f1_perfect_inverse_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [1, 0]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        f1 = self.metric([confidences, confidences], gs)
        print(f1.filtered_data)
        np.testing.assert_approx_equal(f1.score()[1], -1)

    @unittest.skip
    def test_f1_bad_prediction(self):
        gs = pd.DataFrame(np.array([[0, 1], [0, 1]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        confidences = pd.DataFrame(np.array([[1, 0], [0, 0.5]]), ['gene1', 'gene2'], ['tf1', 'tf2'])
        f1 = self.metric([confidences, confidences], gs)
        print(f1.filtered_data)
        np.testing.assert_approx_equal(f1.score()[1], 0)


class TestMTLResults(TestResults):

    def test_mtl_multiple_priors(self):
        rp = results_processor_mtl.ResultsProcessorMultiTask([[self.beta1], [self.beta1]], [[self.beta2], [self.beta2]])
        rp.write_task_files = False
        result = rp.summarize_network(
            None,
            self.gold_standard,
            [self.prior, self.prior],
            task_gold_standards=[self.gold_standard, self.gold_standard])
        self.assertEqual(result.score, 1)
