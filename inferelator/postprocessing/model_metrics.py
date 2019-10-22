import numpy as np
import pandas as pd
import os

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing import TARGET_COLUMN, REGULATOR_COLUMN, PRECISION_COLUMN, RECALL_COLUMN
from inferelator.postprocessing import CONFIDENCE_COLUMN, GOLD_STANDARD_COLUMN

import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt


class RankSummaryPR(RankSummingMetric):
    """
    This class extends RankSumming and calculates precision-recall
    """

    name = "AUPR"
    curve_file_name = "pr_curve.pdf"

    # PR
    aupr = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        super(RankSummaryPR, self).__init__(rankable_data, gold_standard, filter_method=filter_method)

        # Calculate the precision and recall and store them with confidence data
        self.filtered_data = self.calculate_precision_recall(self.filtered_data.copy())

        # Calculate the AUC
        self.aupr = self.calculate_aupr(self.filtered_data)

        # Join the filtered precision/recall onto the full confidences
        join_data = self.filtered_data.loc[:, [TARGET_COLUMN, REGULATOR_COLUMN, PRECISION_COLUMN, RECALL_COLUMN]]
        join_data = join_data.set_index([TARGET_COLUMN, REGULATOR_COLUMN])
        self.confidence_data = self.confidence_data.join(join_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

    def score(self):

        return self.name, self.aupr

    def auc(self):

        return self.aupr

    def curve_dataframe(self):

        return self.filtered_data.loc[:, [PRECISION_COLUMN, RECALL_COLUMN]]

    def output_curve_pdf(self, output_dir, file_name=None):

        file_name = self.curve_file_name if file_name is None else file_name

        # Extract the recall and precision data
        recall, precision = self.modify_pr(self.curve_dataframe())

        # Plot the precision-recall curve
        self.plot_pr_curve(recall, precision, self.aupr, output_dir, file_name)

    @staticmethod
    def plot_pr_curve(recall, precision, aupr, output_dir, file_name):
        if file_name is None or output_dir is None:
            return None
        else:
            # Generate a plot
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('recall')
            plt.ylabel('precision')

            # Add the AUPR as an annotation
            plt.annotate("aupr = {aupr}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')

            # Save the plot and close
            plt.savefig(os.path.join(output_dir, file_name))
            plt.close()

    def num_over_precision_threshold(self, threshold):
        return np.sum(self.confidence_data[CONFIDENCE_COLUMN] >= self.find_threshold(PRECISION_COLUMN, threshold))

    def num_over_recall_threshold(self, threshold):
        return np.sum(self.confidence_data[CONFIDENCE_COLUMN] >= self.find_threshold(PRECISION_COLUMN, threshold))

    def num_over_conf_threshold(self, threshold):
        return np.sum(self.confidence_data[CONFIDENCE_COLUMN] >= threshold)

    def find_threshold(self, column_name, threshold):

        assert check.argument_numeric(threshold, low=0, high=1)

        threshold_index = self.confidence_data[column_name] >= threshold

        # If there's nothing in the index return np.inf.
        if np.sum(threshold_index) == 0:
            return np.inf
        else:
            return self.confidence_data.loc[threshold_index, CONFIDENCE_COLUMN].min()

    @staticmethod
    def calculate_precision_recall(data):
        """
        Calculate the precision & recall based on the confidence scores and gold standard
        :param data: pd.DataFrame
            Dataframe with a gold standard and confidence column, sorted on confidence column
        :return data: pd.DataFrame
            Sorted dataframe with additional precision and recall columns
        """

        # Make sure data is sorted
        if not data.loc[~pd.isnull(data[CONFIDENCE_COLUMN]), CONFIDENCE_COLUMN].is_monotonic_decreasing:
            data = data.sort_values(by=CONFIDENCE_COLUMN, ascending=False, na_position='last')
            data.reset_index(inplace=True)
            utils.Debug.vprint("Resorting confidences for PR", level=0)

        # Find the edges that are in the gold standard
        valid_gs_idx = ~pd.isnull(data[GOLD_STANDARD_COLUMN])
        valid_gs = (data.loc[valid_gs_idx, GOLD_STANDARD_COLUMN] != 0).astype(int)

        # the following mimics the R function ChristophsPR
        # Add nan columns
        data[PRECISION_COLUMN] = np.nan
        data[RECALL_COLUMN] = np.nan

        # Calculate precision [TP / (TP + FP)]
        data.loc[valid_gs_idx, PRECISION_COLUMN] = np.cumsum(valid_gs).astype(float) / np.cumsum([1] * len(valid_gs))

        # Calculate recall [TP / (TP + FN)]
        data.loc[valid_gs_idx, RECALL_COLUMN] = np.cumsum(valid_gs).astype(float) / sum(valid_gs)

        return data

    @staticmethod
    def modify_pr(data):
        """
        Inserts values into the precision and recall to allow for plotting & calculations of area
        :param data: pd.DataFrame
            Sorted confidences with a PRECISION and a RECALL column
        :return precision: np.ndarray
            Precision values
        :return recall: np.ndarray
            Recall values
        """

        data = data.loc[~pd.isnull(data[PRECISION_COLUMN]), :]
        precision = np.insert(data[PRECISION_COLUMN].values, 0, data[PRECISION_COLUMN].iloc[0])
        recall = np.insert(data[RECALL_COLUMN].values, 0, 0)
        return recall, precision

    @staticmethod
    def calculate_aupr(data):
        recall, precision = RankSummaryPR.modify_pr(data)
        # using midpoint integration to calculate the area under the curve
        d_recall = np.diff(recall)
        m_precision = precision[:-1] + np.diff(precision) / 2
        return sum(d_recall * m_precision)
