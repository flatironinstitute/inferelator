import numpy as np
import pandas as pd
import os

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing import (TARGET_COLUMN, REGULATOR_COLUMN, PRECISION_COLUMN, RECALL_COLUMN,
                                        CONFIDENCE_COLUMN, GOLD_STANDARD_COLUMN)

import matplotlib

# If matplotlib is being an idiot and trying to set a tkinter backend, switch to agg
if matplotlib.get_backend() in (i for i in matplotlib.rcsetup.interactive_bk):
    matplotlib.use('agg')

import matplotlib.pyplot as plt


class RankSummaryPR(RankSummingMetric):
    """
    This class extends RankSumming and calculates precision-recall
    """

    name = "AUPR"
    curve_file_name = "pr_curve.pdf"

    # PR
    @property
    def aupr(self):
        return self.calculate_aupr(self.filtered_data)

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        super(RankSummaryPR, self).__init__(rankable_data, gold_standard, filter_method=filter_method)

        # Calculate the precision and recall and store them with confidence data
        self.filtered_data = self.calculate_precision_recall(self.filtered_data.copy())

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

    def output_curve(self, ax=None, figsize=(6, 4)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)

        # Extract the recall and precision data
        recall, precision = self.modify_pr(self.curve_dataframe())

        self.plot_pr_curve(recall, precision, self.aupr, ax)

        return ax

    @staticmethod
    def plot_pr_curve(recall, precision, aupr, ax):

        # Generate a plot
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add the AUPR as an annotation
        ax.annotate("aupr = {aupr:.5f}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')

        return ax

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
    def calculate_precision_recall(data, transform_ties=None):
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
            data = data.reset_index()
            utils.Debug.vprint("Resorting confidences for PR", level=0)

        # Get indices for stuff
        valid_gs_idx = ~pd.isnull(data[GOLD_STANDARD_COLUMN])

        # Find the edges that are in the gold standard
        valid_gs = (data.loc[valid_gs_idx, GOLD_STANDARD_COLUMN] != 0).astype(int)

        # the following mimics the R function ChristophsPR
        # Add nan columns
        data[PRECISION_COLUMN] = np.nan
        data[RECALL_COLUMN] = np.nan

        # Calculate precision [TP / (TP + FP)]
        data.loc[valid_gs_idx, PRECISION_COLUMN] = np.cumsum(valid_gs).astype(float) / np.arange(1, len(valid_gs)+1, 1)

        # Calculate recall [TP / (TP + FN)]
        data.loc[valid_gs_idx, RECALL_COLUMN] = np.cumsum(valid_gs).astype(float) / sum(valid_gs)

        if transform_ties is not None:
            RankSummingMetric.transform_column(data, CONFIDENCE_COLUMN, PRECISION_COLUMN, transform_ties)
            RankSummingMetric.transform_column(data, CONFIDENCE_COLUMN, RECALL_COLUMN, transform_ties)

        else:
            # Overwrite the precision of no-confidence with the mean value
            zero_confidence = data[CONFIDENCE_COLUMN] == 0
            zero_confidence_precision_idx = zero_confidence & valid_gs_idx

            zero_confidence_precision_val = data.loc[zero_confidence_precision_idx, PRECISION_COLUMN].mean()
            data.loc[zero_confidence_precision_idx, PRECISION_COLUMN] = zero_confidence_precision_val

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
