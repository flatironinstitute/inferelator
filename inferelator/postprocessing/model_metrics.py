import numpy as np
import pandas as pd
import os

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing import TARGET_COLUMN, REGULATOR_COLUMN, PRECISION_COLUMN, RECALL_COLUMN
from inferelator.postprocessing import CONFIDENCE_COLUMN, GOLD_STANDARD_COLUMN, MCC_COLUMN

import matplotlib.pyplot as plt

TP, FP, TN, FN = 'TP', 'FP', 'TN', 'FN'


class RankSummaryPR(RankSummingMetric):
    """
    This class extends RankSumming and calculates precision-recall
    """

    name = "AUPR"
    curve_file_name = "pr_curve.pdf"

    # PR
    aupr = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        super().__init__(rankable_data, gold_standard, filter_method=filter_method)

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
    def plot_pr_curve(recall, precision, aupr, output_dir=None, file_name=None, dpi=300, figsize=(6, 4)):

        # Generate a plot
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(recall, precision)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')

        # Add the AUPR as an annotation
        ax.annotate("aupr = {aupr}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')

        if file_name is None or output_dir is None:
            return fig, ax
        else:
            # Save the plot and close
            fig.savefig(os.path.join(output_dir, file_name), dpi=dpi)
            plt.close(fig)

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
            data.reset_index(inplace=True)
            utils.Debug.vprint("Resorting confidences for PR", level=0)

        # Get indices for stuff
        zero_confidence = data[CONFIDENCE_COLUMN] == 0
        valid_gs_idx = ~pd.isnull(data[GOLD_STANDARD_COLUMN])
        zero_confidence_precision_idx = zero_confidence & valid_gs_idx

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
            RankSummaryMCC.transform_column(data, PRECISION_COLUMN, CONFIDENCE_COLUMN, transform_ties)
            RankSummaryMCC.transform_column(data, RECALL_COLUMN, CONFIDENCE_COLUMN, transform_ties)

        else:
            # Overwrite the precision of no-confidence with the mean value
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


class RankSummaryMCC(RankSummingMetric):
    """
    This class extends RankSumming and calculates Matthews correlation coefficient
    """

    name = "MCC"
    curve_file_name = "mccVSconf_curve.pdf"

    # PR
    mcc = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        super().__init__(rankable_data, gold_standard, filter_method=filter_method)

        # Calculate the precision and recall and store them with confidence data
        self.filtered_data = self.calculate_mcc(self.filtered_data.copy())

        # Calculate the AUC
        self.maxmcc = self.calculate_opt_mcc(self.filtered_data)
        self.nnzmcc = self.calculate_nnz_mcc(self.filtered_data)

        # Join the filtered precision/recall onto the full confidences
        join_data = self.filtered_data.loc[:, [TARGET_COLUMN, REGULATOR_COLUMN, MCC_COLUMN]]
        join_data = join_data.set_index([TARGET_COLUMN, REGULATOR_COLUMN])
        self.confidence_data = self.confidence_data.join(join_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

    def score(self):

        return self.name, self.maxmcc

    def curve_dataframe(self):

        conf = self.filtered_data[CONFIDENCE_COLUMN].values
        MCC = self.filtered_data[MCC_COLUMN].values
        return MCC, conf

    def output_curve_pdf(self, output_dir, file_name=None):

        file_name = self.curve_file_name if file_name is None else file_name

        # Extract the recall and precision data
        MCC, conf = self.curve_dataframe()

        # Plot the precision-recall curve
        self.plot_mcc_conf(MCC, conf, self.maxmcc, output_dir, file_name)

    @staticmethod
    def plot_mcc_conf(mcc, conf, optmcc, output_dir=None, file_name=None, dpi=300, figsize=(6, 4), plot1m=False):

        # Generate a plot
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        if plot1m:
            conf = 1 - conf
            xlabel = '1 - confidence'
        else:
            xlabel = 'confidence'
        ax.plot(conf, mcc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('MCC')

        ax.annotate("max MCC = {optmcc}".format(optmcc=optmcc), xy=(0.4, 0.05), xycoords='axes fraction')

        if file_name is None or output_dir is None:
            return fig, ax
        else:
            # Save the plot and close
            fig.savefig(os.path.join(output_dir, file_name), dpi=dpi)
            plt.close(fig)

    @staticmethod
    def calculate_opt_mcc(data):

        return np.nanmax(data[MCC_COLUMN])

    @staticmethod
    def calculate_nnz_mcc(data):

        from sklearn.metrics import matthews_corrcoef as mcc

        gs = data[GOLD_STANDARD_COLUMN].astype(bool).astype(float).values
        preds = data[CONFIDENCE_COLUMN].astype(bool).astype(float).values

        cond = preds.size - preds.sum()

        if cond < np.finfo(float).eps:
            return 0.

        nnzmcc = mcc(gs, preds)
        return nnzmcc

    @staticmethod
    def calculate_mcc(data):

        # Make sure data is sorted
        if not data.loc[~pd.isnull(data[CONFIDENCE_COLUMN]), CONFIDENCE_COLUMN].is_monotonic_decreasing:
            data = data.sort_values(by=CONFIDENCE_COLUMN, ascending=False, na_position='last')
            data.reset_index(inplace=True)
            utils.Debug.vprint("Resorting confidences for MCC", level=0)

        # Get indices for stuff
        valid_gs_idx = ~pd.isnull(data[GOLD_STANDARD_COLUMN])

        # Find the edges that are in the gold standard
        # valid_gs = (data.loc[valid_gs_idx, GOLD_STANDARD_COLUMN] != 0).astype(int)

        df = data.loc[valid_gs_idx, [CONFIDENCE_COLUMN, GOLD_STANDARD_COLUMN]]

        df[GOLD_STANDARD_COLUMN] = df[GOLD_STANDARD_COLUMN].astype(bool)

        df[TP] = (df[GOLD_STANDARD_COLUMN]).astype(int).cumsum()
        df[FP] = (~df[GOLD_STANDARD_COLUMN]).astype(int).cumsum()
        df[TN] = pd.Series((~df[GOLD_STANDARD_COLUMN].iloc[::-1]).astype(int).cumsum(),
                           index=df.index).shift(-1, fill_value=0)
        df[FN] = pd.Series((df[GOLD_STANDARD_COLUMN].iloc[::-1]).astype(int).cumsum(),
                           index=df.index).shift(-1, fill_value=0)

        RankSummaryMCC.transform_column(df, TP, CONFIDENCE_COLUMN, 'max')
        RankSummaryMCC.transform_column(df, FP, CONFIDENCE_COLUMN, 'max')
        RankSummaryMCC.transform_column(df, TN, CONFIDENCE_COLUMN, 'min')
        RankSummaryMCC.transform_column(df, FN, CONFIDENCE_COLUMN, 'min')

        df[MCC_COLUMN] = RankSummaryMCC.confusion_to_mcc(df[TP], df[TN], df[FP], df[FN])

        data.loc[valid_gs_idx, MCC_COLUMN] = df[MCC_COLUMN]
        return data

    @staticmethod
    def confusion_to_mcc(tp, tn, fp, fn):
        denominator = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn + fn)
        return ((tp * tn - fp * fn) / denominator).fillna(0)
