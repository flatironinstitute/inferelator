import numpy as np

from inferelator.postprocessing.precision_recall import RankSummaryPR
from inferelator.postprocessing import (TARGET_COLUMN, REGULATOR_COLUMN, CONFIDENCE_COLUMN,
                                        F1_COLUMN, PRECISION_COLUMN, RECALL_COLUMN)

import matplotlib

# If matplotlib is being an idiot and trying to set a tkinter backend, switch to agg
if matplotlib.get_backend() in (i for i in matplotlib.rcsetup.interactive_bk):
    matplotlib.use('agg')

import matplotlib.pyplot as plt


class RankSummaryF1(RankSummaryPR):
    """
    This class extends RankSumming and calculates Matthews correlation coefficient
    """

    name = "F1"
    curve_file_name = "f1_curve.pdf"

    # F1 values
    f1 = None

    # Plotter function

    @property
    def optconff1(self):
        return RankSummaryF1.calculate_opt_conf_f1(self.filtered_data)

    @property
    def maxf1(self):
        return self.calculate_opt_f1(self.filtered_data)

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):
        super(RankSummaryPR, self).__init__(rankable_data, gold_standard, filter_method=filter_method)

        # Calculate the precision and recall and store them with confidence data
        self.filtered_data = self.calculate_precision_recall(self.filtered_data.copy(), transform_ties='mean')
        self.filtered_data = self.calculate_f1(self.filtered_data.copy())

        # Join the filtered F1 score onto the full confidences
        join_data = self.filtered_data.loc[:, [TARGET_COLUMN, REGULATOR_COLUMN, F1_COLUMN]]
        join_data = join_data.set_index([TARGET_COLUMN, REGULATOR_COLUMN])
        self.confidence_data = self.confidence_data.join(join_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

    def score(self):
        return self.name, self.maxf1

    def curve_dataframe(self):
        return self.filtered_data[[CONFIDENCE_COLUMN, F1_COLUMN]]

    def output_curve(self, ax=None, figsize=(6, 4)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)

        # Extract the recall and precision data
        curve = self.curve_dataframe()
        self.plot_f1_conf(curve[F1_COLUMN].values, curve[CONFIDENCE_COLUMN].values, self.maxf1, self.optconff1, ax,
                          num_edges=(self.confidence_data[CONFIDENCE_COLUMN] >= self.optconff1).sum())

        return ax

    @staticmethod
    def plot_f1_conf(f1, conf, optf1, optconf, ax, num_edges=None):

        num_edges = np.sum(conf >= optconf) if num_edges is None else num_edges

        # Generate a plot
        ax.plot(conf, f1)
        ax.set_xlabel('Confidence')
        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('F1')
        ax.vlines(float(optconf), 0, 1, transform=ax.get_xaxis_transform(), colors='r', linestyles='dashed')

        _msg = "max F1 = {optf1:.4f}\noptimal conf = {optconf:.4f}\nnum_edges = {n}".format(optf1=optf1,
                                                                                            optconf=optconf,
                                                                                            n=num_edges)
        ax.annotate(_msg, xy=(0.4, 0.075), xycoords='axes fraction')

        return ax

    @staticmethod
    def calculate_opt_f1(data):

        return np.nanmax(data[F1_COLUMN])

    @staticmethod
    def calculate_opt_conf_f1(data):

        return data.loc[data[F1_COLUMN] >= np.max(data[F1_COLUMN]), CONFIDENCE_COLUMN].min()

    @staticmethod
    def calculate_f1(data):

        data[F1_COLUMN] = RankSummaryF1.pr_to_f1(data[PRECISION_COLUMN], data[RECALL_COLUMN])
        return data

    @staticmethod
    def pr_to_f1(prec, recall):
        f1 = 2 * (prec * recall) / (prec + recall)
        f1[np.isnan(f1)] = 0.
        return f1
