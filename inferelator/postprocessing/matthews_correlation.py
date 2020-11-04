import numpy as np

from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing import (TARGET_COLUMN, REGULATOR_COLUMN, CONFIDENCE_COLUMN, GOLD_STANDARD_COLUMN,
                                        MCC_COLUMN, TP, FP, TN, FN)

import matplotlib.pyplot as plt


class RankSummaryMCC(RankSummingMetric):
    """
    This class extends RankSumming and calculates Matthews correlation coefficient
    """

    name = "MCC"
    curve_file_name = "mccVSconf_curve.pdf"

    # MCC values
    mcc = None

    @property
    def optconf(self):
        return RankSummaryMCC.calculate_opt_conf_mcc(self.filtered_data)

    @property
    def maxmcc(self):
        return self.calculate_opt_mcc(self.filtered_data)

    @property
    def nnzmmc(self):
        return self.calculate_nnz_mcc(self.filtered_data)

    # Plotter function

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        super(RankSummaryMCC, self).__init__(rankable_data, gold_standard, filter_method=filter_method)

        # Calculate the precision and recall and store them with confidence data
        self.filtered_data = self.calculate_mcc(self.filtered_data.copy())

        # Join the filtered MCC onto the full confidences
        join_data = self.filtered_data.loc[:, [TARGET_COLUMN, REGULATOR_COLUMN, MCC_COLUMN]]
        join_data = join_data.set_index([TARGET_COLUMN, REGULATOR_COLUMN])
        self.confidence_data = self.confidence_data.join(join_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

    def score(self):

        return self.name, self.maxmcc

    def curve_dataframe(self):

        return self.filtered_data[[CONFIDENCE_COLUMN, MCC_COLUMN]]

    def output_curve(self, ax=None, figsize=(6, 4)):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)

        # Extract the recall and precision data
        curve = self.curve_dataframe()
        self.plot_mcc_conf(curve[MCC_COLUMN].values, curve[CONFIDENCE_COLUMN].values, self.maxmcc, self.optconf, ax)

        return ax

    @staticmethod
    def plot_mcc_conf(mcc, conf, optmcc, optconf, ax):

        # Generate a plot
        ax.plot(conf, mcc)
        ax.set_xlabel('Confidence')
        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('MCC')
        ax.vlines(float(optconf), 0, 1, transform=ax.get_xaxis_transform(), colors='r', linestyles='dashed')

        _msg = "max MCC = {optmcc:.4f}\noptimal conf = {optconf:.4f}\nnum_edges = {n}".format(optmcc=optmcc,
                                                                                              optconf=optconf,
                                                                                              n=np.sum(conf >= optconf))
        ax.annotate(_msg, xy=(0.4, 0.075), xycoords='axes fraction')

        return ax

    @staticmethod
    def calculate_opt_mcc(data):

        return np.nanmax(data[MCC_COLUMN])

    @staticmethod
    def calculate_opt_conf_mcc(data):

        return data[CONFIDENCE_COLUMN].iloc[np.argmax(data[MCC_COLUMN])]

    @staticmethod
    def calculate_nnz_mcc(data):

        from sklearn.metrics import matthews_corrcoef as mcc

        nnzmcc = mcc(data[GOLD_STANDARD_COLUMN].astype(bool).values, data[CONFIDENCE_COLUMN].astype(bool).values)
        return nnzmcc

    @staticmethod
    def calculate_mcc(data):

        df = RankSummingMetric.compute_confusion_matrix(data)
        data[MCC_COLUMN] = RankSummaryMCC.confusion_to_mcc(df[TP], df[TN], df[FP], df[FN])
        return data

    @staticmethod
    def confusion_to_mcc(tp, tn, fp, fn):
        return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


