import numpy as np

from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing import (
    TARGET_COLUMN,
    REGULATOR_COLUMN,
    CONFIDENCE_COLUMN,
    MCC_COLUMN,
    TP,
    FP,
    TN,
    FN
)

from ._plot_fix import plt


class RankSummaryMCC(RankSummingMetric):
    """
    This class extends RankSumming and calculates Matthews
    correlation coefficient
    """

    name = "MCC"
    curve_file_name = "mccVSconf_curve.pdf"

    @property
    def mcc(self):
        return self.maxmcc

    @property
    def optconfmcc(self):
        return RankSummaryMCC.calculate_opt_conf_mcc(self.filtered_data)

    @property
    def maxmcc(self):
        return self.calculate_opt_mcc(self.filtered_data)

    @property
    def nnzmmc(self):
        return self.calculate_nnz_mcc(self.filtered_data, self.optconfmcc)

    # Plotter function

    def __init__(
        self,
        rankable_data,
        gold_standard,
        filter_method='keep_all_gold_standard'
    ):

        super(RankSummaryMCC, self).__init__(
            rankable_data,
            gold_standard,
            filter_method=filter_method
        )

        # Calculate the precision and recall and store them with confidence
        # data
        self.filtered_data = self.calculate_mcc(self.filtered_data.copy())

        # Join the filtered MCC onto the full confidences
        join_data = self.filtered_data.loc[
            :,
            [TARGET_COLUMN, REGULATOR_COLUMN, MCC_COLUMN]
        ].set_index([TARGET_COLUMN, REGULATOR_COLUMN])

        self.confidence_data = self.confidence_data.join(
            join_data,
            on=[TARGET_COLUMN, REGULATOR_COLUMN]
        )

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
        self.plot_mcc_conf(
            curve[MCC_COLUMN].values,
            curve[CONFIDENCE_COLUMN].values,
            self.maxmcc,
            self.optconfmcc,
            ax,
            num_edges=(
                self.confidence_data[CONFIDENCE_COLUMN] >= self.optconfmcc
            ).sum()
        )

        return ax

    @staticmethod
    def plot_mcc_conf(mcc, conf, optmcc, optconf, ax, num_edges=None):

        num_edges = np.sum(conf >= optconf) if num_edges is None else num_edges

        y_min = np.nanmin(mcc)
        y_min = 0 if not (0 > y_min) else y_min

        # Generate a plot
        ax.plot(conf, mcc)
        ax.set_xlabel('Confidence')
        ax.set_xlim(1, 0)
        ax.set_ylim(y_min, 1)
        ax.set_ylabel('MCC')
        ax.vlines(
            float(optconf),
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors='r',
            linestyles='dashed'
        )

        ax.annotate(
            f"max MCC = {optmcc:.4f}\n"
            f"optimal conf = {optconf:.4f}\n"
            f"num_edges = {num_edges}",
            xy=(0.4, 0.075),
            xycoords='axes fraction'
        )

        return ax

    @staticmethod
    def calculate_opt_mcc(data):

        return data[MCC_COLUMN].iloc[np.argmax(np.abs(data[MCC_COLUMN]))]

    @staticmethod
    def calculate_opt_conf_mcc(data):

        return data.loc[
            data[MCC_COLUMN] >= np.max(data[MCC_COLUMN]),
            CONFIDENCE_COLUMN
        ].min()

    @staticmethod
    def calculate_nnz_mcc(data, conf):

        return (data[CONFIDENCE_COLUMN] >= conf).sum()

    @staticmethod
    def calculate_mcc(data):

        df = RankSummingMetric.compute_confusion_matrix(data)
        data[MCC_COLUMN] = RankSummaryMCC.confusion_to_mcc(
            df[TP],
            df[TN],
            df[FP],
            df[FN]
        )
        return data

    @staticmethod
    def confusion_to_mcc(tp, tn, fp, fn):
        denominator = (
            np.sqrt(tp + fp) *
            np.sqrt(tp + fn) *
            np.sqrt(tn + fp) *
            np.sqrt(tn + fn)
        )

        # If any denominator value is 0, MCC is 0/0 and by
        # convention will be set to 0.0
        denominator[denominator == 0] = 1.0

        return (tp * tn - fp * fn) / denominator
