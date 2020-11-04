from inferelator.postprocessing.matthews_correlation import RankSummaryMCC
from inferelator.postprocessing.precision_recall import RankSummaryPR
from inferelator.postprocessing.f1_score import RankSummaryF1
from inferelator.postprocessing.model_performance import RankSummingMetric
from inferelator.postprocessing.column_names import (PRECISION_COLUMN, CONFIDENCE_COLUMN, RECALL_COLUMN, MCC_COLUMN,
                                                     F1_COLUMN, TARGET_COLUMN, REGULATOR_COLUMN)

from inferelator.utils import is_string
import matplotlib.pyplot as plt
import os


class MetricHandler(object):

    @classmethod
    def get_metric(cls, metric_ref):

        """
        This wrappers a metric reference so that strings can be used instead of python imports
        Will either return a metric class or will raise an error
        :param metric_ref: str / RankSummingMetric
            String or subclass of RankSummingMetric
        :return: RankSummingMetric
            The metadata parser that corresponds to the string, or the MetadataParser object will be passed through
        """
        if is_string(metric_ref):
            metric_ref = metric_ref.lower()
            if metric_ref == "aupr" or metric_ref == "precision-recall":
                return RankSummaryPR
            if metric_ref == "mcc" or metric_ref == "matthews correlation coefficient":
                return RankSummaryMCC
            if metric_ref == "f1" or metric_ref == "f1 score":
                return RankSummaryF1
            if metric_ref == "combined":
                return CombinedMetric
            else:
                raise ValueError("Parser {parser_str} unknown".format(parser_str=metric_ref))
        elif issubclass(metric_ref, RankSummingMetric):
            return metric_ref
        else:
            raise ValueError("Handler must be a string or a RankSummingMetric class")


class CombinedMetric(RankSummaryF1, RankSummaryPR, RankSummaryMCC):

    name = "aupr"

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):

        # Call to the super __init__ to rank data
        RankSummingMetric.__init__(self, rankable_data, gold_standard, filter_method=filter_method)

        # Add Precision / Recall
        self.calculate_precision_recall(self.filtered_data)
        self.calculate_mcc(self.filtered_data)
        self.calculate_f1(self.filtered_data)

        # Join the filtered scores onto the full confidences
        join_data = self.filtered_data.loc[:, [TARGET_COLUMN, REGULATOR_COLUMN,
                                               PRECISION_COLUMN, RECALL_COLUMN, MCC_COLUMN, F1_COLUMN]]
        join_data = join_data.set_index([TARGET_COLUMN, REGULATOR_COLUMN])

        self.confidence_data = self.confidence_data.join(join_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

    def score(self):
        return self.name, self.aupr

    def auc(self):
        return self.aupr

    def curve_dataframe(self):
        return self.filtered_data.loc[:, [CONFIDENCE_COLUMN, PRECISION_COLUMN, RECALL_COLUMN, MCC_COLUMN, F1_COLUMN]]

    def output_curve_pdf(self, output_dir, file_name=None, dpi=300, figsize=(8, 10)):

        file_name = self.curve_file_name if file_name is None else file_name

        # Create a figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)

        RankSummaryPR.output_curve(self, ax=axes[0, 0])
        RankSummaryMCC.output_curve(self, ax=axes[0, 1])
        RankSummaryF1.output_curve(self, ax=axes[1, 0])

        # If there's a file name set, make the output file
        if file_name is not None and output_dir is not None:
            # Save the plot and close
            fig.savefig(os.path.join(output_dir, file_name), dpi=dpi)
