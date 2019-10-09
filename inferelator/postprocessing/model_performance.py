import pandas as pd
import numpy as np
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing import GOLD_STANDARD_COLUMN, CONFIDENCE_COLUMN, TARGET_COLUMN, REGULATOR_COLUMN


class RankSummingMetric(object):
    """
    This class takes a data set that has some rankable values and a gold standard for which elements of that data set
    are true and calculates confidence
    """

    # Metric name
    name = "Confidences"

    # Filter methods to align gold standard and confidences
    filter_method = None
    filter_method_lookup = {'overlap': 'filter_to_overlap', 'keep_all_gold_standard': 'filter_to_left_size'}

    # Data (wide)
    rankable_data = None
    gold_standard = None
    all_confidences = None

    # Processed data (long)
    confidence_data = None
    filtered_data = None

    # File name
    curve_file_name = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard'):
        """
        Take rankable data and process it into confidence scores which are stored in this object
        :param rankable_data: list(pd.DataFrame) [B x [G x K]]
            A list of numeric dataframes (with identical axes)
        :param gold_standard: pd.DataFrame [G x K]
            A dataframe which corresponds to known, gold-standard data
        :param filter_method: str
            The method of aligning the

        """

        # Get the filtering method
        assert check.argument_enum(filter_method, self.filter_method_lookup.keys())
        self.filter_method = getattr(self, self.filter_method_lookup[filter_method])

        # Explicitly cast the gold standard data to a boolean array [0,1]
        gold_standard = (gold_standard != 0).astype(int)
        self.gold_standard = gold_standard

        # Calculate confidences based on the ranked data
        self.all_confidences = self.compute_combined_confidences(rankable_data)

        # Convert the confidence data to long format
        confidence_data = utils.melt_and_reindex_dataframe(self.all_confidences, CONFIDENCE_COLUMN,
                                                           idx_name=TARGET_COLUMN, col_name=REGULATOR_COLUMN)

        # Attach the gold standard
        confidence_data = self.attach_gs_to_confidences(confidence_data, gold_standard)

        # Sort by confidence (descending) and reset the index
        self.confidence_data = confidence_data.sort_values(by=CONFIDENCE_COLUMN, ascending=False, na_position='last')
        self.confidence_data.reset_index(inplace=True)

        # Filter the gold standard and confidences down to a format that can be directly compared
        utils.Debug.vprint("GS: {gs} edges, Confidences: {conf} edges".format(gs=gold_standard.shape[0],
                                                                              conf=self.confidence_data.shape[0]),
                           level=0)

        self.filtered_data = self.filter_method(GOLD_STANDARD_COLUMN, CONFIDENCE_COLUMN, self.confidence_data)
        utils.Debug.vprint("Filtered data to {e} edges".format(e=self.filtered_data.shape[0], level=0))

    def score(self):
        raise NotImplementedError

    def auc(self):
        raise NotImplementedError

    def confidence_dataframe(self):
        return self.confidence_data

    def curve_dataframe(self):
        raise NotImplementedError

    def output_curve_pdf(self, output_dir, file_name):
        raise NotImplementedError

    @staticmethod
    def attach_gs_to_confidences(confidence_data, gold_standard):
        """
        Outer join the gold standard into the confidence data

        :param confidence_data: pd.DataFrame [G*K x n]
        :param gold_standard: pd.DataFrame [G x K]
        :return:
        """

        gold_standard = utils.melt_and_reindex_dataframe(gold_standard, GOLD_STANDARD_COLUMN, idx_name=TARGET_COLUMN,
                                                         col_name=REGULATOR_COLUMN)

        return confidence_data.join(gold_standard, how='outer', on=[TARGET_COLUMN, REGULATOR_COLUMN])

    @staticmethod
    def compute_combined_confidences(rankable_data):
        """
        Calculate confidences based on ranking value in all of the data frames and summing the ranks
        :param rankable_data: list(pd.DataFrame [M x N])
        :return combine_conf: pd.DataFrame [M x N]
        """
        # Create an 0s dataframe shaped to the data to be ranked
        combine_conf = pd.DataFrame(np.zeros(rankable_data[0].shape),
                                    index=rankable_data[0].index,
                                    columns=rankable_data[0].columns)

        for replicate in rankable_data:
            # Flatten and rank based on the beta error reductions
            ranked_replicate = np.reshape(pd.DataFrame(replicate.values.flatten()).rank().values, replicate.shape)
            # Sum the rankings for each bootstrap
            combine_conf += ranked_replicate

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        combine_conf = (combine_conf - min_element) / (len(rankable_data) * combine_conf.size - min_element)
        return combine_conf

    @staticmethod
    def filter_to_left_size(left_column, right_column, data):
        # Return data where one column (left) is not NA
        return data.dropna(subset=[left_column])

    @staticmethod
    def filter_to_overlap(left_column, right_column, data):
        # Return data where both columns are not NA
        return data.dropna(subset=[left_column, right_column])


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
        if utils.is_string(metric_ref):
            if metric_ref.lower() == "aupr" or metric_ref.lower() == "precision-recall":
                from inferelator.postprocessing.model_metrics import RankSummaryPR
                return RankSummaryPR
            else:
                raise ValueError("Parser {parser_str} unknown".format(parser_str=metric_ref))
        elif issubclass(metric_ref, RankSummingMetric):
            return metric_ref
        else:
            raise ValueError("Handler must be a string or a RankSummingMetric class")