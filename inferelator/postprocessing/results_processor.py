import numpy as np
import pandas as pd

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_metrics import RankSummingMetric, MetricHandler
from inferelator.postprocessing.inferelator_results import InferelatorResults
from inferelator.postprocessing import (BETA_SIGN_COLUMN, MEDIAN_EXPLAIN_VAR_COLUMN, CONFIDENCE_COLUMN,
                                        BETA_THRESHOLD_COLUMN, TARGET_COLUMN, REGULATOR_COLUMN, PRIOR_COLUMN)

FILTER_METHODS = ("overlap", "keep_all_gold_standard")
DEFAULT_BOOTSTRAP_THRESHOLD = 0.5
DEFAULT_FILTER_METHOD = "overlap"
DEFAULT_METRIC = "precision-recall"


class ResultsProcessor(object):
    # Data
    betas = None
    rescaled_betas = None
    filter_method = DEFAULT_FILTER_METHOD

    # Processed Network
    network_data = None

    # Cutoffs
    threshold = DEFAULT_BOOTSTRAP_THRESHOLD

    # File names
    network_file_name = "network.tsv"
    confidence_file_name = "combined_confidences.tsv"
    threshold_file_name = "betas_stack.tsv"
    pr_curve_file_name = "result_curve.pdf"

    # Flag to write results
    write_results = True

    # Model result object
    result_object = InferelatorResults

    # Model metric
    metric = None

    def __init__(self, betas, rescaled_betas, threshold=None, filter_method=None, metric=None):
        """
        :param betas: list(pd.DataFrame[G x K]) [B]
            A list of model weights per bootstrap
        :param rescaled_betas: list(pd.DataFrame[G x K]) [B]
            A list of the variance explained by each parameter per bootstrap
        :param threshold: float
            The proportion of bootstraps which an model weight must be non-zero for inclusion in the network output
        :param filter_method: str
            How to handle gold standard filtering ('overlap' filters to beta, 'keep_all_gold_standard' doesn't filter)
        :param metric: str / RankSummingMetric
            The scoring metric to use
        """

        self.validate_init_args(betas, rescaled_betas, threshold=threshold, filter_method=filter_method, metric=metric)

        self.betas = betas
        self.rescaled_betas = rescaled_betas
        self.filter_method = self.filter_method if filter_method is None else filter_method
        self.threshold = self.threshold if threshold is None else threshold

        metric = metric if metric is not None else DEFAULT_METRIC
        self.metric = MetricHandler.get_metric(metric)

    @staticmethod
    def validate_init_args(betas, rescaled_betas, threshold=None, filter_method=None, metric=None):
        assert check.argument_type(betas, list)
        assert check.argument_type(betas[0], pd.DataFrame)
        assert check.dataframes_align(betas)
        assert check.argument_type(rescaled_betas, list)
        assert check.argument_type(rescaled_betas[0], pd.DataFrame)
        assert check.dataframes_align(rescaled_betas)
        assert check.argument_enum(filter_method, FILTER_METHODS, allow_none=True)
        assert check.argument_numeric(threshold, 0, 1, allow_none=True)

    def summarize_network(self, output_dir, gold_standard, priors):
        """
        Take the betas and rescaled beta_errors, construct a network, and test it against the gold standard
        :param output_dir: str
            Path to write files into. Don't write anything if this is None.
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard to test the network against
        :param priors: pd.DataFrame [G x K]
            Prior data
        :return result: InferelatorResult
            Returns an InferelatorResult
        """

        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(gold_standard, pd.DataFrame)
        assert check.argument_type(priors, pd.DataFrame)

        rs_calc = self.metric(self.rescaled_betas, gold_standard, filter_method=self.filter_method)
        beta_threshold, beta_sign, beta_nonzero = self.threshold_and_summarize(self.betas, self.threshold)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        extra_cols = {BETA_SIGN_COLUMN: beta_sign, MEDIAN_EXPLAIN_VAR_COLUMN: resc_betas_median}

        m_name, score = rs_calc.score()
        utils.Debug.vprint("Model {metric}:\t{score}".format(metric=m_name, score=score), level=0)

        # Process data into a network dataframe
        network_data = self.process_network(rs_calc, priors, beta_threshold=beta_threshold, extra_columns=extra_cols)

        # Create a InferelatorResult object and have it write output files
        result = self.result_object(network_data, beta_threshold, rs_calc.all_confidences, rs_calc,
                                    betas_sign=beta_sign, betas=self.betas)

        if self.write_results and output_dir is not None:
            result.write_result_files(output_dir)

        return result

    @staticmethod
    def process_network(metric, priors, confidence_threshold=0, beta_threshold=None, extra_columns=None):
        """
        Process rank-summed results into a network data frame
        :param metric: RankSummingMetric
            The rank-sum object with the math in it
        :param priors: pd.DataFrame [G x K]
            Prior data
        :param confidence_threshold: numeric
            The minimum confidence score needed to write a network edge
        :param beta_threshold: pd.DataFrame [G x K]
            The thresholded betas to include in the network. If None, include everything.
        :param extra_columns: dict(col_name: pd.DataFrame [G x K])
            Any additional data to include, keyed by column name and indexable with row and column names
        :return network_data: pd.DataFrame [(G*K) x 7+]
            Network edge dataframe

        """

        assert check.argument_type(metric, RankSummingMetric)
        assert check.argument_type(priors, pd.DataFrame, allow_none=True)
        assert check.argument_type(beta_threshold, pd.DataFrame, allow_none=True)
        assert check.argument_numeric(confidence_threshold, 0, 1)

        # Get the combined confidences and subset for confidence threshold
        network_data = metric.confidence_dataframe()
        network_data = network_data.loc[network_data[CONFIDENCE_COLUMN] > confidence_threshold, :]

        # If beta_threshold has been provided, melt and join it to the network data
        # Then discard anything which isn't meeting the threshold
        if beta_threshold is not None and False:
            beta_data = utils.melt_and_reindex_dataframe(beta_threshold, BETA_THRESHOLD_COLUMN)
            network_data = network_data.join(beta_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])
            network_data = network_data.loc[network_data[BETA_THRESHOLD_COLUMN] == 1, :]
            del network_data[BETA_THRESHOLD_COLUMN]

        if priors is not None:
            prior_data = utils.melt_and_reindex_dataframe(priors, PRIOR_COLUMN)
            network_data = network_data.join(prior_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

        # Add any extra columns as needed
        if extra_columns is not None:
            for k in sorted(extra_columns.keys()):
                extra_data = utils.melt_and_reindex_dataframe(extra_columns[k], k)
                network_data = network_data.join(extra_data, on=[TARGET_COLUMN, REGULATOR_COLUMN])

        # Make sure all missing values are NaN
        network_data[pd.isnull(network_data)] = np.nan

        return network_data

    @staticmethod
    def threshold_and_summarize(betas, threshold):
        """
        Summarize a stack of betas
        Returns dataframes
        :param betas: list(pd.DataFrame)
            A list of dataframes that are aligned on both axes
        :param threshold: numeric
            The proportion of bootstraps an interaction must occur in to be valid
        :return betas_threshold: pd.DataFrame
        :return betas_sign: pd.DataFrame
        :return betas_non_zero: pd.DataFrame
        """
        betas_sign, betas_non_zero = ResultsProcessor.summarize(betas)
        betas_threshold = ResultsProcessor.passes_threshold(betas_non_zero, len(betas), threshold)
        return betas_threshold, betas_sign, betas_non_zero

    @staticmethod
    def summarize(betas):
        """
        Compute summary information about betas

        :param betas: list(pd.DataFrame) B x [M x N]
            A list of dataframes that are aligned on both axes
        :return betas_sign: pd.DataFrame [M x N]
            A dataframe with the summation of np.sign() for each bootstrap
        :return betas_non_zero: pd.DataFrame [M x N]
            A dataframe with a count of the number of non-zero betas for an interaction
        """

        assert check.dataframes_align(betas)

        betas_sign = pd.DataFrame(np.zeros(betas[0].shape), index=betas[0].index, columns=betas[0].columns)
        betas_non_zero = pd.DataFrame(np.zeros(betas[0].shape), index=betas[0].index, columns=betas[0].columns)
        for beta in betas:
            # Convert betas to -1,0,1 based on signing and then sum the results for each bootstrap
            betas_sign = betas_sign + np.sign(beta.values)
            # Tally all non-zeros for each bootstrap
            betas_non_zero = betas_non_zero + (beta != 0).astype(int)

        return betas_sign, betas_non_zero

    @staticmethod
    def passes_threshold(betas_non_zero, max_num, threshold):
        """

        :param betas_non_zero: pd.DataFrame [M x N]
            A dataframe of integer counts indicating how many times the original data was non-zero
        :param max_num: int
            The maximum number of possible counts (# bootstraps)
        :param threshold: float
            The proportion of integer counts over max possible in order to consider the interaction valid
        :return: pd.DataFrame [M x N]
            A bool dataframe where 1 corresponds to interactions that are in more than the threshold proportion of
            bootstraps
        """

        assert check.argument_type(betas_non_zero, pd.DataFrame)
        assert check.argument_integer(max_num)
        assert check.argument_numeric(threshold, low=0, high=1)

        return ((betas_non_zero / max_num) >= threshold).astype(int)

    @staticmethod
    def mean_and_median(stack):
        """
        Calculate the mean and median values of a list of dataframes
        Returns dataframes with the same dimensions as any one of the input stack
        :param stack: list(pd.DataFrame)
            List of dataframes which have the same size and dimensions
        :return mean_data: pd.DataFrame
            Mean values
        :return median_data:
            Median values
        """

        assert check.dataframes_align(stack)

        matrix_stack = [x.values for x in stack]
        mean_data = pd.DataFrame(np.mean(matrix_stack, axis=0), index=stack[0].index, columns=stack[0].columns)
        median_data = pd.DataFrame(np.median(matrix_stack, axis=0), index=stack[0].index, columns=stack[0].columns)
        return mean_data, median_data
