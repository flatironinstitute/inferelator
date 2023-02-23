import warnings

import numpy as np
import pandas as pd

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_metrics import (
    RankSummingMetric,
    MetricHandler
)

from inferelator.postprocessing.inferelator_results import InferelatorResults
from inferelator.postprocessing import (
    BETA_SIGN_COLUMN,
    MEDIAN_EXPLAIN_VAR_COLUMN,
    CONFIDENCE_COLUMN,
    TARGET_COLUMN,
    REGULATOR_COLUMN,
    PRIOR_COLUMN,
    MODEL_COEF_COLUMN,
    MODEL_EXP_VAR_COLUMN
)

FILTER_METHODS = ("overlap", "keep_all_gold_standard")
DEFAULT_FILTER_METHOD = "overlap"
DEFAULT_METRIC = "precision-recall"


class ResultsProcessor:

    # Data
    betas = None
    rescaled_betas = None
    filter_method = DEFAULT_FILTER_METHOD

    # Processed Network
    network_data = None

    # Flag to write results
    write_results = True

    # Model result object
    result_object = InferelatorResults

    # Model metric
    metric = None

    def __init__(
        self,
        betas,
        rescaled_betas,
        filter_method=None,
        metric=None
    ):
        """
        :param betas: A list of dataframes [G x K] with model
            weights per bootstrap
        :type betas: list(pd.DataFrame)
        :param rescaled_betas: A list of dataframes [G x K] with
            the variance explained by each parameter per bootstrap
        :type rescaled_betas: list(pd.DataFrame)
        :param filter_method: How to handle gold standard filtering.
            'overlap' filters to beta
            'keep_all_gold_standard' doesn't filter and uses the entire
            gold standard scoring network
        :type filter_method: str
        :param metric: The scoring metric to use
        :type metric: str, RankSummingMetric
        """

        self.validate_init_args(
            betas,
            rescaled_betas,
            filter_method=filter_method
        )

        self.betas = betas
        self.rescaled_betas = rescaled_betas

        if filter_method is not None:
            self.filter_method = filter_method

        self.metric = MetricHandler.get_metric(
            metric if metric is not None else DEFAULT_METRIC
        )

    @staticmethod
    def validate_init_args(
        betas,
        rescaled_betas,
        filter_method=None
    ):
        assert check.argument_type(betas, list)
        assert check.argument_type(betas[0], pd.DataFrame)
        assert check.dataframes_align(betas)

        assert check.argument_type(rescaled_betas, list)
        assert check.argument_type(rescaled_betas[0], pd.DataFrame)
        assert check.dataframes_align(rescaled_betas)

        assert check.argument_enum(
            filter_method,
            FILTER_METHODS,
            allow_none=True
        )

    def summarize_network(
        self,
        output_dir,
        gold_standard,
        priors,
        full_model_betas=None,
        full_model_var_exp=None,
        extra_cols=None
    ):
        """
        Take the betas and rescaled beta_errors, construct a network,
        and test it against the gold standard

        :param output_dir: Path to write files into.
            Don't write anything if this is None.
        :type output_dir: str, None
        :param gold_standard: Gold standard DataFrame [G x K] to test
            the network against
        :type gold_standard: pd.DataFrame
        :param priors: Priors dataframe [G x K]
        :type priors: pd.DataFrame
        :param extra_cols: Extra columns to add to the network dataframe,
            as a dict of G x K dataframes keyed by column name
        :return result: Result object
        :rtype: InferelatorResult
        """

        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(gold_standard, pd.DataFrame)
        assert check.argument_type(priors, pd.DataFrame)

        rs_calc = self.metric(
            self.rescaled_betas,
            gold_standard,
            filter_method=self.filter_method
        )

        beta_sign, beta_nonzero = self.summarize(
            self.betas
        )

        resc_betas_mean, resc_betas_median = self.mean_and_median(
            self.rescaled_betas
        )

        if extra_cols is None:
            extra_cols = {}

        extra_cols.update({
            BETA_SIGN_COLUMN: beta_sign,
            MEDIAN_EXPLAIN_VAR_COLUMN: resc_betas_median
        })

        m_name, score = rs_calc.score()

        utils.Debug.vprint(
            f"Model {m_name}:\t{score:.05f}",
            level=0
        )

        # Process data into a network dataframe
        network_data = self.process_network(
            rs_calc,
            priors,
            extra_columns=extra_cols,
            full_model_betas=full_model_betas,
            full_model_var_exp=full_model_var_exp
        )

        # Create a InferelatorResult object and have it write output files
        result = self.result_object(
            network_data,
            full_model_betas,
            rs_calc.all_confidences,
            rs_calc,
            betas_sign=beta_sign,
            betas=self.betas,
            priors=priors,
            gold_standard=gold_standard
        )

        if self.write_results and output_dir is not None:
            result.write_result_files(output_dir)

        return result

    @staticmethod
    def process_network(
        metric,
        priors,
        full_model_betas=None,
        full_model_var_exp=None,
        confidence_threshold=0,
        beta_threshold=None,
        extra_columns=None
    ):
        """
        Process rank-summed results into a network data frame

        :param metric: The rank-sum object with the math in it
        :type metric: RankSummingMetric
        :param priors: Prior data [G x K]
        :type priors: pd.DataFrame
        :param confidence_threshold: The minimum confidence score needed
            to write a network edge
        :type confidence_threshold: numeric
        :param beta_threshold: Deprecated
        :param extra_columns: Any additional data to include, keyed by
            column name and indexable with row and column names
        :type extra_columns:  dict(col_name: pd.DataFrame [G x K])
        :return network_data: Network edge dataframe [(G*K) x 7+]
        :rtype: pd.DataFrame

        """

        assert check.argument_type(metric, RankSummingMetric)
        assert check.argument_type(priors, pd.DataFrame, allow_none=True)
        assert check.argument_numeric(confidence_threshold, 0, 1)

        if beta_threshold is not None:
            warnings.warn(
                "beta_threshold is deprecated and has no effect",
                DeprecationWarning
            )

        # Get the combined confidences and subset for confidence threshold
        network_data = metric.confidence_dataframe()
        network_data = network_data.loc[
            network_data[CONFIDENCE_COLUMN] > confidence_threshold,
            :
        ]

        if priors is not None:
            network_data = network_data.join(
                utils.melt_and_reindex_dataframe(
                    priors,
                    PRIOR_COLUMN
                ),
                on=[TARGET_COLUMN, REGULATOR_COLUMN]
            )

        if full_model_betas is not None:
            network_data = network_data.join(
                utils.melt_and_reindex_dataframe(
                    full_model_betas,
                    MODEL_COEF_COLUMN
                ),
                on=[TARGET_COLUMN, REGULATOR_COLUMN]
            )

        if full_model_var_exp is not None:
            network_data = network_data.join(
                utils.melt_and_reindex_dataframe(
                    full_model_var_exp,
                    MODEL_EXP_VAR_COLUMN
                ),
                on=[TARGET_COLUMN, REGULATOR_COLUMN]
            )

        # Add any extra columns as needed
        if extra_columns is not None:
            for k in sorted(extra_columns.keys()):
                network_data = network_data.join(
                    utils.melt_and_reindex_dataframe(extra_columns[k], k),
                    on=[TARGET_COLUMN, REGULATOR_COLUMN]
                )

        # Make sure all missing values are NaN
        network_data[pd.isnull(network_data)] = np.nan

        return network_data

    @staticmethod
    def summarize(betas):
        """
        Compute summary information about betas

        :param betas: A list of dataframes B x [G x K] that are aligned
            on both axes
        :type betas: list(pd.DataFrame)
        :return: A dataframe [G x K] with the summation of np.sign() for
            each bootstrap, and a dataframe with a count of the number of
            non-zero betas for an interaction
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.dataframes_align(betas)

        betas_sign = pd.DataFrame(
            np.zeros(betas[0].shape),
            index=betas[0].index,
            columns=betas[0].columns
        )

        betas_non_zero = pd.DataFrame(
            np.zeros(betas[0].shape),
            index=betas[0].index,
            columns=betas[0].columns
        )

        for beta in betas:
            # Convert betas to -1,0,1 based on signing
            # and then sum the results for each bootstrap
            betas_sign += np.sign(beta.values)

            # Tally all non-zeros for each bootstrap
            betas_non_zero += (beta != 0).astype(int)

        return betas_sign, betas_non_zero

    @staticmethod
    def mean_and_median(stack):
        """
        Calculate the mean and median values of a list of dataframes
        Returns dataframes with the same dimensions as any one of
        the input stack

        :param stack: list(pd.DataFrame)
            List of dataframes which have the same size and dimensions
        :return mean_data: pd.DataFrame
            Mean values
        :return median_data:
            Median values
        """

        assert check.dataframes_align(stack)

        matrix_stack = [x.values for x in stack]

        mean_data = pd.DataFrame(
            np.mean(matrix_stack, axis=0),
            index=stack[0].index,
            columns=stack[0].columns
        )

        median_data = pd.DataFrame(
            np.median(matrix_stack, axis=0),
            index=stack[0].index,
            columns=stack[0].columns
        )

        return mean_data, median_data
