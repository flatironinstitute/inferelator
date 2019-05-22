import numpy as np
import pandas as pd
import os
import csv

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing.model_performance import RankSummaryPR, RankSumming

FILTER_METHODS = ("overlap", "keep_all_gold_standard")
DEFAULT_BOOTSTRAP_THRESHOLD = 0.5
DEFAULT_FILTER_METHOD = "overlap"


class ResultsProcessor:
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
    pr_curve_file_name = "pr_curve.pdf"

    def __init__(self, betas, rescaled_betas, threshold=None, filter_method=None):
        """
        :param betas: list(pd.DataFrame[G x K]) [B]
            A list of model weights per bootstrap
        :param rescaled_betas: list(pd.DataFrame[G x K]) [B]
            A list of the variance explained by each parameter per bootstrap
        :param threshold: float
            The proportion of bootstraps which an model weight must be non-zero for inclusion in the network output
        :param filter_method: str
            How to handle gold standard filtering ('overlap' filters to beta, 'keep_all_gold_standard' doesn't filter)
        """

        assert check.argument_type(betas, list)
        assert check.argument_type(betas[0], pd.DataFrame)
        assert check.dataframes_align(betas)
        self.betas = betas

        assert check.argument_type(rescaled_betas, list)
        assert check.argument_type(rescaled_betas[0], pd.DataFrame)
        assert check.dataframes_align(rescaled_betas)
        self.rescaled_betas = rescaled_betas

        assert check.argument_enum(filter_method, FILTER_METHODS, allow_none=True)
        self.filter_method = self.filter_method if filter_method is None else filter_method

        assert check.argument_numeric(threshold, 0, 1, allow_none=True)
        self.threshold = self.threshold if threshold is None else threshold

    def summarize_network(self, output_dir, gold_standard, priors):
        """
        Take the betas and rescaled beta_errors, construct a network, and test it against the gold standard
        :param output_dir: str
            Path to write files into. Don't write anything if this is None.
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard to test the network against
        :param priors: pd.DataFrame [G x K]
            Prior data
        :return aupr: float
            Returns the AUPR calculated from the network and gold standard
        """

        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(gold_standard, pd.DataFrame)
        assert check.argument_type(priors, pd.DataFrame)

        pr_calc = RankSummaryPR(self.rescaled_betas, gold_standard, filter_method=self.filter_method)
        beta_threshold, beta_sign, beta_nonzero = self.threshold_and_summarize(self.betas, self.threshold)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        extra_cols = {'beta.sign.sum': beta_sign, 'var.exp.median': resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=pr_calc.aupr), level=0)

        # Plot PR curve & Output results to a TSV
        self.network_data = self.write_output_files(pr_calc, output_dir, priors, beta_threshold, extra_cols)

        return pr_calc.aupr

    def write_output_files(self, pr_calc, output_dir, priors, beta_threshold, extra_cols, threshold_network=True):

        assert check.argument_type(pr_calc, RankSummaryPR)
        assert check.argument_path(output_dir, allow_none=True, create_if_needed=True)

        self.write_csv(pr_calc.combined_confidences(), output_dir, self.confidence_file_name)
        self.write_csv(beta_threshold, output_dir, self.threshold_file_name)
        pr_calc.output_pr_curve_pdf(output_dir, file_name=self.pr_curve_file_name)

        # Threshold the network with the boolean beta_threshold if threshold_network is True
        beta_threshold = beta_threshold if threshold_network else None

        # Process data into a network dataframe, write it out, and return it
        network_data = self.process_network(pr_calc, priors, beta_threshold=beta_threshold, extra_columns=extra_cols)
        self.save_network_to_tsv(network_data, output_dir, output_file_name=self.network_file_name)
        return network_data

    @staticmethod
    def save_network_to_tsv(network_data, output_dir, output_file_name="network.tsv"):
        """
        Create a network file and save it
        :param network_data: pd.DataFrame
            The network as an edge dataframe
        :param output_dir: str
            The path to the output file. If None, don't save anything
        :param output_file_name: str
            The output file name. If None, don't save anything

        """

        assert check.argument_type(network_data, pd.DataFrame)
        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(output_file_name, str, allow_none=True)

        # Write output
        if output_dir is not None and output_file_name is not None:
            network_data.to_csv(os.path.join(output_dir, output_file_name), sep="\t", index=False, header=True)

    @staticmethod
    def process_network(pr_calc, priors, confidence_threshold=0, beta_threshold=None, extra_columns=None):
        """
        Process rank-summed results into a network data frame
        :param pr_calc: RankSummaryPR
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

        assert check.argument_type(pr_calc, RankSumming)
        assert check.argument_type(priors, pd.DataFrame, allow_none=True)
        assert check.argument_type(beta_threshold, pd.DataFrame, allow_none=True)
        assert check.argument_numeric(confidence_threshold, 0, 1)

        recall_data, precision_data = pr_calc.dataframe_recall_precision()

        # Get the combined confidences in order, convert them to a dataframe, and subset for confidence threshold
        network_data = list(pr_calc.confidence_ordered_generator())
        network_data = pd.DataFrame(network_data, columns=['target', 'regulator', 'combined_confidences'])
        network_data = network_data.loc[network_data['combined_confidences'] > confidence_threshold,
                                        ['regulator', 'target', 'combined_confidences']]

        # If beta_threshold has been provided, melt and join it to the network data
        # Then discard anything which isn't meeting the threshold
        if beta_threshold is not None and False:
            beta_data = ResultsProcessor.melt_and_reindex_dataframe(beta_threshold, 'beta_threshold')
            network_data = network_data.join(beta_data, on=["target", "regulator"])
            network_data = network_data.loc[network_data['beta_threshold'] == 1, :]
            del network_data['beta_threshold']

        # Convert each column's data to a dataframe with a multiindex
        gold_data = ResultsProcessor.melt_and_reindex_dataframe(pr_calc.gold_standard, "gold_standard")
        recall_data = ResultsProcessor.melt_and_reindex_dataframe(recall_data, "recall")
        precision_data = ResultsProcessor.melt_and_reindex_dataframe(precision_data, "precision")

        if priors is not None:
            prior_data = ResultsProcessor.melt_and_reindex_dataframe(priors, "prior")
            network_data = network_data.join(prior_data, on=["target", "regulator"])

        # Join each column's data to the network edges
        network_data = network_data.join(gold_data, on=["target", "regulator"])
        network_data = network_data.join(precision_data, on=["target", "regulator"])
        network_data = network_data.join(recall_data, on=["target", "regulator"])

        # Add any extra columns as needed
        if extra_columns is not None:
            for k in sorted(extra_columns.keys()):
                extra_data = ResultsProcessor.melt_and_reindex_dataframe(extra_columns[k], k)
                network_data = network_data.join(extra_data, on=["target", "regulator"])

        # Make sure all missing values are NaN
        network_data[pd.isnull(network_data)] = np.nan

        return network_data

    @staticmethod
    def melt_and_reindex_dataframe(data_frame, value_name, idx_name="target", col_name="regulator"):
        """
        Take a pandas dataframe and melt it into a one column dataframe (with the column `value_name`) and a multiindex
        of the original index + column
        :param data_frame: pd.DataFrame [M x N]
            Meltable dataframe
        :param value_name: str
            The column name for the values of the dataframe
        :param idx_name: str
            The name to assign to the original data_frame index values
        :param col_name: str
            The name to assign to the original data_frame column values
        :return: pd.DataFrame [(M*N) x 1]
            Melted dataframe with a single column of values and a multiindex that is the original index + column for
            that value
        """

        assert check.argument_type(data_frame, pd.DataFrame)

        # Copy the dataframe and move the index to a column
        data_frame = data_frame.copy()
        data_frame[idx_name] = data_frame.index

        # Melt it into a [(M*N) x 3] dataframe
        data_frame = data_frame.melt(id_vars=idx_name, var_name=col_name, value_name=value_name)

        # Create a multiindex and then drop the columns that are now in the index
        data_frame.index = pd.MultiIndex.from_frame(data_frame.loc[:, [idx_name, col_name]])
        del data_frame[idx_name]
        del data_frame[col_name]

        return data_frame

    @staticmethod
    def write_csv(data, pathname, filename):
        assert check.argument_path(pathname, allow_none=True)
        assert check.argument_type(filename, str, allow_none=True)
        assert check.argument_type(data, pd.DataFrame)

        if pathname is not None and filename is not None:
            data.to_csv(os.path.join(pathname, filename), sep='\t')

    @staticmethod
    def threshold_and_summarize(betas, threshold):
        betas_sign, betas_non_zero = ResultsProcessor.summarize(betas)
        betas_threshold = ResultsProcessor.passes_threshold(betas_non_zero, len(betas), threshold)
        return betas_threshold, betas_sign, betas_non_zero

    @staticmethod
    def summarize(betas):
        """
        Compute summary information about betas

        :param betas: list(pd.DataFrame) B x [M x N]
            A dataframe with the original data
        :return betas_sign: pd.DataFrame [M x N]
            A dataframe with the summation of np.sign() for each bootstrap
        :return betas_non_zero: pd.DataFrame [M x N]
            A dataframe with a count of the number of non-zero betas for an interaction
        """
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
        return ((betas_non_zero / max_num) >= threshold).astype(int)

    @staticmethod
    def mean_and_median(stack):
        matrix_stack = [x.values for x in stack]
        mean_data = pd.DataFrame(np.mean(matrix_stack, axis=0), index=stack[0].index, columns=stack[0].columns)
        median_data = pd.DataFrame(np.median(matrix_stack, axis=0), index=stack[0].index, columns=stack[0].columns)
        return mean_data, median_data
