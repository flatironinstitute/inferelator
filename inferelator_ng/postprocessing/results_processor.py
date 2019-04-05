import numpy as np
import pandas as pd
import os
import csv

from inferelator_ng import utils
from inferelator_ng.utils import Validator as check
from inferelator_ng.postprocessing.model_performance import RankSummaryPR, RankSumming

FILTER_METHODS = ("overlap", "keep_all_gold_standard")


class ResultsProcessor:
    # Data
    betas = None
    rescaled_betas = None
    filter_method = None

    # Cutoffs
    threshold = None

    # File names
    network_file_name = "network.tsv"
    confidence_file_name = "combined_confidences.tsv"
    threshold_file_name = "betas_stack.tsv"
    pr_curve_file_name = "pr_curve.pdf"

    def __init__(self, betas, rescaled_betas, threshold=0.5, filter_method='overlap'):
        """
        :param betas: list(pd.DataFrame[G x K])
        :param rescaled_betas: list(pd.DataFrame[G x K])
        :param threshold: float
        :param filter_method: str
            How to handle gold standard filtering ('overlap' filters to beta, 'keep_all_gold_standard' doesn't filter)
        """

        assert check.dataframes_align(betas)
        self.betas = betas

        assert check.dataframes_align(rescaled_betas)
        self.rescaled_betas = rescaled_betas

        assert check.argument_enum(filter_method, FILTER_METHODS)
        self.filter_method = filter_method

        assert check.argument_numeric(threshold, 0, 1)
        self.threshold = threshold

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
        beta_sign, beta_nonzero = self.summarize(self.betas)
        beta_threshold = self.passes_threshold(beta_nonzero, len(self.betas), self.threshold)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        network_data = {'beta.sign.sum': beta_sign, 'var.exp.median': resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=pr_calc.aupr), level=0)

        # Plot PR curve & Output results to a TSV
        self.write_output_files(pr_calc, output_dir, priors, beta_threshold, network_data)

        return pr_calc.aupr

    def write_output_files(self, pr_calc, output_dir, priors, beta_threshold, network_data, threshold_network=True):

        assert check.argument_type(pr_calc, RankSummaryPR)
        assert check.argument_path(output_dir, allow_none=True, create_if_needed=True)

        self.write_csv(pr_calc.combined_confidences(), output_dir, self.confidence_file_name)
        self.write_csv(beta_threshold, output_dir, self.threshold_file_name)
        pr_calc.output_pr_curve_pdf(output_dir, file_name=self.pr_curve_file_name)

        # Threshold the network with the boolean beta_threshold if threshold_network is True
        beta_threshold = beta_threshold if threshold_network else None

        # Write output
        self.save_network_to_tsv(pr_calc, priors, output_dir, output_file_name=self.network_file_name,
                                 beta_threshold=beta_threshold, extra_columns=network_data)

    @staticmethod
    def save_network_to_tsv(pr_calc, priors, output_dir, confidence_threshold=0, output_file_name="network.tsv",
                            beta_threshold=None, extra_columns=None):
        """
        Create a network file and save it
        :param pr_calc: RankSummaryPR
            The rank-sum object with the math in it
        :param priors: pd.DataFrame [G x K]
            Prior data
        :param output_dir: str
            The path to the output file. If None, don't save anything
        :param confidence_threshold: numeric
            The minimum confidence score needed to write a network edge
        :param output_file_name: str
            The output file name. If None, don't save anything
        :param beta_threshold: pd.DataFrame [G x K]
            The thresholded betas to include in the network. If None, include everything.
        :param extra_columns: dict(col_name: pd.DataFrame [G x K])
            Any additional data to include, keyed by column name and indexable with row and column names
        """

        assert check.argument_type(pr_calc, RankSumming)
        assert check.argument_type(priors, pd.DataFrame)
        assert check.argument_type(beta_threshold, pd.DataFrame, allow_none=True)
        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(output_file_name, str, allow_none=True)
        assert check.argument_numeric(confidence_threshold, 0, 1)

        if output_dir is None or output_file_name is None:
            return False

        header = ['regulator', 'target', 'combined_confidences', 'prior', 'gold_standard', 'precision', 'recall']
        if extra_columns is not None:
            header += [k for k in sorted(extra_columns.keys())]

        output_list = [header]

        recall_data, precision_data = pr_calc.dataframe_recall_precision()

        for row_name, column_name, conf in pr_calc.confidence_ordered_generator():
            if conf <= confidence_threshold:
                continue

            if beta_threshold is not None and not beta_threshold.at[row_name, column_name]:
                continue

            row_data = [column_name, row_name, conf]

            # Add prior value (or nan if the priors does not cover this interaction)
            if row_name in priors.index and column_name in priors.columns:
                row_data += [priors.at[row_name, column_name]]
            else:
                row_data += [np.nan]

            # Add gold standard, precision, and recall (or nan if the gold standard does not cover this interaction)
            if row_name in pr_calc.gold_standard.index and column_name in pr_calc.gold_standard.columns:
                row_data += [pr_calc.gold_standard.at[row_name, column_name], precision_data.at[row_name, column_name],
                             recall_data.at[row_name, column_name]]
            else:
                row_data += [np.nan, np.nan, np.nan]

            if extra_columns is not None:
                for k in sorted(extra_columns.keys()):
                    if row_name in extra_columns[k].index and column_name in extra_columns[k].columns:
                        row_data += [extra_columns[k].at[row_name, column_name]]
                    else:
                        row_data += [np.nan]

            output_list.append(row_data)

        with open(os.path.join(output_dir, output_file_name), 'w') as myfile:
            wr = csv.writer(myfile, delimiter='\t')
            for row in output_list:
                wr.writerow(row)

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
