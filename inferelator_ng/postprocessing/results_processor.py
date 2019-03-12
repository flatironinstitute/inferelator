import numpy as np
import pandas as pd
import os
import csv
import matplotlib
from inferelator_ng import utils
from inferelator_ng.utils import Validator as check

matplotlib.use('pdf')
import matplotlib.pyplot as plt

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

        assert check.argument_type(pr_calc, RankSummaryPR)
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
            if conf < confidence_threshold:
                continue

            if beta_threshold is not None and not beta_threshold.ix[row_name, column_name]:
                continue

            row_data = [column_name, row_name, conf]

            # Add prior value (or nan if the priors does not cover this interaction)
            if row_name in priors.index and column_name in priors.columns:
                row_data += [priors.ix[row_name, column_name]]
            else:
                row_data += [np.nan]

            # Add gold standard, precision, and recall (or nan if the gold standard does not cover this interaction)
            if row_name in pr_calc.gold_standard.index and column_name in pr_calc.gold_standard.columns:
                row_data += [pr_calc.gold_standard.ix[row_name, column_name], precision_data.ix[row_name, column_name],
                             recall_data.ix[row_name, column_name]]
            else:
                row_data += [np.nan, np.nan, np.nan]

            if extra_columns is not None:
                for k in sorted(extra_columns.keys()):
                    if row_name in extra_columns[k].index and column_name in extra_columns[k].columns:
                        row_data += [extra_columns[k].ix[row_name, column_name]]
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


class RankSumming(object):
    """
    This class takes a data set that has some rankable values and a gold standard for which elements of that data set
    are true and calculates confidence
    """

    # Filter methods to align gold standard and confidences
    filter_method = None
    filter_method_lookup = {'overlap': 'filter_to_overlap',
                            'keep_all_gold_standard': 'filter_to_left_size'}

    # Data
    rankable_data = None
    gold_standard = None

    # Confidences
    all_confidences = None
    filtered_confidences = None

    # PR
    precision = None
    recall = None
    aupr = None

    # Ranking
    ranked_idx = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard', rank_method="sum"):

        assert check.argument_enum(filter_method, self.filter_method_lookup.keys())
        self.filter_method = getattr(self, self.filter_method_lookup[filter_method])

        # Calculate confidences based on the ranked data
        self.all_confidences = self.compute_combined_confidences(rankable_data, rank_method=rank_method)

        # Filter the gold standard and confidences down to a format that can be directly compared
        utils.Debug.vprint("GS: {gs}, Confidences: {conf}".format(gs=gold_standard.shape,
                                                                  conf=self.all_confidences.shape),
                           level=0)
        self.gold_standard, self.filtered_confidences = self.filter_method(gold_standard, self.all_confidences)
        utils.Debug.vprint("Filtered to GS: {gs}, Confidences: {conf}".format(gs=gold_standard.shape,
                                                                              conf=self.all_confidences.shape),
                           level=0)

        # Calculate the precision and recall and save the index that sorts the ranked confidences (filtered)
        self.recall, self.precision, self.ranked_idx = self.calculate_precision_recall(self.filtered_confidences,
                                                                                       self.gold_standard)
        self.aupr = self.calculate_aupr(self.recall, self.precision)


    @staticmethod
    def rank_sum(rankable_data):
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
    def rank_sum_threshold(rankable_data, data_threshold=0.9):
        """
        Calculate confidences based on ranking value in all of the data frames, discarding ranks that don't meet
        a threshold, and summing the remainder
        :param rankable_data: list(pd.DataFrame [M x N])
        :param data_threshold: float
        :return combine_conf: pd.DataFrame [M x N]
        """
        combine_conf = pd.DataFrame(np.zeros(rankable_data[0].shape),
                                    index=rankable_data[0].index,
                                    columns=rankable_data[0].columns)

        for replicate in rankable_data:
            # Flatten and rank based on the beta error reductions
            ranked_replicate = pd.DataFrame(np.reshape(pd.DataFrame(replicate.values.flatten()).rank().values,
                                                       replicate.shape),
                                            index=replicate.index,
                                            columns=replicate.columns)
            # Find the values we want to keep
            to_include = replicate >= data_threshold
            # Sum the rankings for each bootstrap
            combine_conf[to_include] += ranked_replicate[to_include]

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        max_element = max(combine_conf.values.flatten())
        combine_conf = (combine_conf - min_element) / (max_element - min_element)
        return combine_conf

    @staticmethod
    def rank_max_value(rankable_data):
        """
        Calculate confidences based on ranking the maximum value in all of the data frames
        :param rankable_data: list(pd.DataFrame [M x N])
        :return combine_conf: pd.DataFrame [M x N]
        """
        combine_conf = pd.DataFrame(np.zeros(rankable_data[0].shape),
                                    index=rankable_data[0].index,
                                    columns=rankable_data[0].columns)

        for replicate in rankable_data:
            # Max the values for each bootstrap
            combine_conf = pd.DataFrame(np.maximum(combine_conf.values, replicate.values),
                                        index=combine_conf.index,
                                        columns=combine_conf.columns)

        # Rank the maxed values
        combine_conf = pd.DataFrame(np.reshape(pd.DataFrame(combine_conf.values.flatten()).rank().values,
                                               combine_conf.shape),
                                    index=combine_conf.index,
                                    columns=combine_conf.columns)

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        max_element = max(combine_conf.values.flatten())
        combine_conf = (combine_conf - min_element) / (max_element - min_element)
        return combine_conf

    @staticmethod
    def rank_geo_mean(rankable_data):
        """
        Calculate confidences based on ranking value in all of the data frames and taking the geo mean of the ranks
        :param rankable_data: list(pd.DataFrame [M x N])
        :return combine_conf: pd.DataFrame [M x N]
        """
        combine_conf = pd.DataFrame(np.ones(rankable_data[0].shape),
                                    index=rankable_data[0].index,
                                    columns=rankable_data[0].columns)
        include_counts = pd.DataFrame(np.zeros(rankable_data[0].shape),
                                      index=rankable_data[0].index,
                                      columns=rankable_data[0].columns)

        for replicate in rankable_data:
            # Flatten and rank based on the beta error reductions
            ranked_replicate = pd.DataFrame(np.reshape(pd.DataFrame(replicate.values.flatten()).rank().values,
                                                       replicate.shape),
                                            index=replicate.index,
                                            columns=replicate.columns)
            non_zero_replicate = replicate != 0
            # Combine the rankings for each bootstrap
            combine_conf[non_zero_replicate] *= ranked_replicate[non_zero_replicate]
            include_counts += non_zero_replicate

        non_zero_all = include_counts != 0
        include_counts[non_zero_all] = 1 / include_counts[non_zero_all]
        combine_conf[~non_zero_all] = 0
        combine_conf[non_zero_all] = np.power(combine_conf[non_zero_all].values, include_counts[non_zero_all])

        # Rank the mean values
        combine_conf = pd.DataFrame(np.reshape(pd.DataFrame(combine_conf.values.flatten()).rank().values,
                                               combine_conf.shape),
                                    index=combine_conf.index,
                                    columns=combine_conf.columns)

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        combine_conf = (combine_conf - min_element) / (combine_conf.size - min_element)
        return combine_conf


class RankSummaryPR(RankSumming):
    """
    This class extends RankSumming and calculates precision-recall
    """

    def recall_precision(self):
        return self.recall, self.precision

    def plottable_recall_precision(self):
        return self.modify_pr(self.recall, self.precision)

    def dataframe_recall_precision(self):
        reverse_index = np.argsort(self.ranked_idx)
        precision = pd.DataFrame(self.precision[reverse_index].reshape(self.filtered_confidences.shape),
                                 index=self.filtered_confidences.index, columns=self.filtered_confidences.columns)
        recall = pd.DataFrame(self.recall[reverse_index].reshape(self.filtered_confidences.shape),
                              index=self.filtered_confidences.index, columns=self.filtered_confidences.columns)
        return recall, precision

    def output_pr_curve_pdf(self, output_dir, file_name="pr_curve.pdf"):
        if output_dir is None:
            return False
        else:
            recall, precision = self.modify_pr(self.recall, self.precision)
            self.plot_pr_curve(recall, precision, self.aupr, output_dir, file_name=file_name)

    def combined_confidences(self):
        return self.all_confidences

    def confidence_ordered_generator(self, threshold=None, desc=True):
        idx = self.sorted_confidence_index(threshold=threshold, desc=desc)
        num_cols = len(self.all_confidences.columns)
        for i in idx:
            row_name = self.all_confidences.index[int(i / num_cols)]
            column_name = self.all_confidences.columns[i % num_cols]
            yield row_name, column_name, self.all_confidences.ix[row_name, column_name]

    def sorted_confidence_index(self, threshold=None, desc=True):
        conf_values = self.all_confidences.values
        idx = np.argsort(conf_values, axis=None)
        if threshold is None:
            pass
        elif 1 >= threshold >= 0:
            drop_count = np.sum(conf_values.flatten() < threshold)
            if drop_count > 0:
                idx = idx[:(-1 * drop_count)]
        else:
            raise ValueError("Threshold must be between 0 and 1")

        if desc:
            return idx[::-1]
        else:
            return idx

    def num_over_precision_threshold(self, threshold):
        return np.sum(self.all_confidences.values >= self.precision_threshold(threshold), axis=None)

    def num_over_recall_threshold(self, threshold):
        return np.sum(self.all_confidences.values >= self.recall_threshold(threshold), axis=None)

    def num_over_conf_threshold(self, threshold):
        return np.sum(self.all_confidences.values >= threshold, axis=None)

    def precision_threshold(self, threshold):
        return self.find_pr_threshold(self.precision, threshold)

    def recall_threshold(self, threshold):
        return self.find_pr_threshold(self.recall, threshold)

    def find_pr_threshold(self, pr, threshold):
        if 1 >= threshold >= 0:
            threshold_index = pr > threshold
        else:
            raise ValueError("Precision/recall threshold must be between 0 and 1")

        # If there's nothing in the index return np.inf.
        if np.sum(threshold_index) == 0:
            return np.inf
        else:
            return np.min(self.filtered_confidences.values.flatten()[self.ranked_idx][threshold_index])

    @staticmethod
    def compute_combined_confidences(rankable_data, **kwargs):
        """
        Calculate combined confidences from rank sum
        :param rankable_data: list(pd.DataFrame) R x [M x N]
            List of dataframes which have the same axes and need to be rank summed
        :return combine_conf: pd.DataFrame [M x N]
        """

        rank_method = kwargs.pop("rank_method", "sum")
        assert check.argument_enum(rank_method, ("sum", "threshold_sum", "max", "geo_mean"))
        assert check.argument_type(rankable_data, list, allow_none=False)

        if rank_method == "sum":
            return RankSummaryPR.rank_sum(rankable_data)
        elif rank_method == "threshold_sum":
            return RankSummaryPR.rank_sum_threshold(rankable_data, data_threshold=kwargs.pop("data_threshold", 0.9))
        elif rank_method == "max":
            return RankSummaryPR.rank_max_value(rankable_data)
        elif rank_method == "geo_mean":
            return RankSummaryPR.rank_geo_mean(rankable_data)



    @staticmethod
    def calculate_precision_recall(conf, gold):
        # Get the index to sort the confidences
        ranked_idx = np.argsort(conf.values, axis=None)[::-1]
        gs_values = (gold.values != 0).astype(int)
        gs_values = gs_values.flatten()[ranked_idx]

        # the following mimicks the R function ChristophsPR
        precision = np.cumsum(gs_values).astype(float) / np.cumsum([1] * len(gs_values))
        recall = np.cumsum(gs_values).astype(float) / sum(gs_values)

        return recall, precision, ranked_idx

    @staticmethod
    def filter_to_left_size(left, right):
        # Find out if there are any rows or columns NOT in the left data frame
        missing_idx = left.index.difference(right.index)
        missing_col = left.columns.difference(right.columns)

        # Fill out the right dataframe with 0s
        right_filtered = pd.concat((right, pd.DataFrame(0.0, index=missing_idx, columns=right.columns)), axis=0)
        right_filtered = pd.concat((right_filtered, pd.DataFrame(0.0, index=right_filtered.index, columns=missing_col)),
                                   axis=1)

        # Return the right dataframe sized to the left
        return left, right_filtered.loc[left.index, left.columns]

    @staticmethod
    def filter_to_overlap(left, right):
        # Find out of there are any rows or columns in both data frames
        intersect_idx = right.index.intersection(left.index)
        intersect_col = right.columns.intersection(left.columns)

        # Return both dataframes sized to the overlap
        return left.loc[intersect_idx, intersect_col], right.loc[intersect_idx, intersect_col]

    @staticmethod
    def modify_pr(recall, precision):
        """
        Inserts values into the precision and recall to allow for plotting & calculations of area
        :param recall:
        :param precision:
        :return:
        """
        precision = np.insert(precision, 0, precision[0])
        recall = np.insert(recall, 0, 0)
        return recall, precision

    @staticmethod
    def calculate_aupr(recall, precision):
        recall, precision = RankSummaryPR.modify_pr(recall, precision)
        # using midpoint integration to calculate the area under the curve
        d_recall = np.diff(recall)
        m_precision = precision[:-1] + np.diff(precision) / 2
        return sum(d_recall * m_precision)

    @staticmethod
    def plot_pr_curve(recall, precision, aupr, output_dir, file_name="pr_curve.pdf"):
        if file_name is None or output_dir is None:
            return False
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = {aupr}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()
