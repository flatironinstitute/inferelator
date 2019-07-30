import pandas as pd
import numpy as np
import os
import matplotlib
from inferelator import utils
from inferelator.utils import Validator as check

matplotlib.use('pdf')
import matplotlib.pyplot as plt


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

    def auc(self):
        raise NotImplementedError

    def output_curve_pdf(self, output_dir, file_name):
        raise NotImplementedError

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
            return RankSumming.rank_sum(rankable_data)
        elif rank_method == "threshold_sum":
            return RankSumming.rank_sum_threshold(rankable_data, data_threshold=kwargs.pop("data_threshold", 0.9))
        elif rank_method == "max":
            return RankSumming.rank_max_value(rankable_data)
        elif rank_method == "geo_mean":
            return RankSumming.rank_geo_mean(rankable_data)

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


class RankSummaryPR(RankSumming):
    """
    This class extends RankSumming and calculates precision-recall
    """

    # PR
    precision = None
    recall = None
    aupr = None

    def __init__(self, rankable_data, gold_standard, filter_method='keep_all_gold_standard', rank_method="sum"):

        super(RankSummaryPR, self).__init__(rankable_data, gold_standard, filter_method=filter_method,
                                            rank_method=rank_method)

        # Calculate the precision and recall and save the index that sorts the ranked confidences (filtered)
        self.recall, self.precision, self.ranked_idx = self.calculate_precision_recall(self.filtered_confidences,
                                                                                       self.gold_standard)
        self.aupr = self.calculate_aupr(self.recall, self.precision)

    def auc(self):
        return self.aupr

    def output_curve_pdf(self, output_dir, file_name):
        self.output_pr_curve_pdf(output_dir, file_name=file_name)

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
            yield row_name, column_name, self.all_confidences.at[row_name, column_name]

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
    def plot_pr_curve(recall, precision, aupr, output_dir, file_name="pr_curve.pdf", pr_curve_file="pr_curve.tsv"):
        if file_name is None or output_dir is None:
            return False
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = {aupr}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()

        # produce tsv dataframe of recall and precision
        precision_recall = pd.DataFrame(np.column_stack([recall, precision]), columns=['recall', 'precision'])
        file_name_pr = os.path.splitext(file_name)[0] + '.tsv'
        precision_recall.to_csv(os.path.join(output_dir, file_name_pr), sep='\t', index=False)
