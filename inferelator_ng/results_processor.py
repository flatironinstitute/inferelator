import numpy as np
import pandas as pd
import os
import csv
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

from inferelator_ng import utils


class ResultsProcessor:
    filter_method_lookup = {'overlap': 'filter_to_overlap',
                            'keep_all_gold_standard': 'filter_to_left_size'}
    filter_method = None

    def __init__(self, betas, rescaled_betas, threshold=0.5, filter_method='overlap'):
        self.betas = betas
        self.rescaled_betas = rescaled_betas

        if 1 >= threshold >= 0:
            self.threshold = threshold
        else:
            raise ValueError("Threshold must be a float in the interval [0, 1]")
        try:
            self.filter_method = getattr(self, self.filter_method_lookup[filter_method])
        except KeyError:
            raise ValueError("{val} is not an allowed filter_method option".format(val=filter_method))

    def calculate_precision_recall(self, conf, gold):
        # Filter down to stuff that we have anything in the gold standard for
        gold = np.abs(gold.loc[(gold != 0).any(axis=1), (gold != 0).any(axis=0)])

        utils.Debug.vprint("GS: {gs}, Confidences: {conf}".format(gs=gold.shape, conf=conf.shape), level=0)
        gold, conf = self.filter_method(gold, conf)
        utils.Debug.vprint("Filtered to GS: {gs}, Confidences: {conf}".format(gs=gold.shape, conf=conf.shape), level=0)

        # Get the index to sort the confidences
        self.conf_sort_idx = np.argsort(conf.values, axis=None)[::-1]
        gs_values = gold.values.flatten()[self.conf_sort_idx]

        # Save sorted confidences for later
        self.sorted_pr_confidences = conf.values.flatten()[self.conf_sort_idx]

        # the following mimicks the R function ChristophsPR
        self.precision = np.cumsum(gs_values).astype(float) / np.cumsum([1] * len(gs_values))
        self.recall = np.cumsum(gs_values).astype(float) / sum(gs_values)

        # Insert values to make it plot cleanly
        precision = np.insert(self.precision, 0, self.precision[0])
        recall = np.insert(self.recall, 0, 0)

        return recall, precision

    def save_network_to_tsv(self, combined_confidences, betas_sign, resc_betas_median, priors, gold_standard,
                            output_dir, output_file_name="network.tsv", conf_threshold=0):

        if output_dir is None:
            return

        output_list = [
            ['regulator', 'target', 'beta.sign.sum', 'var.exp.median', 'combined_confidences',
             'prior', 'gold.standard', 'precision', 'recall']]

        sorted_by_confidence = np.argsort(combined_confidences.values, axis=None)[::-1]
        num_cols = len(combined_confidences.columns)

        reverse_index = np.argsort(self.conf_sort_idx)
        precision_data = pd.DataFrame(self.precision[reverse_index].reshape(gold_standard.shape),
                                      index=gold_standard.index, columns=gold_standard.columns)
        recall_data = pd.DataFrame(self.recall[reverse_index].reshape(gold_standard.shape),
                                      index=gold_standard.index, columns=gold_standard.columns)

        for i in sorted_by_confidence:
            row_data = []
            # Since this was sorted using a flattened index, we need to reconvert into labeled 2d index
            row_name = combined_confidences.index[int(i / num_cols)]
            column_name = combined_confidences.columns[i % num_cols]
            comb_conf = combined_confidences.ix[row_name, column_name]

            # Add interactor names, beta_sign, median_beta, and combined_confidence
            row_data += [column_name, row_name, betas_sign.ix[row_name, column_name],
                         resc_betas_median.ix[row_name, column_name], comb_conf]

            # Add prior value (or nan if the priors does not cover this interaction)
            if row_name in priors.index and column_name in priors.columns:
                row_data += [priors.ix[row_name, column_name]]
            else:
                row_data += [np.nan]

            # Add gold standard, precision, and recall (or nan if the gold standard does not cover this interaction)
            if row_name in gold_standard.index and column_name in gold_standard.columns:
                row_data += [gold_standard.ix[row_name, column_name], precision_data.ix[row_name, column_name],
                             recall_data.ix[row_name, column_name]]
            else:
                row_data += [np.nan, np.nan, np.nan]

            if comb_conf > conf_threshold:
                output_list.append(row_data)

        with open(os.path.join(output_dir, output_file_name), 'w') as myfile:
            wr = csv.writer(myfile, delimiter='\t')
            for row in output_list:
                wr.writerow(row)

    def summarize_network(self, output_dir, gold_standard, priors):
        self.combined_confidences = self.compute_combined_confidences(self.rescaled_betas)
        self.beta_threshold, self.betas_sign, self.beta_nonzero = self.threshold_and_summarize(self.betas,
                                                                                               self.threshold)

        # Output results to a TSV
        self.write_csv(self.combined_confidences, output_dir, 'combined_confidences.tsv')
        self.write_csv(self.beta_threshold, output_dir, 'betas_stack.tsv')

        # Calculate precision & recall
        recall, precision = self.calculate_precision_recall(self.combined_confidences, gold_standard)
        aupr = self.calculate_aupr(recall, precision)
        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=aupr), level=0)

        # Plot PR curve
        self.plot_pr_curve(recall, precision, aupr, output_dir)


        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        self.save_network_to_tsv(self.combined_confidences, self.betas_sign, resc_betas_median, priors, gold_standard,
                                 output_dir)
        return aupr

    def find_conf_threshold(self, precision_threshold=None, recall_threshold=None):
        """
        Determine the combined confidence at a precision or a recall threshold
        :param precision_threshold: float
        :param recall_threshold: float
        :return: float
            Confidence value threshold
        """

        if precision_threshold is None and recall_threshold is None:
            raise ValueError("Set precision or recall")
        if precision_threshold is not None and recall_threshold is not None:
            raise ValueError("Set precision or recall. Not both.")

        if precision_threshold is not None:
            if 1 >= precision_threshold >= 0:
                threshold_index = self.precision > precision_threshold
            else:
                raise ValueError("Precision must be between 0 and 1")

        if recall_threshold is not None:
            if 1 >= recall_threshold >= 0:
                threshold_index = self.recall > recall_threshold
            else:
                raise ValueError("Recall must be between 0 and 1")

        # If there's nothing in the index return 2. Which might as well be np.inf.
        if np.sum(threshold_index) == 0:
            return 2.0
        else:
            return np.min(self.sorted_pr_confidences[threshold_index])

    @staticmethod
    def write_csv(data, pathname, filename):
        if pathname is not None and filename is not None:
            data.to_csv(os.path.join(pathname, filename), sep='\t')


    @staticmethod
    def compute_combined_confidences(rescaled_betas):
        """
        Calculate combined confidences based on the rescaled betas
        :param rescaled_betas: list(pd.DataFrame) B x [G x K]
            List of beta_resc dataframes (each dataframe is the result of one bootstrap run)
        :return combine_conf: pd.DataFrame [G x K]
        """

        # Create an 0s dataframe shaped to the data to be ranked
        combine_conf = pd.DataFrame(np.zeros(rescaled_betas[0].shape),
                                    index=rescaled_betas[0].index,
                                    columns=rescaled_betas[0].columns)

        for beta_resc in rescaled_betas:
            # Flatten and rank based on the beta error reductions
            ranked_df = np.reshape(pd.DataFrame(beta_resc.values.flatten()).rank(method="average").values,
                                   rescaled_betas[0].shape)
            # Sum the rankings for each bootstrap
            combine_conf = combine_conf + ranked_df

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        combine_conf = (combine_conf - min_element) / (len(rescaled_betas) * combine_conf.size - min_element)
        return combine_conf

    @staticmethod
    def threshold_and_summarize(betas, threshold):
        """
        Compute summary information about betas

        :param betas: list(pd.DataFrame) B x [G x K]
        :param threshold: float
            The proportion of bootstraps (0 to 1)
        :return thresholded_matrix: pd.DataFrame [G x K]
            A bool dataframe where 1 corresponds to interactions that are in more than the threshold proportion of
            bootstraps
        :return betas_sign: pd.DataFrame [G x K]
            A dataframe with the summation of np.sign() for each bootstrap
        :return betas_non_zero: pd.DataFrame [G x K]
            A dataframe with a count of the number of non-zero betas for an interaction
        """
        betas_sign = pd.DataFrame(np.zeros(betas[0].shape), index=betas[0].index, columns=betas[0].columns)
        betas_non_zero = pd.DataFrame(np.zeros(betas[0].shape), index=betas[0].index, columns=betas[0].columns)
        for beta in betas:
            # Convert betas to -1,0,1 based on signing and then sum the results for each bootstrap
            betas_sign = betas_sign + np.sign(beta.values)
            # Tally all non-zeros for each bootstrap
            betas_non_zero = betas_non_zero + (beta != 0).astype(int)

        # The following line returns 1 for all entries that appear in more than (or equal to) self.threshold fraction of
        # bootstraps and 0 otherwise

        thresholded_matrix = ((betas_non_zero / len(betas)) >= threshold).astype(int)
        # Note that the current version is blind to the sign of those betas, so the betas_sign matrix is not used. Later
        # we might want to modify this such that only same-sign interactions count.
        return thresholded_matrix, betas_sign, betas_non_zero

    @staticmethod
    def plot_pr_curve(recall, precision, aupr, output_dir):
        if output_dir is None:
            return
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = {aupr}".format(aupr=aupr), xy=(0.4, 0.05), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, 'pr_curve.pdf'))
        plt.close()

    @staticmethod
    def calculate_aupr(recall, precision):
        # using midpoint integration to calculate the area under the curve
        d_recall = np.diff(recall)
        m_precision = precision[:-1] + np.diff(precision) / 2
        return sum(d_recall * m_precision)

    @staticmethod
    def mean_and_median(stack):
        matrix_stack = [x.values for x in stack]
        mean_data = pd.DataFrame(np.mean(matrix_stack, axis=0), index = stack[0].index, columns = stack[0].columns)
        median_data = pd.DataFrame(np.median(matrix_stack, axis=0), index = stack[0].index, columns = stack[0].columns)
        return mean_data, median_data

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
