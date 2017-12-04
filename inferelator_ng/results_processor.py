import numpy as np
import pandas as pd
import os
import csv
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt



class ResultsProcessor:

    def __init__(self, betas, rescaled_betas, threshold=0.5):
        self.betas = betas
        self.rescaled_betas = rescaled_betas
        self.threshold = threshold

    def compute_combined_confidences(self):
        combined_confidences = pd.DataFrame(np.zeros((self.betas[0].shape)), index = self.betas[0].index, columns = self.betas[0].columns )
        for beta_resc in self.rescaled_betas:
            # this ranking code is especially wordy because the rank function only works in one dimension (col or row), so I had to flatten the matrix
            ranked_df =np.reshape(pd.DataFrame(np.ndarray.flatten(beta_resc.values)).rank( method = "average").values, self.rescaled_betas[0].shape)
            combined_confidences = combined_confidences + ranked_df

        min_element = min(combined_confidences.min())
        combined_confidences = (combined_confidences - min_element) / (len(self.betas) * combined_confidences.size - min_element)
        return combined_confidences

    def threshold_and_summarize(self):
        self.betas_sign = pd.DataFrame(np.zeros((self.betas[0].shape)), index = self.betas[0].index, columns = self.betas[0].columns )
        self.betas_non_zero = pd.DataFrame(np.zeros((self.betas[0].shape)), index = self.betas[0].index, columns = self.betas[0].columns )
        for beta in self.betas:
            self.betas_sign = self.betas_sign + np.sign(beta.values)
            self.betas_non_zero = self.betas_non_zero + np.absolute(np.sign(beta.values))
     
        #The following line returns 1 for all entries that appear in more than (or equal to) self.threshold fraction of bootstraps and 0 otherwise
        thresholded_matrix = ((self.betas_non_zero / len(self.betas)) >= self.threshold).astype(int)
        #Note that the current version is blind to the sign of those betas, so the betas_sign matrix is not used. Later we might want to modify this such that only same-sign interactions count.
        return thresholded_matrix

    def calculate_precision_recall(self, combined_confidences, gold_standard):
        # this code only runs for a positive gold standard, so explicitly transform it using the absolute value: 
        gold_standard = np.abs(gold_standard)
        # filter gold standard
        gold_standard_nozero = gold_standard.loc[(gold_standard!=0).any(axis=1), (gold_standard!=0).any(axis=0)]
        intersect_index = combined_confidences.index.intersection(gold_standard_nozero.index)
        intersect_cols = combined_confidences.columns.intersection(gold_standard_nozero.columns)
        gold_standard_filtered = gold_standard_nozero.loc[intersect_index, intersect_cols]
        combined_confidences_filtered = combined_confidences.loc[intersect_index, intersect_cols]
        # rank from highest to lowest confidence
        sorted_candidates = np.argsort(combined_confidences_filtered.values, axis = None)[::-1]
        gs_values = gold_standard_filtered.values.flatten()[sorted_candidates]
        #the following mimicks the R function ChristophsPR
        precision = np.cumsum(gs_values).astype(float) / np.cumsum([1] * len(gs_values))
        recall = np.cumsum(gs_values).astype(float) / sum(gs_values)
        precision = np.insert(precision,0,precision[0])
        recall = np.insert(recall,0,0)
        return (recall, precision)

    def calculate_aupr(self, recall, precision):
        #using midpoint integration to calculate the area under the curve
        d_recall = np.diff(recall)
        m_precision = precision[:-1] + np.diff(precision) / 2
        return sum(d_recall * m_precision)


    def plot_pr_curve(self, recall, precision, aupr, output_dir):
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = " + aupr.astype("string"), xy=(0.4, 0.05), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, 'pr_curve.pdf'))
        plt.close()

    def mean_and_median(self, stack):
        matrix_stack = [x.values for x in stack]
        mean = np.mean(matrix_stack, axis = 0)
        median = np.median(matrix_stack, axis = 0)
        return (mean, median)

    def save_network_to_tsv(self,combined_confidences, resc_betas_median, priors, output_dir):
        output_list = [['regulator', 'target', 'beta.sign.sum', 'beta.non.zero', 'var.exp.median', 'combined_confidences', 'prior']]
        sorted_by_confidence = np.argsort(combined_confidences.values, axis = None)[::-1]
        num_cols = len(combined_confidences.columns)
        for i in sorted_by_confidence:
            # Since this was sorted using a flattened index, we need to reconvert into labeled 2d index
            index_idx = i / num_cols
            column_idx = i % num_cols
            row_name = combined_confidences.index[index_idx]   
            column_name = combined_confidences.columns[column_idx]
            if row_name in priors.index:
                prior_value = priors.ix[row_name, column_name]
            else:
                prior_value = np.nan
            if (combined_confidences.ix[row_name, column_name] > 0):
                output_list.append([column_name, row_name, self.betas_sign.ix[row_name, column_name],
                    self.betas_non_zero.ix[row_name, column_name], resc_betas_median[index_idx, column_idx],
                    combined_confidences.ix[row_name, column_name],prior_value])
        with open(os.path.join(output_dir, 'network.tsv'), 'w') as myfile:
            wr = csv.writer(myfile,  delimiter = '\t')
            for row in output_list:
                wr.writerow(row)

    def summarize_network(self, output_dir, gold_standard, priors):
        combined_confidences = self.compute_combined_confidences()
        betas_stack = self.threshold_and_summarize()
        combined_confidences.to_csv(os.path.join(output_dir, 'combined_confidences.tsv'), sep = '\t')
        betas_stack.to_csv(os.path.join(output_dir,'betas_stack.tsv'), sep = '\t')
        (recall, precision) = self.calculate_precision_recall(combined_confidences, gold_standard)
        aupr = self.calculate_aupr(recall, precision)
        self.plot_pr_curve(recall, precision, aupr, output_dir)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        self.save_network_to_tsv(combined_confidences, resc_betas_median, priors, output_dir)
        