import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import condition
import os

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
        betas_sign = pd.DataFrame(np.zeros((self.betas[0].shape)), index = self.betas[0].index, columns = self.betas[0].columns )
        betas_non_zero = pd.DataFrame(np.zeros((self.betas[0].shape)), index = self.betas[0].index, columns = self.betas[0].columns )
        for beta in self.betas:
            betas_sign = betas_sign + np.sign(beta.values)
            betas_non_zero = betas_non_zero + np.absolute(np.sign(beta.values))
     
        #The following line returns 1 for all entries that appear in more than (or equal to) self.threshold fraction of bootstraps and 0 otherwise
        thresholded_matrix = ((betas_non_zero / len(self.betas)) >= self.threshold).astype(int)
        #Note that the current version is blind to the sign of those betas, so the betas_sign matrix is not used. Later we might want to modify this such that only same-sign interactions count.
        return thresholded_matrix

    def calculate_precision_recall(self, combined_confidences, gold_standard):
        # filter gold standard
        gold_standard_filtered_cols = gold_standard[combined_confidences.columns]
        gold_standard_filtered = gold_standard_filtered_cols.loc[combined_confidences.index]
        #the following six lines remove all rows and columns that consist of all zeros in the gold standard
        index_cols = np.where(gold_standard_filtered.abs().sum(axis=0) > 0)[0]
        index_rows = np.where(gold_standard_filtered.abs().sum(axis=1) > 0)[0]
        gold_standard_nozero_cols = gold_standard_filtered[index_cols]
        gold_standard_nozero = gold_standard_nozero_cols.iloc[index_rows]
        combined_confidences_nozero_cols = combined_confidences[index_cols]
        combined_confidences_nozero = combined_confidences_nozero_cols.iloc[index_rows]
        # rank from highest to lowest confidence
        sorted_candidates = np.argsort(combined_confidences_nozero.values, axis = None)[::-1]
        gs_values = gold_standard_nozero.values.flatten()[sorted_candidates]
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
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = " + aupr.astype("string"), xy=(0.4, 0.05), xycoords='axes fraction')
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))