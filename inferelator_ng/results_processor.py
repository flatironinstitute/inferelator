import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import condition

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

    def calculate_aupr(self, combined_confidences, gold_standard):
        candidates = np.where( combined_confidences > 0)
        # filter gold standard
        gold_standard_filtered = gold_standard_filtered[combined_confidences.columns]
        condition_positive = len(np.where(gold_standard_filtered.values > 0)[0])
        # rank from highest to lowest confidence
        sorted_candidates = np.argsort(combined_confidences.values[candidates], axis = None)[::-1]
        combined_confidences.values[candidates][sorted_candidates[0]]
        gs_values = np.array(gold_standard_filtered.values[candidates])
        TP = 0.0
        FP = 0.0
        precision = []
        recall = []
        for i in sorted_candidates:
            truth = gs_values[i]
            if truth == 1:
                TP = TP + 1
            else:
                FP = FP + 1
            precision.append(TP / (TP + FP))
            recall.append(TP / condition_positive)

    def plot_pr_curve(self, recall, precision):
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.save('pr_curve.png')