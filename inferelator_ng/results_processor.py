import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import condition

class ResultsProcessor:

    def __init__(self, betas, rescaled_betas, threshold=0.5):
        self.betas = betas
        self.rescaled_betas = rescaled_betas
        self.threshold = threshold

    def compute_combined_confidences():
        combined_confidences = pandas.DataFrame(np.zeros((betas[0].shape)), index = betas[0].index, columns = betas[0].columns )
        for beta_resc in betas_resc:
            # this ranking code is especially wordy because the rank function only works in one dimension (col or row), so I had to flatten the matrix
            ranked_df =np.reshape(pandas.DataFrame(np.ndarray.flatten(beta_resc.values)).rank( method = "average").values, betas_resc[0].shape)
            combined_confidences = combined_confidences + ranked_df

        min_element = min(combined_confidences.min())
        combined_confidences = (combined_confidences - min_element) / (len(betas) * combined_confidences.size - min_element)
        return combined_confidences

    def threshold_and_summarize():
        betas_sign = pandas.DataFrame(np.zeros((betas[0].shape)), index = betas[0].index, columns = betas[0].columns )
        betas_non_zero = pandas.DataFrame(np.zeros((betas[0].shape)), index = betas[0].index, columns = betas[0].columns )
        for beta in betas:
            betas_sign = betas_sign + np.sign(beta.values)
            betas_non_zero = betas_non_zero + np.absolute(np.sign(beta.values))

        # we only care about interactions that are present in more than th (fraction) bootstraps
        index_vector = np.where( betas_non_zero > len(betas) * self.threshold)
        betas_stack = np.stack([b.values[index_vector] for b in betas])
        return betas_stack

    def calculate_aupr(combined_confidences, gold_standard):
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

    def plot_pr_curve(recall, precision):
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.save('pr_curve.png')