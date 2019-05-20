import pandas as pd
import numpy as np
import os

from inferelator import default
from inferelator import utils
from inferelator.postprocessing import results_processor
from inferelator.postprocessing import model_performance


class ResultsProcessorMultiTask(results_processor.ResultsProcessor):
    """
    This results processor should handle the results of the MultiTask inferelator

    It will output the results for each task, as well as rank-combining to construct a network from all tasks
    """

    write_task_files = True

    tasks_names = None
    tasks_networks = None

    def __init__(self, betas, rescaled_betas, threshold=0.5, filter_method='overlap', tasks_names=None):
        """
        :param betas: list(pd.DataFrame[G x K])
        :param rescaled_betas: list(pd.DataFrame[G x K])
        :param threshold: float
        :param filter_method: str
            How to handle gold standard filtering ('overlap' filters to beta, 'keep_all_gold_standard' doesn't filter)
        :param tasks_names: list(str)
            The names for each task
        """
        self.betas = betas
        self.rescaled_betas = rescaled_betas
        self.filter_method = filter_method

        if 1 >= threshold >= 0:
            self.threshold = threshold
        else:
            raise ValueError("Threshold must be a float in the interval [0, 1]")

        if tasks_names is not None:
            self.tasks_names = tasks_names
        else:
            self.tasks_names = []

    def summarize_network(self, output_dir, gold_standard, priors, confidence_threshold=default.DEFAULT_CONF,
                          precision_threshold=default.DEFAULT_PREC):
        """
        Take the betas and rescaled beta_errors, construct a network, and test it against the gold standard
        :param output_dir: str
            Path to write files into. Don't write anything if this is None.
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard to test the network against
        :param priors: pd.DataFrame [G x K] or list(pd.DataFrame [G x K])
            Prior data
        :param confidence_threshold: float
        :param precision_threshold: float
        :return aupr: float
            Returns the AUPR calculated from the network and gold standard
        :return stable_interactions: int
            Number of interactions with a combined confidence over confidence_threshold
        :return precision_interactions: int
            Number of interactions with a combined confidence over the precision from precision_threshold
        """

        overall_confidences = []
        overall_resc_betas = []
        overall_sign = pd.DataFrame(np.zeros(self.betas[0][0].shape), index=self.betas[0][0].index,
                                    columns=self.betas[0][0].columns)
        overall_threshold = overall_sign.copy()

        self.tasks_networks = {}
        for task_id, task_dir in enumerate(self.tasks_names):
            pr_calc = model_performance.RankSummaryPR(self.rescaled_betas[task_id], gold_standard,
                                                      filter_method=self.filter_method)
            task_threshold, task_sign, task_nonzero = self.threshold_and_summarize(self.betas[task_id], self.threshold)
            task_resc_betas_mean, task_resc_betas_median = self.mean_and_median(self.rescaled_betas[task_id])
            network_data = {'beta.sign.sum': task_sign, 'var.exp.median': task_resc_betas_median}

            # Pile up data
            overall_confidences.append(pr_calc.combined_confidences())
            overall_resc_betas.append(task_resc_betas_median)
            overall_sign += np.sign(task_sign)
            overall_threshold += task_threshold

            utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=pr_calc.aupr), level=0)

            if self.write_task_files is True and output_dir is not None:
                task_net = self.write_output_files(pr_calc, os.path.join(output_dir, task_dir), priors, task_threshold,
                                                   network_data)
                self.tasks_networks[task_id] = task_net

        overall_pr_calc = model_performance.RankSummaryPR(overall_confidences, gold_standard,
                                                          filter_method=self.filter_method)

        overall_threshold = (overall_threshold / len(overall_confidences) > self.threshold).astype(int)
        overall_resc_betas_mean, overall_resc_betas_median = self.mean_and_median(overall_resc_betas)
        network_data = {'beta.sign.sum': overall_sign, 'var.exp.median': overall_resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=overall_pr_calc.aupr), level=0)

        self.network_data = self.write_output_files(overall_pr_calc, output_dir, priors, overall_threshold,
                                                    network_data, threshold_network=False)

        # Calculate how many interactions are stable (are above the combined confidence threshold)
        stable_interactions = overall_pr_calc.num_over_conf_threshold(confidence_threshold)
        # Calculate how many interactions we should keep for our model (are above the precision threshold)
        precision_interactions = overall_pr_calc.num_over_precision_threshold(precision_threshold)

        return overall_pr_calc.aupr, stable_interactions, precision_interactions

