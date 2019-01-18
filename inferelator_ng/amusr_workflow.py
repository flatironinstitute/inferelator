"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import os

# Shadow built-in zip with itertools.izip if this is python2 (This puts out a memory dumpster fire)
try:
    from itertools import izip as zip
except ImportError:
    pass

import pandas as pd
import numpy as np
from inferelator_ng import utils
from inferelator_ng import single_cell_puppeteer_workflow
from inferelator_ng import single_cell_workflow
from inferelator_ng import default
from inferelator_ng import amusr_regression
from inferelator_ng import results_processor


class ResultsProcessorMultiTask(results_processor.ResultsProcessor):
    """
    This results processor should handle the results of the MultiTask inferelator
    """

    def summarize_network(self, output_dir, gold_standard, priors, confidence_threshold=default.DEFAULT_CONF,
                          precision_threshold=default.DEFAULT_PREC, output_task_files=True):
        overall_confidences = []
        overall_resc_betas = []
        overall_sign = pd.DataFrame(np.zeros(self.betas[0][0].shape), index=self.betas[0][0].index,
                                    columns=self.betas[0][0].columns)
        overall_threshold = overall_sign.copy()

        for task_id, task_dir in enumerate(output_dir[1]):
            pr_calc = results_processor.RankSummaryPR(self.rescaled_betas[task_id], gold_standard,
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

            if output_task_files:
                self.write_output_files(pr_calc, task_dir, priors, gold_standard, task_threshold, network_data)

        overall_pr_calc = results_processor.RankSummaryPR(overall_confidences, gold_standard,
                                                          filter_method=self.filter_method)

        overall_threshold = (overall_threshold / len(overall_confidences) > self.threshold).astype(int)
        overall_resc_betas_mean, overall_resc_betas_median = self.mean_and_median(overall_resc_betas)
        network_data = {'beta.sign.sum': overall_sign, 'var.exp.median': overall_resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=overall_pr_calc.aupr), level=0)

        self.write_output_files(overall_pr_calc, output_dir[0], priors, gold_standard, overall_threshold, network_data)

        # Calculate how many interactions are stable (are above the combined confidence threshold)
        stable_interactions = overall_pr_calc.num_over_conf_threshold(confidence_threshold)
        # Calculate how many interactions we should keep for our model (are above the precision threshold)
        precision_interactions = overall_pr_calc.num_over_precision_threshold(precision_threshold)

        return overall_pr_calc.aupr, stable_interactions, precision_interactions


class SingleCellMultiTask(single_cell_workflow.SingleCellWorkflow, single_cell_puppeteer_workflow.PuppeteerWorkflow):
    regression_type = amusr_regression
    prior_weight = 1.
    task_expression_filter = "intersection"

    # Task-specific data
    n_tasks = None
    task_design = []
    task_response = []
    task_meta_data = []
    task_bootstraps = []
    tasks_dir = []

    # Axis labels to keep
    targets = None
    regulators = None

    def startup_finish(self):
        # If the expression matrix is [G x N], transpose it for preprocessing
        if not self.expression_matrix_columns_are_genes:
            self.expression_matrix = self.expression_matrix.transpose()

        # Filter expression and priors to align
        self.filter_expression_and_priors()
        self.separate_tasks_by_metadata()
        self.process_task_data()

    def align_priors_and_expression(self):
        pass

    def separate_tasks_by_metadata(self, meta_data_column=default.DEFAULT_METADATA_FOR_BATCH_CORRECTION):
        """
        Take a single expression matrix and break it into multiple dataframes based on meta_data. Reset the
        self.expression_matrix and self.meta_data with a list of dataframes

        :param meta_data_column: str
            Meta_data column which corresponds to task ID
        """

        task_name, task_data, task_metadata = [], [], []

        for task in self.meta_data[meta_data_column].unique():
            task_idx = self.meta_data[meta_data_column] == task
            task_data.append(self.expression_matrix.loc[:, task_idx])
            task_metadata.append(self.meta_data.loc[task_idx, :])
            task_name.append(task)

        self.n_tasks = len(task_data)
        self.expression_matrix = task_data
        self.meta_data = task_metadata
        self.tasks_dir = task_name

        utils.Debug.vprint("Separated data into {ntask} tasks".format(ntask=self.n_tasks), level=0)

    def process_task_data(self):
        """
        Preprocess the individual task data using a child worker into task design and response data
        """

        self.task_design, self.task_response, self.task_meta_data, self.task_bootstraps = [], [], [], []
        targets, regulators = [], []

        for expr_data, meta_data in zip(self.expression_matrix, self.meta_data):
            task = self.new_puppet(expr_data, meta_data, seed=self.random_seed)
            task.startup_finish()
            self.task_design.append(task.design)
            self.task_response.append(task.response)
            self.task_meta_data.append(task.meta_data)
            self.task_bootstraps.append(task.get_bootstraps())

            regulators.append(task.design.index)
            targets.append(task.response.index)

        self.targets = amusr_regression.filter_genes_on_tasks(targets, self.task_expression_filter)
        self.regulators = amusr_regression.filter_genes_on_tasks(regulators, self.task_expression_filter)
        self.expression_matrix = None

        utils.Debug.vprint("Processed data into design/response [{g} x {k}]".format(g=len(self.targets),
                                                                                    k=len(self.regulators)), level=0)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors_data):
        """
        Output result report(s) for workflow run.
        """
        if self.is_master():
            self.create_output_dir()
            self.output_dir = (self.output_dir, [os.path.join(self.output_dir, tk) for tk in self.tasks_dir])

            for task_path in self.output_dir[1]:
                try:
                    os.makedirs(task_path)
                except OSError:
                    pass

            rp = ResultsProcessorMultiTask(betas, rescaled_betas, filter_method=self.gold_standard_filter_method)
            results = rp.summarize_network(self.output_dir, gold_standard, priors_data)
            self.aupr, self.n_interact, self.precision_interact = results
        else:
            self.aupr, self.n_interact, self.precision_interact = None, None, None
