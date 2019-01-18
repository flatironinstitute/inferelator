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

    def summarize_network(self, output_dir, gold_standard, priors):
        """

        :param output_dir: (path, [path])
        :param gold_standard:
        :param priors:
        :return:
        """
        overall_confidences = []
        overall_resc_betas = []
        overall_sign = pd.DataFrame(np.zeros(self.betas[0][0].shape), index=self.betas[0][0].index,
                                    columns=self.betas[0][0].columns)
        overall_threshold = overall_sign.copy()

        for task_id, task_dir in enumerate(output_dir[1]):
            combined_confidences = self.compute_combined_confidences(self.rescaled_betas[task_id])
            task_threshold, task_sign, task_nonzero = self.threshold_and_summarize(self.betas[task_id], self.threshold)

            overall_confidences.append(combined_confidences)
            overall_sign += np.sign(task_sign)
            overall_threshold += task_threshold

            combined_confidences.to_csv(os.path.join(task_dir, 'combined_confidences.tsv'), sep='\t')
            task_threshold.to_csv(os.path.join(task_dir, 'betas_stack.tsv'), sep='\t')

            recall, precision = self.calculate_precision_recall(combined_confidences, gold_standard)
            aupr = self.calculate_aupr(recall, precision)
            utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=aupr), level=0)

            # Plot PR curve
            self.plot_pr_curve(recall, precision, aupr, output_dir)

            task_resc_betas_mean, task_resc_betas_median = self.mean_and_median(self.rescaled_betas[task_id])
            overall_resc_betas.append(task_resc_betas_median)

            self.save_network_to_tsv(combined_confidences, task_sign, task_resc_betas_median, priors, gold_standard,
                                     task_dir)

        rank_combine_conf = self.compute_combined_confidences(overall_confidences)
        rank_combine_conf.to_csv(os.path.join(output_dir[0], 'combined_confidences.tsv'), sep='\t')

        overall_threshold = (overall_threshold / len(overall_confidences) > self.threshold).astype(int)
        overall_threshold.to_csv(os.path.join(output_dir[0], 'betas_stack.tsv'), sep='\t')

        recall, precision = self.calculate_precision_recall(rank_combine_conf, gold_standard)
        aupr = self.calculate_aupr(recall, precision)
        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=aupr), level=0)

        task_resc_betas_mean, task_resc_betas_median = self.mean_and_median(overall_resc_betas)
        self.save_network_to_tsv(rank_combine_conf, overall_sign, task_resc_betas_median, priors, gold_standard,
                                 output_dir[0])

        return aupr


class SingleCellMultiTask(single_cell_workflow.SingleCellWorkflow, single_cell_puppeteer_workflow.PuppeteerWorkflow):
    regression_type = amusr_regression
    prior_weight = 1.
    task_expression_filter = "intersection"

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
            rp.summarize_network(self.output_dir, gold_standard, priors_data)
