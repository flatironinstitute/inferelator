"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import os

# Shadow built-in zip with itertools.izip if this is python2 (This puts out a memory dumpster fire)
try:
    from itertools import izip as zip
except ImportError:
    pass

import numpy as np
from inferelator_ng import utils
from inferelator_ng import single_cell_puppeteer_workflow
from inferelator_ng import default
from inferelator_ng import amusr_regression
from inferelator_ng import results_processor


class SingleCellMultiTask(single_cell_puppeteer_workflow.SingleCellPuppeteerWorkflow):
    regression_type = amusr_regression
    prior_weight = 1

    def run(self):
        np.random.seed(self.random_seed)

        self.startup()
        self.separate_tasks_by_metadata()
        self.process_task_data()
        self.set_regression_type()

        betas, betas_resc = self.run_regression()
        # Write the results out to a file
        self.emit_results(betas, betas_resc)

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
        self.targets, self.regulators = None, None

        for expr_data, meta_data in zip(self.expression_matrix, self.meta_data):
            task = self.new_puppet(expr_data, meta_data, seed=self.random_seed)
            task.startup()
            self.task_design.append(task.design)
            self.task_response.append(task.response)
            self.task_meta_data.append(task.meta_data)
            self.task_bootstraps.append(task.get_bootstraps())

            if self.regulators is None:
                self.regulators = task.design.index
            else:
                self.regulators = self.regulators.intersection(task.design.index)

            if self.targets is None:
                self.targets = task.response.index
            else:
                self.targets = self.targets.intersection(task.response.index)

        self.expression_matrix = None

        utils.Debug.vprint("Processed data into design/response [{g} x {k}]".format(g=len(self.targets),
                                                                                    k=len(self.regulators)), level=0)

    def emit_results(self, betas, rescaled_betas):
        """
        Output result report(s) for workflow run.
        """
        self.create_output_dir()
        for k in range(self.n_tasks):
            output_dir = os.path.join(self.output_dir, self.tasks_dir[k])
            os.makedirs(output_dir)
            rp = results_processor.ResultsProcessor(betas[k], rescaled_betas[k],
                                                    filter_method=self.gold_standard_filter_method)
            rp.summarize_network(output_dir, self.gold_standard, self.priors_data)

    def set_gold_standard_and_priors(self):
        """
        Read priors file into priors_data and gold standard file into gold_standard
        """
        self.priors_data = self.input_dataframe(self.priors_file)

        if self.split_priors_for_gold_standard:
            self.split_priors_into_gold_standard()
        else:
            self.gold_standard = self.input_dataframe(self.gold_standard_file)

        if self.split_gold_standard_for_crossvalidation:
            self.cross_validate_gold_standard()
