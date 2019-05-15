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
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator import crossvalidation_workflow
from inferelator import single_cell_workflow
from inferelator import default
from inferelator.regression import amusr_regression
from inferelator.postprocessing import results_processor
from inferelator.postprocessing import model_performance
from inferelator.preprocessing.metadata_parser import MetadataHandler


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
        :param priors: pd.DataFrame [G x K]
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


class MultitaskLearningWorkflow(single_cell_workflow.SingleCellWorkflow, crossvalidation_workflow.PuppeteerWorkflow):
    """
    Class that implements AMuSR multitask learning for single cell data

    Extends SingleCellWorkflow
    Inherits from PuppeteerWorkflow so that task preprocessing can be done more easily
    """

    prior_weight = default.DEFAULT_prior_weight
    task_expression_filter = "intersection"

    # Task-specific data
    n_tasks = None
    task_design = None
    task_response = None
    task_meta_data = None
    task_bootstraps = None
    tasks_names = None

    meta_data_handlers = None
    meta_data_task_column = default.DEFAULT_METADATA_FOR_BATCH_CORRECTION

    # Axis labels to keep
    targets = None
    regulators = None

    # Multi-task result processor
    result_processor_driver = ResultsProcessorMultiTask

    # Workflow type for task processing
    cv_workflow_type = single_cell_workflow.SingleCellWorkflow

    def read_expression(self, file=None):
        """
        Load a list of expression files (as tasks) or call the workflowbase loader
        """

        file = file if file is not None else self.expression_matrix_file

        # Load a list of metadatas
        if isinstance(file, list):
            self.expression_matrix = [self.input_dataframe(task_expr) for task_expr in file]
            self.n_tasks = len(self.expression_matrix)
        else:
            super(MultitaskLearningWorkflow, self).read_expression(file=file)

    def read_metadata(self, file=None):
        """
        Load a list of metadata files (as tasks) or call the workflowbase loader
        TODO: Break this up and clean this up
        """

        file = file if file is not None else self.meta_data_file

        # Load a list of metadatas from a list of files
        # Create a default metadata if the file name is None
        if isinstance(file, list):
            assert len(file) == self.n_tasks
            meta_handler = MetadataHandler.get_handler()

            self.meta_data = list()
            for task_id, task_meta in enumerate(file):
                if task_meta is None:
                    self.set_metadata_handler(task_id)
                    meta_handler = MetadataHandler.get_handler()
                    self.meta_data.append(meta_handler.create_default_meta_data(self.expression_matrix[task_id]))
                else:
                    self.meta_data.append(self.input_dataframe(task_meta, index_col=None))

        # Extract the metadata from each expression matrix
        elif isinstance(self.expression_matrix, list) and self.extract_metadata_from_expression_matrix:
            if not isinstance(self.expression_matrix_metadata[0], list):
                expr_meta_cols = [self.expression_matrix_metadata] * len(self.expression_matrix)
            else:
                assert len(self.expression_matrix_metadata) == self.n_tasks
                expr_meta_cols = self.expression_matrix_metadata

            self.meta_data = [None] * len(self.expression_matrix)

            for task_id in range(len(self.expression_matrix)):
                processed_data = self.dataframe_split(self.expression_matrix[task_id], expr_meta_cols[task_id])
                self.expression_matrix[task_id], self.meta_data[task_id] = processed_data
        else:
            super(MultitaskLearningWorkflow, self).read_metadata(file=file)

    def set_metadata_handler(self, task_id):
        """
        Check the meta_data_handlers instance variable and reset the metadata handler to match if needed
        :param task_id: int
        """

        # If meta_data_handlers is a list, pick the correct one from the list by task_id
        if isinstance(self.meta_data_handlers, list):
            assert len(self.meta_data_handlers) == self.n_tasks
            MetadataHandler.set_handler(self.meta_data_handlers[task_id])

        # If meta_data_handlers isn't a list, set it for any task_id
        elif self.meta_data_handlers is not None:
            MetadataHandler.set_handler(self.meta_data_handlers)

        # If meta_data_handlers is None, just move on with our lives
        else:
            pass

    def transpose_expression_matrix(self):
        """
        Transpose the expression matrix or transpose each of a list of expression matrixes
        """

        # If the expression_matrix_columns_are_genes is a list of bools, flip according to the list
        if isinstance(self.expression_matrix, list) and isinstance(self.expression_matrix_columns_are_genes, list):
            assert len(self.expression_matrix) == len(self.expression_matrix_columns_are_genes)
            for task_id, flip_bool in enumerate(self.expression_matrix_columns_are_genes):
                if flip_bool:
                    self.expression_matrix[task_id] = self.expression_matrix[task_id].transpose()

        # If the expression_matrix_columns_are_genes is True, flip all of the expression matrixes
        elif isinstance(self.expression_matrix, list) and self.expression_matrix_columns_are_genes:
            self.expression_matrix = list(map(lambda x: x.transpose(), self.expression_matrix))

        # If expression_matrix isn't a list, call to the super workflow function
        else:
            super(MultitaskLearningWorkflow, self).transpose_expression_matrix()

    def read_priors(self, priors_file=None, gold_standard_file=None):
        """
        Load a list of priors (as tasks) or gold standards (as tasks) or call the workflowbase loader
        """

        priors_file = priors_file if priors_file is not None else self.priors_file
        gold_standard_file = gold_standard_file if gold_standard_file is not None else self.gold_standard_file

        if isinstance(priors_file, list):
            self.priors_data = [self.input_dataframe(task_priors) for task_priors in priors_file]
        elif priors_file is not None:
            self.priors_data = self.input_dataframe(priors_file)
        if isinstance(gold_standard_file, list):
            self.gold_standard = [self.input_dataframe(task_gold) for task_gold in gold_standard_file]
        elif gold_standard_file is not None:
            self.gold_standard = self.input_dataframe(gold_standard_file)

    def startup_finish(self):
        # Make sure tasks are set correctly
        self.prepare_tasks()
        self.check_tasks()
        self.process_task_data()

    def prepare_tasks(self):
        """
        Process one expression matrix/metadata file into multiple tasks if necessary
        """

        if isinstance(self.expression_matrix, pd.DataFrame) and isinstance(self.meta_data, pd.DataFrame):
            self.separate_tasks_by_metadata()
        elif isinstance(self.expression_matrix, pd.DataFrame) != isinstance(self.meta_data, pd.DataFrame):
            raise NotImplementedError("Metadata and expression must both be a single file or both be a list of files")
        else:
            pass

    def check_tasks(self):
        """
        Confirm that task data has been separated and that the multitask workflow is ready for regression
        """

        assert check.argument_type(self.expression_matrix, list)
        assert check.argument_type(self.meta_data, list)

        if self.n_tasks is None:
            raise ValueError("n_tasks is not set")
        if self.n_tasks != len(self.expression_matrix):
            raise ValueError("n_tasks is inconsistent with task expression data")
        if self.n_tasks != len(self.meta_data):
            raise ValueError("n_tasks is inconsistent with task meta data")
        if self.tasks_names is None:
            utils.Debug.vprint("Creating default task names")
            self.tasks_names = list(map(str, range(self.n_tasks)))

        return True

    def separate_tasks_by_metadata(self, meta_data_column=None):
        """
        Take a single expression matrix and break it into multiple dataframes based on meta_data. Reset the
        self.expression_matrix and self.meta_data with a list of dataframes, self.n_tasks with the number of tasks,
        and self.tasks_names with the name from meta_data for each task.

        :param meta_data_column: str
            Meta_data column which corresponds to task ID

        """

        meta_data_column = meta_data_column if meta_data_column is not None else self.meta_data_task_column

        task_name, task_data, task_metadata = [], [], []
        tasks = self.meta_data[meta_data_column].unique().tolist()

        utils.Debug.vprint("Creating {n} tasks from metadata column {col}".format(n=len(tasks), col=meta_data_column),
                           level=0)

        for task in tasks:
            task_idx = (self.meta_data[meta_data_column] == task).tolist()
            task_data.append(self.expression_matrix.loc[:, task_idx])
            task_metadata.append(self.meta_data.loc[task_idx, :])
            task_name.append(task)

        self.n_tasks = len(task_data)
        self.expression_matrix = task_data
        self.meta_data = task_metadata
        self.tasks_names = task_name

        utils.Debug.vprint("Separated data into {ntask} tasks".format(ntask=self.n_tasks), level=0)

    def process_task_data(self):
        """
        Preprocess the individual task data using a child worker into task design and response data. Set
        self.task_design, self.task_response, self.task_meta_data, self.task_bootstraps with lists which contain
        DataFrames.

        Also set self.regulators and self.targets with pd.Indexes that correspond to the genes and tfs to model
        This is chosen based on the filtering strategy set in self.task_expression_filter
        """

        self.task_design, self.task_response, self.task_meta_data, self.task_bootstraps = [], [], [], []
        targets, regulators = [], []

        for task_id, (expr_data, meta_data) in enumerate(zip(self.expression_matrix, self.meta_data)):
            utils.Debug.vprint("Processing {task} [{sh}]".format(task=self.tasks_names[task_id], sh=expr_data.shape),
                               level=1)

            if isinstance(self.priors_data, list):
                task_prior = self.priors_data[task_id]
            else:
                task_prior = self.priors_data

            self.set_metadata_handler(task_id)
            task = self.new_puppet(expr_data, meta_data, seed=self.random_seed, priors_data=task_prior)
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
            rp = self.result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                              tasks_names=self.tasks_names)
            results = rp.summarize_network(self.output_dir, gold_standard, priors_data)
            self.aupr, self.n_interact, self.precision_interact = results
            return rp.network_data
        else:
            self.aupr, self.n_interact, self.precision_interact = None, None, None
