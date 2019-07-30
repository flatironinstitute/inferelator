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
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator import crossvalidation_workflow
from inferelator import single_cell_workflow
from inferelator import default
from inferelator.regression import amusr_regression
from inferelator.preprocessing.metadata_parser import MetadataHandler
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask


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

    def startup_run(self):
        self.get_data()

    def startup_finish(self):
        # Make sure tasks are set correctly
        self.prepare_tasks()
        self.check_tasks()
        self.process_task_data()

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
        """

        file = file if file is not None else self.meta_data_file

        # Load a list of metadatas from a list of files
        # Create a default metadata if the file name is None
        if isinstance(file, list):
            self.read_metadata_list(file)
        # Extract the metadata from each expression matrix
        elif isinstance(self.expression_matrix, list) and self.extract_metadata_from_expression_matrix:
            self.extract_metadata_from_list()
        else:
            super(MultitaskLearningWorkflow, self).read_metadata(file=file)

    def read_metadata_list(self, file_list):
        """
        Read a list of metadata file names into a list of metadata dataframes
        :param file_list:
        """

        assert len(file_list) == self.n_tasks
        self.meta_data = []

        for task_id, task_meta in enumerate(file_list):
            if task_meta is None:
                self.set_metadata_handler(task_id)
                meta_handler = MetadataHandler.get_handler()
                self.meta_data.append(meta_handler.create_default_meta_data(self.expression_matrix[task_id]))
            else:
                self.meta_data.append(self.input_dataframe(task_meta, index_col=None))

    def extract_metadata_from_list(self):
        """
        Process a list of expression data dataframes to extract any metadata columns set in
        self.expression_matrix_metadata
        """

        # If self.expression_matrix_metadata is a list of lists, use each list as a set of columns to extract
        if isinstance(self.expression_matrix_metadata[0], list):
            assert len(self.expression_matrix_metadata) == self.n_tasks
            expr_meta_cols = self.expression_matrix_metadata
        # If self.expression_matrix_metadata is a list of column names, use those columns for every expression matrix
        else:
            expr_meta_cols = [self.expression_matrix_metadata] * len(self.expression_matrix)

        # For every task, extract the metadata from the expression matrix
        self.meta_data = [None] * len(self.expression_matrix)
        for task_id in range(len(self.expression_matrix)):
            processed_data = self.dataframe_split(self.expression_matrix[task_id], expr_meta_cols[task_id])
            self.expression_matrix[task_id], self.meta_data[task_id] = processed_data

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
        elif isinstance(self.expression_matrix, pd.DataFrame) and self.expression_matrix_columns_are_genes:
            super(MultitaskLearningWorkflow, self).transpose_expression_matrix()

        # Don't do anything
        else:
            pass

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

        assert check.argument_type(self.meta_data, pd.DataFrame)
        assert check.argument_type(self.expression_matrix, pd.DataFrame)
        assert self.meta_data.shape[0] == self.expression_matrix.shape[1]

        meta_data_column = meta_data_column if meta_data_column is not None else self.meta_data_task_column

        task_name, task_data, task_metadata = [], [], []
        tasks = self.meta_data[meta_data_column].unique().tolist()

        utils.Debug.vprint("Creating {n} tasks from metadata column {col}".format(n=len(tasks), col=meta_data_column),
                           level=0)

        for task in tasks:
            task_idx = self.meta_data[meta_data_column] == task
            task_data.append(self.expression_matrix.iloc[:, [i for i, j in enumerate(task_idx) if j]])
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
