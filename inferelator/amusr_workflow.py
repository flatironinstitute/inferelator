"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import copy
import gc

# Shadow built-in zip with itertools.izip if this is python2 (This puts out a memory dumpster fire)
try:
    from itertools import izip as zip
except ImportError:
    pass

import pandas as pd
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator import workflow
from inferelator import single_cell_workflow
from inferelator import default
from inferelator.regression import amusr_regression
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask

TRANSFER_ATTRIBUTES = ['count_minimum', 'preprocessing_workflow', 'input_dir']


class MultitaskLearningWorkflow(single_cell_workflow.SingleCellWorkflow):
    """
    Class that implements AMuSR multitask learning for single cell data

    Extends SingleCellWorkflow
    Inherits from PuppeteerWorkflow so that task preprocessing can be done more easily
    """

    prior_weight = default.DEFAULT_prior_weight
    regulator_expression_filter = "intersection"
    target_expression_filter = "union"

    # Task-specific data
    n_tasks = None
    task_design = None
    task_response = None
    task_meta_data = None
    task_bootstraps = None
    task_priors = None
    task_names = None
    task_objects = None

    # Axis labels to keep
    targets = None
    regulators = None

    # Multi-task result processor
    result_processor_driver = ResultsProcessorMultiTask

    # Workflow type for task processing
    cv_workflow_type = single_cell_workflow.SingleCellWorkflow

    def startup_run(self):

        # Task data has expression & metadata and may have task-specific files for anything else
        self._load_tasks()

        # Priors, gold standard, tf_names, and gene metadata will be loaded if set
        self.read_priors()
        self.read_genes()

    def startup_finish(self):
        # Make sure tasks are set correctly
        self._process_default_priors()
        self._process_task_priors()
        self._process_task_data()

    def create_task(self, task_name=None, expression_matrix_file=None, input_dir=None, meta_data_file=None,
                    tf_names_file=None, priors_file=None, gold_standard_file=None, workflow_type="single-cell",
                    preprocessing_workflow=None, **kwargs):
        """
        Create a task object and set any arguments to this function as attributes of that task object
        This creates a TaskData class and puts it in self.task_objects

        :param task_name: str
        :param expression_matrix_file: str
        :param input_dir: str
        :param meta_data_file: str
        :param tf_names_file: str
        :param priors_file: str
        :param gold_standard_file: str
        :param workflow_type: str
        :param kwargs:
        """

        # Create a TaskData object from a workflow and set the formal arguments into it
        task_object = create_task_data_object(workflow_class=workflow_type)
        task_object.task_name = task_name
        task_object.input_dir = input_dir
        task_object.expression_matrix_file = expression_matrix_file
        task_object.meta_data_file = meta_data_file
        task_object.tf_names_file = tf_names_file
        task_object.priors_file = priors_file
        task_object.gold_standard_file = gold_standard_file
        task_object.preprocessing_workflow = preprocessing_workflow

        # Pass forward any kwargs (raising errors if they're for attributes that don't exist)
        for attr, val in kwargs.items():
            if hasattr(task_object, attr):
                setattr(task_object, attr, val)
            else:
                raise ValueError("Argument {attr} cannot be set as an attribute".format(attr=attr))

        if self.task_objects is None:
            self.task_objects = [task_object]
        else:
            self.task_objects.append(task_object)

        utils.Debug.vprint(str(task_object), level=0)

    def _load_tasks(self):
        """
        Run load_task_data in all the TaskData objects created with create_task
        """
        if self.task_objects is None:
            raise ValueError("Tasks have not been created with .create_task()")

        for tobj in self.task_objects:
            for attr in TRANSFER_ATTRIBUTES:
                try:
                    if getattr(self, attr) is not None and getattr(tobj, attr) is None:
                        setattr(tobj, attr, getattr(self, attr))
                except AttributeError:
                    pass

        # Run load_task_data and create a list of lists of TaskData objects
        # This allows a TaskData object to copy and split itself if needed
        self.task_objects = [tobj.get_data() for tobj in self.task_objects]

        # Flatten the list
        self.task_objects = [tobj for tobj_list in self.task_objects for tobj in tobj_list]
        self.n_tasks = len(self.task_objects)

    def read_priors(self, priors_file=None, gold_standard_file=None):
        """
        Load priors and gold standard. Make sure all tasks have priors
        """

        priors_file = priors_file if priors_file is not None else self.priors_file
        gold_standard_file = gold_standard_file if gold_standard_file is not None else self.gold_standard_file

        if priors_file is not None:
            self.priors_data = self.input_dataframe(priors_file)
        if gold_standard_file is not None:
            self.gold_standard = self.input_dataframe(gold_standard_file)
        else:
            raise ValueError("A gold standard must be provided to `gold_standard_file` in MultiTaskLearningWorkflow")

        # Check to see if there are any tasks which don't have priors
        no_priors = sum(map(lambda x: x.priors_data is None, self.task_objects))
        if no_priors > 0 and self.priors_data is None:
            raise ValueError("{n} tasks have no priors (no default prior is set)".format(n=no_priors))

    def _process_default_priors(self):
        """
        Process the default priors in the parent workflow for crossvalidation or shuffling
        """

        priors = self.priors_data if self.priors_data is not None else self.gold_standard.copy()

        # Crossvalidation
        if self.split_gold_standard_for_crossvalidation:
            priors, self.gold_standard = self.prior_manager.cross_validate_gold_standard(priors, self.gold_standard,
                                                                                         self.cv_split_axis,
                                                                                         self.cv_split_ratio,
                                                                                         self.random_seed)
        # Filter to regulators
        if self.tf_names is not None:
            priors = self.prior_manager.filter_to_tf_names_list(priors, self.tf_names)

        # Filter to targets
        if self.gene_metadata is not None and self.gene_list_index is not None:
            gene_list = self.gene_metadata[self.gene_list_index].tolist()
            priors = self.prior_manager.filter_priors_to_genes(priors, gene_list)

        # Shuffle labels
        if self.shuffle_prior_axis is not None:
            priors = self.prior_manager.shuffle_priors(priors, self.shuffle_prior_axis, self.random_seed)

        # Reset the priors_data in the parent workflow if it exists
        self.priors_data = priors if self.priors_data is not None else None

    def _process_task_priors(self):
        """
        Process & align the default priors for crossvalidation or shuffling
        """
        for task_obj in self.task_objects:

            # Set priors if task-specific priors are not present
            if task_obj.priors_data is None:
                assert self.priors_data is not None
                task_obj.priors_data = self.priors_data.copy()

            # Set gene metadata if task-specific gene metadata is not present
            if task_obj.gene_metadata is None:
                task_obj.gene_metadata = copy.copy(self.gene_metadata)
                task_obj.gene_list_index = self.gene_list_index

            # Process priors in the task data
            task_obj.process_priors_and_gold_standard(gold_standard=self.gold_standard,
                                                      cv_flag=self.split_gold_standard_for_crossvalidation,
                                                      cv_axis=self.cv_split_axis,
                                                      shuffle_priors=self.shuffle_prior_axis)

    def _process_task_data(self):
        """
        Preprocess the individual task data using the TaskData worker into task design and response data. Set
        self.task_design, self.task_response, self.task_meta_data, self.task_bootstraps with lists which contain
        DataFrames.

        Also set self.regulators and self.targets with pd.Indexes that correspond to the genes and tfs to model
        This is chosen based on the filtering strategy set in self.target_expression_filter and
        self.regulator_expression_filter
        """
        self.task_design, self.task_response, self.task_meta_data = [], [], []
        self.task_bootstraps, self.task_names, self.task_priors = [], [], []
        targets, regulators = [], []

        # Iterate through a list of TaskData objects holding data
        for task_id, task_obj in enumerate(self.task_objects):
            # Get task name from Task
            task_name = task_obj.task_name if task_obj.task_name is not None else str(task_id)

            task_str = "Processing task #{tid} [{t}] {sh}"
            utils.Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.expression_matrix.shape), level=1)

            # Run the preprocessing workflow
            task_obj.startup_finish()

            # Put the processed data into lists
            self.task_design.append(task_obj.design)
            self.task_response.append(task_obj.response)
            self.task_meta_data.append(task_obj.meta_data)
            self.task_bootstraps.append(task_obj.get_bootstraps())
            self.task_names.append(task_name)
            self.task_priors.append(task_obj.priors_data)

            regulators.append(task_obj.design.index)
            targets.append(task_obj.response.index)

            task_str = "Processing task #{tid} [{t}] complete [{sh} & {sh2}]"
            utils.Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.design.shape,
                                               sh2=task_obj.response.shape), level=1)

        self.targets = amusr_regression.filter_genes_on_tasks(targets, self.target_expression_filter)
        self.regulators = amusr_regression.filter_genes_on_tasks(regulators, self.regulator_expression_filter)

        utils.Debug.vprint("Processed data into design/response [{g} x {k}]".format(g=len(self.targets),
                                                                                    k=len(self.regulators)), level=0)

        # Clean up the TaskData objects and force a cyclic collection
        del self.task_objects
        gc.collect()

    def emit_results(self, betas, rescaled_betas, gold_standard, priors_data):
        """
        Output result report(s) for workflow run.
        """
        if self.is_master():
            self.create_output_dir()
            rp = self.result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                              tasks_names=self.task_names)
            results = rp.summarize_network(self.output_dir, gold_standard, priors_data)
            self.aupr, self.n_interact, self.precision_interact = results
            return rp.network_data
        else:
            self.aupr, self.n_interact, self.precision_interact = None, None, None


def create_task_data_object(workflow_class="single-cell"):
    return create_task_data_class(workflow_class=workflow_class)()


def create_task_data_class(workflow_class="single-cell"):
    task_parent = workflow.create_inferelator_workflow(regression="base", workflow=workflow_class)

    class TaskData(task_parent):

        task_name = None
        tasks_from_metadata = False
        meta_data_task_column = None

        task_workflow_class = str(workflow_class)

        str_attrs = ["input_dir", "expression_matrix_file", "tf_names_file", "meta_data_file", "priors_file",
                     "gold_standard_file", "expression_matrix_columns_are_genes",
                     "extract_metadata_from_expression_matrix", "expression_matrix_metadata"]

        def __str__(self):
            """
            Create a printable report of the settings in this TaskData object
            :return task_str: str
                Settings in str_attrs in a printable string
            """

            task_str = "{n}:\n\tWorkflow Class: {cl}\n".format(n=self.task_name, cl=self.task_workflow_class)
            for attr in self.str_attrs:
                try:
                    task_str += "\t{attr}: {val}\n".format(attr=attr, val=getattr(self, attr))
                except AttributeError:
                    task_str += "\t{attr}: Nonexistant\n".format(attr=attr)
            return task_str

        def __init__(self):
            pass

        def initialize_multiprocessing(self):
            """
            Don't do anything with multiprocessing in this object
            """
            pass

        def startup(self):
            raise NotImplementedError

        def startup_run(self):
            raise NotImplementedError

        def get_data(self):
            """
            Load all the data and then return a list of references to TaskData objects
            There will be multiple objects returned if tasks_from_metadata is set.
            :return: list(TaskData)
                List of TaskData objects with loaded data
            """
            utils.Debug.vprint("Loading data for task {task_name}".format(task_name=self.task_name))

            super(TaskData, self).get_data()

            if self.tasks_from_metadata:
                return self.separate_tasks_by_metadata()
            else:
                return [self]

        def process_priors_and_gold_standard(self, gold_standard=None, cv_flag=None, cv_axis=None, shuffle_priors=None):
            """
            Make sure that the priors for this task are correct
            """

            gold_standard = self.gold_standard if gold_standard is None else gold_standard
            cv_flag = self.split_gold_standard_for_crossvalidation if cv_flag is None else cv_flag
            cv_axis = self.cv_split_axis if cv_axis is None else cv_axis
            shuffle_priors = self.shuffle_prior_axis if shuffle_priors is None else shuffle_priors

            # Remove circularity from the gold standard
            if cv_flag:
                self.priors_data, _ = self.prior_manager._remove_prior_circularity(self.priors_data, gold_standard,
                                                                                   split_axis=cv_axis)

            if self.tf_names is not None:
                self.priors_data = self.prior_manager.filter_to_tf_names_list(self.priors_data, self.tf_names)

            # Filter priors and expression to a list of genes
            self.filter_to_gene_list()

            # Shuffle prior labels
            if shuffle_priors is not None:
                self.priors_data = self.prior_manager.shuffle_priors(self.priors_data, shuffle_priors, self.random_seed)

            if min(self.priors_data.shape) == 0:
                raise ValueError("Priors for task {n} have an axis of length 0".format(n=self.task_name))

        def separate_tasks_by_metadata(self, meta_data_column=None):
            """
            Take a single expression matrix and break it into multiple dataframes based on meta_data. Return a list of
            TaskData objects which have the task-specific data loaded into them

            :param meta_data_column: str
                Meta_data column which corresponds to task ID
            :return new_task_objects: list(TaskData)
                List of the TaskData objects with only one task's data each

            """

            assert check.argument_type(self.meta_data, pd.DataFrame)
            assert check.argument_type(self.expression_matrix, pd.DataFrame)
            assert self.meta_data.shape[0] == self.expression_matrix.shape[1]

            meta_data_column = meta_data_column if meta_data_column is not None else self.meta_data_task_column
            if meta_data_column is None:
                raise ValueError("tasks_from_metadata is set but meta_data_task_column is not")

            new_task_objects = list()
            tasks = self.meta_data[meta_data_column].unique().tolist()

            utils.Debug.vprint(
                "Creating {n} tasks from metadata column {col}".format(n=len(tasks), col=meta_data_column),
                level=0)

            # Remove data references from self
            expr_data = self.expression_matrix
            meta_data = self.meta_data
            self.expression_matrix = None
            self.meta_data = None

            for task in tasks:
                # Copy this object
                task_obj = copy.deepcopy(self)

                # Get an index of the stuff to keep
                task_idx = meta_data[meta_data_column] == task

                # Reset expression matrix, metadata, and task_name in the copy
                task_obj.expression_matrix = expr_data.iloc[:, [i for i, j in enumerate(task_idx) if j]]
                task_obj.meta_data = meta_data.loc[task_idx, :]
                task_obj.task_name = task
                new_task_objects.append(task_obj)

            utils.Debug.vprint("Separated data into {ntask} tasks".format(ntask=len(new_task_objects)), level=0)

            return new_task_objects

    return TaskData
