"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import copy
import gc
import warnings

from inferelator.utils import Debug
from inferelator import workflow
from inferelator import single_cell_workflow
from inferelator.regression import amusr_regression
from inferelator.postprocessing.results_processor_mtl import ResultsProcessorMultiTask

TRANSFER_ATTRIBUTES = ['count_minimum', 'preprocessing_workflow', 'input_dir']
NON_TASK_ATTRIBUTES = ["gold_standard_file", "random_seed", "num_bootstraps"]


class MultitaskLearningWorkflow(single_cell_workflow.SingleCellWorkflow):
    """
    Class that implements multitask learning. Handles loading and validation of multiple data packages.
    """

    _regulator_expression_filter = "intersection"
    _target_expression_filter = "union"

    # Task-specific data
    _n_tasks = None
    _task_design = None
    _task_response = None
    _task_bootstraps = None
    _task_priors = None
    _task_names = None
    _task_objects = None

    task_results = None

    # Axis labels to keep
    _targets = None
    _regulators = None

    # Multi-task result processor
    _result_processor_driver = ResultsProcessorMultiTask

    @property
    def _num_obs(self):
        if self._task_objects is not None:
            return sum([t if t is not None else 0 for t in map(lambda x: x._num_obs, self._task_objects)])
        else:
            return None

    @property
    def _num_genes(self):
        if self._task_objects is not None:
            return max([t if t is not None else 0 for t in map(lambda x: x._num_genes, self._task_objects)])
        else:
            return None

    @property
    def _num_tfs(self):
        if self._task_objects is not None:
            return max([t if t is not None else 0 for t in map(lambda x: x._num_tfs, self._task_objects)])
        else:
            return None

    def set_task_filters(self, regulator_expression_filter=None, target_expression_filter=None):
        """
        Set the filtering criteria for regulators and targets between tasks

        :param regulator_expression_filter:
            "union" includes regulators which are present in any task,
            "intersection" includes regulators which are present in all tasks
        :type regulator_expression_filter: str, optional
        :param target_expression_filter:
            "union" includes targets which are present in any task,
            "intersection" includes targets which are present in all tasks
        :type target_expression_filter: str, optional
        """

        self._set_without_warning("_regulator_expression_filter", regulator_expression_filter)
        self._set_without_warning("_target_expression_filter", target_expression_filter)

    def startup_run(self):
        """
        Load data.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """

        self.get_data()
        self.validate_data()

    def get_data(self):
        # Task data has expression & metadata and may have task-specific files for anything else
        self._load_tasks()

        # Priors, gold standard, tf_names, and gene metadata will be loaded if set
        self.read_tfs()
        self.read_priors()
        self.read_genes()

    def startup_finish(self):
        """
        Process task data and priors.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """

        # Make sure tasks are set correctly
        self._process_default_priors()
        self._process_task_priors()
        self._process_task_data()

    def create_task(self, task_name=None, input_dir=None, expression_matrix_file=None, meta_data_file=None,
                    tf_names_file=None, priors_file=None, gene_names_file=None, gene_metadata_file=None,
                    workflow_type="single-cell", **kwargs):
        """
        Create a task object and set any arguments to this function as attributes of that task object. TaskData objects
        are stored internally in _task_objects.

        :param task_name: A descriptive name for this task
        :type task_name: str
        :param input_dir: A path containing the input files
        :type input_dir: str
        :param expression_matrix_file: Path to the expression data
        :type expression_matrix_file: str
        :param meta_data_file: Path to the meta data
        :type meta_data_file: str, optional
        :param tf_names_file: Path to a list of regulator names to include in the model
        :type tf_names_file: str
        :param priors_file: Path to a prior data file
        :type priors_file: str
        :param gene_metadata_file: Path to a genes annotation file
        :type gene_metadata_file: str, optional
        :param gene_names_file: Path to a list of genes to include in the model (optional)
        :type gene_names_file: str, optional
        :param workflow_type: The type of workflow for data preprocessing.
            "tfa" uses the TFA workflow,
            "single-cell" uses the Single-Cell TFA workflow
        :type workflow_type: str, `inferelator.BaseWorkflow` subclass
        :param kwargs: Any additional arguments are assigned to the task object.
        :return: Returns a task reference which can be additionally modified by calling any valid Workflow function to
            set task parameters
        :rtype: TaskData instance
        """

        # Create a TaskData object from a workflow and set the formal arguments into it
        task_object = create_task_data_object(workflow_class=workflow_type)
        task_object.task_name = task_name
        task_object.input_dir = input_dir if input_dir is not None else self.input_dir
        task_object.expression_matrix_file = expression_matrix_file
        task_object.meta_data_file = meta_data_file
        task_object.tf_names_file = tf_names_file
        task_object.priors_file = priors_file
        task_object.gene_names_file = gene_names_file
        task_object.gene_metadata_file = gene_metadata_file

        # Warn if there is an attempt to set something that isn't supported
        msg = "Task-specific {} is not supported. This setting will be ignored. Set this in the parent workflow."
        for bad in NON_TASK_ATTRIBUTES:
            if bad in kwargs:
                del kwargs[bad]
                warnings.warn(msg.format(bad))

        # Pass forward any kwargs (raising errors if they're for attributes that don't exist)
        for attr, val in kwargs.items():
            if hasattr(task_object, attr):
                setattr(task_object, attr, val)
                task_object.str_attrs.append(attr)
            else:
                raise ValueError("Argument {attr} cannot be set as an attribute".format(attr=attr))

        if self._task_objects is None:
            self._task_objects = [task_object]
        else:
            self._task_objects.append(task_object)

        Debug.vprint(str(task_object), level=0)

        return task_object

    def _load_tasks(self):
        """
        Run load_task_data in all the TaskData objects created with create_task
        """
        if self._task_objects is None:
            raise ValueError("Tasks have not been created with .create_task()")

        for tobj in self._task_objects:
            # Transfer attributes from parent if they haven't been set in the task
            for attr in TRANSFER_ATTRIBUTES:
                try:
                    if getattr(self, attr) is not None and getattr(tobj, attr) is None:
                        setattr(tobj, attr, getattr(self, attr))
                except AttributeError:
                    pass

            # Set the random seed in the task to the same as the parent
            tobj.random_seed = self.random_seed

            # Set the num_bootstraps in the task to the same as the parent
            tobj.num_bootstraps = self.num_bootstraps

        # Run load_task_data and create a list of lists of TaskData objects
        # This allows a TaskData object to copy and split itself if needed
        self._task_objects = [tobj.get_data() for tobj in self._task_objects]

        # Flatten the list
        self._task_objects = [tobj for tobj_list in self._task_objects for tobj in tobj_list]
        self._n_tasks = len(self._task_objects)

    def validate_data(self):
        """
        Make sure that the data that's loaded is acceptable

        This is called when `.startup()` is run. It is not necessary to call separately.

        :raises ValueError: Raises a ValueError if any tasks have invalid priors or gold standard structures
        """
        if self.gold_standard is None:
            raise ValueError("A gold standard must be provided to `gold_standard_file` in MultiTaskLearningWorkflow")

        # Check to see if there are any tasks which don't have priors
        no_priors = sum(map(lambda x: x.priors_data is None, self._task_objects))
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
        if self.gene_names is not None:
            priors = self.prior_manager.filter_priors_to_genes(priors, self.gene_names)

        # Shuffle labels
        if self.shuffle_prior_axis is not None:
            priors = self.prior_manager.shuffle_priors(priors, self.shuffle_prior_axis, self.random_seed)

        # Reset the priors_data in the parent workflow if it exists
        self.priors_data = priors if self.priors_data is not None else None

    def _process_task_priors(self):
        """
        Process & align the default priors for crossvalidation or shuffling
        """
        for task_obj in self._task_objects:

            # Set priors if task-specific priors are not present
            if task_obj.priors_data is None and self.priors_data is None:
                raise ValueError("No priors exist in the main workflow or in tasks")
            elif task_obj.priors_data is None:
                task_obj.priors_data = self.priors_data.copy()

            # Set gene names if task-specific gene names is not present
            if task_obj.gene_names is None:
                task_obj.gene_names = copy.copy(self.gene_names)

            # Set tf_names if task-specific tf names are not present
            if task_obj.tf_names is None:
                task_obj.tf_names = copy.copy(self.tf_names)

            # Process priors in the task data
            task_obj.process_priors_and_gold_standard(gold_standard=self.gold_standard,
                                                      cv_flag=self.split_gold_standard_for_crossvalidation,
                                                      cv_axis=self.cv_split_axis,
                                                      shuffle_priors=self.shuffle_prior_axis)

    def _process_task_data(self):
        """
        Preprocess the individual task data using the TaskData worker into task design and response data. Set
        self.task_design, self.task_response, self.task_bootstraps with lists which contain
        DataFrames.

        Also set self.regulators and self.targets with pd.Indexes that correspond to the genes and tfs to model
        This is chosen based on the filtering strategy set in self.target_expression_filter and
        self.regulator_expression_filter
        """
        self._task_design, self._task_response = [], []
        self._task_bootstraps, self._task_names, self._task_priors = [], [], []
        targets, regulators = [], []

        # Iterate through a list of TaskData objects holding data
        for task_id, task_obj in enumerate(self._task_objects):
            # Get task name from Task
            task_name = task_obj.task_name if task_obj.task_name is not None else str(task_id)

            task_str = "Processing task #{tid} [{t}] {sh}"
            Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.data.shape), level=1)

            # Run the preprocessing workflow
            task_obj.startup_finish()

            # Put the processed data into lists
            self._task_design.append(task_obj.design)
            self._task_response.append(task_obj.response)
            self._task_bootstraps.append(task_obj.get_bootstraps())
            self._task_names.append(task_name)
            self._task_priors.append(task_obj.priors_data)

            regulators.append(task_obj.design.gene_names)
            targets.append(task_obj.response.gene_names)

            task_str = "Processing task #{tid} [{t}] complete [{sh} & {sh2}]"
            Debug.vprint(task_str.format(tid=task_id, t=task_name, sh=task_obj.design.shape,
                                         sh2=task_obj.response.shape), level=1)

        self._targets = amusr_regression.filter_genes_on_tasks(targets, self._target_expression_filter)
        self._regulators = amusr_regression.filter_genes_on_tasks(regulators, self._regulator_expression_filter)

        Debug.vprint("Processed data into design/response [{g} x {k}]".format(g=len(self._targets),
                                                                              k=len(self._regulators)), level=0)

        # Clean up the TaskData objects and force a cyclic collection
        del self._task_objects
        gc.collect()

        # Make sure that the task data files have the correct columns
        for d in self._task_design:
            d.trim_genes(remove_constant_genes=False, trim_gene_list=self._regulators)

        for r in self._task_response:
            r.trim_genes(remove_constant_genes=False, trim_gene_list=self._targets)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors_data):
        """
        Output result report(s) for workflow run.

        This is called when `.startup()` is run. It is not necessary to call separately.
        """
        if self.is_master():
            self.create_output_dir()
            rp = self._result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                               metric=self.metric)
            rp.tasks_names = self._task_names
            self.results = rp.summarize_network(self.output_dir, gold_standard, self._task_priors)
            self.task_results = rp.tasks_networks
            return self.results
        else:
            return None


def create_task_data_object(workflow_class="single-cell"):
    return create_task_data_class(workflow_class=workflow_class)()


def create_task_data_class(workflow_class="single-cell"):
    task_parent = workflow._factory_build_inferelator(regression="base", workflow=workflow_class)

    class TaskData(task_parent):
        """
        TaskData is a workflow object which only loads and preprocesses data from files.
        """

        task_name = None
        tasks_from_metadata = False
        meta_data_task_column = None

        task_workflow_class = str(workflow_class)

        str_attrs = ["input_dir", "expression_matrix_file", "tf_names_file", "meta_data_file", "priors_file"]

        def __str__(self):
            """
            Create a printable report of the settings in this TaskData object

            :return: Settings in str_attrs in a printable string
            :rtype: str
            """

            task_str = "{n}:\n\tWorkflow Class: {cl}\n".format(n=self.task_name, cl=self.task_workflow_class)
            for attr in self.str_attrs:
                try:
                    task_str += "\t{attr}: {val}\n".format(attr=attr, val=getattr(self, attr))
                except AttributeError:
                    task_str += "\t{attr}: Nonexistant\n".format(attr=attr)
            return task_str

        def __init__(self):
            if self._file_format_settings is None:
                self._file_format_settings = dict()

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
            If tasks_from_metadata is not set, the list contains only this task (self)

            :return: List of TaskData objects with loaded data
            :rtype: list(TaskData)
            """
            Debug.vprint("Loading data for task {task_name}".format(task_name=self.task_name))
            super(TaskData, self).get_data()

            if self.tasks_from_metadata:
                return self.separate_tasks_by_metadata()
            else:
                return [self]

        def validate_data(self):
            """
            Don't validate data in TaskData. The parent workflow will check.
            """

            pass

        def set_run_parameters(self):
            """
            Set parameters used during runtime
            """

            warnings.warn("Task-specific `num_bootstraps` and `random_seed` is not supported. Set on parent workflow.")

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

            :param meta_data_column: Meta_data column which corresponds to task ID
            :type meta_data_column: str
            :return new_task_objects: List of the TaskData objects with only one task's data each
            :rtype: list(TaskData)

            """

            if self.data is None:
                raise ValueError("No data has been loaded prior to `separate_tasks_by_metadata`")

            meta_data_column = meta_data_column if meta_data_column is not None else self.meta_data_task_column
            if meta_data_column is None:
                raise ValueError("tasks_from_metadata is set but meta_data_task_column is not")
            elif meta_data_column not in self.data.meta_data:
                msg = "meta_data_task_column is not found in task {t}".format(t=str(self))
                raise ValueError(msg)

            new_task_objects = list()
            tasks = self.data.meta_data[meta_data_column].unique().tolist()
            Debug.vprint("Creating {n} tasks from metadata column {col}".format(n=len(tasks), col=meta_data_column),
                         level=0)

            # Remove data references from self
            data = self.data
            self.data = None

            for task in tasks:
                # Copy this object
                task_obj = copy.deepcopy(self)

                # Get an index of the stuff to keep
                task_idx = data.meta_data[meta_data_column] == task

                # Reset expression matrix, metadata, and task_name in the copy
                task_obj.data = data.subset_copy(row_index=task_idx)
                task_obj.data.name = task
                task_obj.task_name = task
                new_task_objects.append(task_obj)

            Debug.vprint("Separated data into {ntask} tasks".format(ntask=len(new_task_objects)), level=0)

            return new_task_objects

    return TaskData
