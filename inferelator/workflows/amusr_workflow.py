"""
Run Multitask Network Inference with TFA-AMuSR.
"""
import copy
import gc
import warnings
import pandas as pd

from inferelator.utils import (
    Debug,
    join_pandas_index
)
from inferelator import workflow
from inferelator.workflows import single_cell_workflow
from inferelator.regression import amusr_regression
from inferelator.postprocessing.results_processor_mtl import (
    ResultsProcessorMultiTask
)

TRANSFER_ATTRIBUTES = [
    'count_minimum',
    'preprocessing_workflow',
    'input_dir',
    'make_data_noise'
]

NON_TASK_ATTRIBUTES = [
    "random_seed",
    "num_bootstraps"
]

TASK_STR_ATTRS = [
    "input_dir",
    "expression_matrix_file",
    "tf_names_file",
    "meta_data_file",
    "priors_file",
    "gold_standard_file"
]


class MultitaskLearningWorkflow(single_cell_workflow.SingleCellWorkflow):
    """
    Class that implements multitask learning. Handles loading and
    validation of multiple data packages
    """

    _regulator_expression_filter = "intersection"
    _target_expression_filter = "union"

    # Task-specific data
    _n_tasks = None
    _task_design = None
    _task_response = None
    _task_bootstraps = None
    _task_priors = None
    _task_gold_standards = None
    _task_names = None
    _task_objects = None
    _task_genes = None

    task_results = None

    # Axis labels to keep
    _targets = None
    _regulators = None

    # Multi-task result processor
    _result_processor_driver = ResultsProcessorMultiTask

    # Prior noise taskwise flag
    add_prior_noise_to_task_priors = True

    @property
    def _num_obs(self):

        # If the task objects still exist
        # Sum task._num_obs
        if self._task_objects is not None:
            return sum(
                [
                    t if t is not None else 0
                    for t in map(lambda x: x._num_obs, self._task_objects)
                ]
            )

        # If the task design data exists
        # Sum the columns of that
        elif self._task_design is not None:
            return sum(
                [
                    t.num_obs
                    for t in self._task_design
                ]
            )

        # Otherwise return None
        else:
            return None

    @property
    def _num_genes(self):

        if self._gene_names is not None:
            return len(self._gene_names)
        else:
            return None

    @property
    def _num_tfs(self):

        if self._tf_names is not None:
            return len(self._tf_names)
        else:
            return None

    @property
    def _tf_names(self):

        # Use column names from design data objects which have TF columns
        if self._task_design is not None:
            task_ref = [
                t.gene_names
                for t in self._task_design
            ]

        # Use the tf_names list if design isn't calculated yet
        elif self._task_objects is not None:
            task_ref = [
                pd.Index(t.tf_names)
                for t in self._task_objects if t.tf_names is not None
            ]

        else:
            return None

        # Intersect each task's tf indices
        return join_pandas_index(
            *task_ref,
            method='intersection'
        )

    @property
    def _gene_names(self):

        # Use column names from response data objects which have gene columns
        if self._task_response is not None:
            task_ref = [
                t.gene_names
                for t in self._task_response
            ]

        # Use the raw gene expression data if response isn't calculated yet
        elif self._task_objects is not None:
            task_ref = [
                t.data.gene_names
                for t in self._task_objects if t.data is not None
            ]

        else:
            return None

        # Intersect each task's gene indices
        return join_pandas_index(
            *task_ref,
            method='intersection'
        )

    def set_task_filters(
        self,
        regulator_expression_filter=None,
        target_expression_filter=None
    ):
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

        self._set_without_warning(
            "_regulator_expression_filter",
            regulator_expression_filter
        )

        self._set_without_warning(
            "_target_expression_filter",
            target_expression_filter
        )

    def startup_run(self):
        """
        Load data.

        This is called when `.startup()` is run.
        It is not necessary to call separately.
        """

        self.get_data()

        # Set the random seed in the task to the same as the parent
        for tobj in self._task_objects:
            tobj.random_seed = self.random_seed

        self.validate_data()

    def get_data(self):
        # Task data has expression & metadata and may have task-specific
        # files for anything else
        self._load_tasks()

        # Priors, gold standard, tf_names, and gene metadata
        # will be loaded if set
        self.read_tfs()
        self.read_priors()
        self.read_genes()

    def startup_finish(self):
        """
        Process task data and priors.

        This is called when `.startup()` is run.
        It is not necessary to call separately.
        """

        # Make sure tasks are set correctly
        self._process_default_priors()
        self._process_task_priors()
        self._process_task_data()

    def create_task(
        self,
        task_name=None,
        input_dir=None,
        expression_matrix_file=None,
        meta_data_file=None,
        tf_names_file=None,
        priors_file=None,
        gold_standard_file=None,
        gene_names_file=None,
        gene_metadata_file=None,
        workflow_type="single-cell",
        **kwargs
    ):
        """
        Create a task object and set any arguments to this function as
        attributes of that task object. TaskData objects
        are stored internally in _task_objects.

        :param task_name: A descriptive name for this task
        :type task_name: str
        :param input_dir: A path containing the input files
        :type input_dir: str
        :param expression_matrix_file: Path to the expression data
        :type expression_matrix_file: str
        :param meta_data_file: Path to the meta data
        :type meta_data_file: str, optional
        :param tf_names_file: Path to a list of regulator names to include
            in the model
        :type tf_names_file: str
        :param priors_file: Path to a prior data file
        :type priors_file: str
        :param gene_metadata_file: Path to a genes annotation file
        :type gene_metadata_file: str, optional
        :param gene_names_file: Path to a list of genes to include in
            the model (optional)
        :type gene_names_file: str, optional
        :param workflow_type: The type of workflow for data preprocessing.
            "tfa" uses the TFA workflow,
            "single-cell" uses the Single-Cell TFA workflow
        :type workflow_type: str, `inferelator.BaseWorkflow` subclass
        :param kwargs: Any additional arguments are assigned to thetask object
        :return: Returns a task reference which can be additionally modified
            by calling any valid Workflow function to set task parameters
        :rtype: TaskData instance
        """

        # Create a TaskData object from a workflow and set the
        # formal arguments into it
        task_object = create_task_data_object(workflow_class=workflow_type)
        task_object.task_name = task_name

        if input_dir is not None:
            task_object.input_dir = input_dir
        else:
            task_object.input_dir = self.input_dir

        task_object.expression_matrix_file = expression_matrix_file
        task_object.meta_data_file = meta_data_file
        task_object.tf_names_file = tf_names_file
        task_object.priors_file = priors_file
        task_object.gene_names_file = gene_names_file
        task_object.gene_metadata_file = gene_metadata_file
        task_object.gold_standard_file = gold_standard_file

        # Warn if there is an attempt to set something that isn't supported
        for bad in NON_TASK_ATTRIBUTES:
            if bad in kwargs:
                del kwargs[bad]
                warnings.warn(
                    f"Task-specific {bad} is not supported. "
                    "This setting will be ignored. "
                    "Set this in the parent workflow."
                )

        # Pass forward any kwargs (raising errors if they're for
        # attributes that don't exist)
        for attr, val in kwargs.items():
            if hasattr(task_object, attr):
                setattr(task_object, attr, val)
                task_object.str_attrs.append(attr)
            else:
                raise ValueError(
                    f"Argument {attr} cannot be set as an attribute"
                )

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
            # Transfer attributes from parent if they haven't been
            # set in the task
            for attr in TRANSFER_ATTRIBUTES:
                try:
                    if getattr(tobj, attr) is None:
                        setattr(tobj, attr, getattr(self, attr))
                except AttributeError:
                    pass

            # Set the num_bootstraps in the task to the same as the parent
            tobj.num_bootstraps = self.num_bootstraps

        # Run load_task_data and create a list of lists of TaskData objects
        # This allows a TaskData object to copy and split itself if needed
        self._task_objects = [tobj.get_data() for tobj in self._task_objects]

        # Flatten the list
        self._task_objects = [
            t for t_list in self._task_objects
            for t in t_list
        ]
        self._n_tasks = len(self._task_objects)

    def validate_data(self):
        """
        Make sure that the data that's loaded is acceptable

        This is called when `.startup()` is run. It is not necessary
        to call separately.

        :raises ValueError: Raises a ValueError if any tasks have invalid
        priors or gold standard structures
        """

        super().validate_data(check_prior=False)

        # Check to see if there are any tasks which don't have priors
        no_priors = sum(map(
            lambda x: x.priors_data is None,
            self._task_objects
        ))

        _missing_prior = no_priors > 0 and self.priors_data is None

        if _missing_prior and self.use_no_prior:
            self.priors_data = self._create_null_prior(
                    self._gene_names,
                    self.tf_names
                )

            warnings.warn(
                f"A null prior will be used for {no_priors}"
                " tasks which have no prior set"
            )

        elif _missing_prior:
            raise ValueError(
                f"{no_priors} tasks have no priors "
                "(no default prior is set). "
                "worker.set_network_data_flags(use_no_prior=True) "
                "will override this error."
            )

    def _process_default_priors(self):
        """
        Process the default priors in the parent workflow for
        crossvalidation or shuffling
        """

        # Use priors if given to the MTL workflow
        if self.priors_data is not None:
            priors = self.priors_data

        # If they all have priors don't worry about it -
        # use a 0 prior here for crossvalidation selection if needed
        elif self.priors_data is None and self.gold_standard is not None:
            priors = pd.DataFrame(
                0,
                index=self.gold_standard.index,
                columns=self.gold_standard.columns
            )

        elif self.priors_data is None and self.tf_names is not None:
            priors = pd.DataFrame(
                0,
                index=self._gene_names,
                columns=self.tf_names
            )

        # If there's no gold standard or use_no_prior isn't set,
        # raise a RuntimeError
        else:
            raise RuntimeError(
                "No base prior or gold standard or TF list has been provided."
            )

        # Crossvalidation
        if self.split_gold_standard_for_crossvalidation:
            priors, self.gold_standard = self.prior_manager.cross_validate_gold_standard(
                priors,
                self.gold_standard,
                self.cv_split_axis,
                self.cv_split_ratio,
                self.random_seed
            )

        # Filter to regulators
        if self.tf_names is not None:
            priors = self.prior_manager.filter_to_tf_names_list(
                priors,
                self.tf_names
            )

        # Filter to targets
        if self.gene_names is not None:
            priors = self.prior_manager.filter_priors_to_genes(
                priors,
                self.gene_names
            )

        # Shuffle labels
        if self.shuffle_prior_axis is not None:
            priors = self.prior_manager.shuffle_priors(
                priors,
                self.shuffle_prior_axis,
                self.random_seed
            )

        # Add prior noise now (to the base prior)
        # if add_prior_noise_to_task_priors is False
        # Otherwise add later to the task priors
        # (will be different for each task)
        if self.add_prior_noise is not None and not self.add_prior_noise_to_task_priors:
            priors = self.prior_manager.add_prior_noise(
                priors,
                self.add_prior_noise,
                self.random_seed
            )

            _has_prior = [
                t.task_name
                for t in self._task_objects
                if t.priors_data is not None
            ]

            if len(_has_prior) > 0:
                Debug.vprint(
                    f"Overriding task priors in {_has_prior} because "
                    "add_prior_noise_to_task_priors is False",
                    level=0
                )

                for t in self._task_objects:
                    t.priors_data = priors.copy()

        # Reset the priors_data in the parent workflow if it exists
        self.priors_data = priors if self.priors_data is not None else None

    def _process_task_priors(self):
        """
        Process & align the default priors for crossvalidation or shuffling
        """
        for task_obj in self._task_objects:

            # Set priors if task-specific priors are not present
            if task_obj.priors_data is None and self.priors_data is None:
                raise ValueError(
                    "No priors exist in the main workflow or in tasks"
                )

            elif task_obj.priors_data is None:
                task_obj.priors_data = self.priors_data.copy()

            # Set gene names if task-specific gene names is not present
            if task_obj.gene_names is None:
                task_obj.gene_names = copy.deepcopy(self.gene_names)

            # Set tf_names if task-specific tf names are not present
            if task_obj.tf_names is None:
                task_obj.tf_names = copy.deepcopy(self.tf_names)

            if self.add_prior_noise_to_task_priors is True:
                _add_prior_noise = self.add_prior_noise
            else:
                _add_prior_noise = None
            # Process priors in the task data

            task_obj.process_priors_and_gold_standard(
                gold_standard=self.gold_standard,
                cv_flag=self.split_gold_standard_for_crossvalidation,
                cv_axis=self.cv_split_axis,
                shuffle_priors=self.shuffle_prior_axis,
                add_prior_noise=_add_prior_noise
            )

    def _process_task_data(self):
        """
        Preprocess the individual task data using the TaskData worker
        into task design and response data. Set self.task_design,
        self.task_response, self.task_bootstraps with lists which contain
        DataFrames.

        Also set self.regulators and self.targets with pd.Indexes that
        correspond to the genes and tfs to model.

        This is chosen based on the filtering strategy set in
        self.target_expression_filter and self.regulator_expression_filter
        """

        # Create empty task data lists
        for attr in [
            "_task_design",
            "_task_response",
            "_task_bootstraps",
            "_task_names",
            "_task_priors",
            "_task_gold_standards"
        ]:
            setattr(self, attr, [])

        targets, regulators = [], []

        # Iterate through a list of TaskData objects holding data
        for task_id, task_obj in enumerate(self._task_objects):
            # Get task name from Task

            if task_obj.task_name is not None:
                task_name = task_obj.task_name
            else:
                task_name = str(task_id)

            Debug.vprint(
                f"Processing task #{task_id} [{task_name}] "
                f"{task_obj.data.shape}",
                level=1
            )

            # Run the preprocessing workflow
            task_obj.startup_finish()

            # Put the processed data into lists
            self._task_design.append(task_obj.design)
            self._task_response.append(task_obj.response)
            self._task_bootstraps.append(task_obj.get_bootstraps())
            self._task_names.append(task_name)
            self._task_priors.append(task_obj.priors_data)

            if task_obj.gold_standard is not None:
                self._task_gold_standards.append(task_obj.gold_standard)
            else:
                self._task_gold_standards.append(self.gold_standard.copy())

            regulators.append(task_obj.design.gene_names)
            targets.append(task_obj.response.gene_names)

            Debug.vprint(
                f"Processing task #{task_id} [{task_name}] complete "
                f"[{task_obj.design.shape} & {task_obj.response.shape}]",
                level=1
            )

        self._targets = amusr_regression.filter_genes_on_tasks(
            targets,
            self._target_expression_filter
        )

        self._regulators = amusr_regression.filter_genes_on_tasks(
            regulators,
            self._regulator_expression_filter
        )

        self._task_genes = amusr_regression.genes_tasks(
            self._targets,
            self._task_response
        )

        Debug.vprint(
            "Processed data into design/response "
            f"[{len(self._targets)} x {len(self._regulators)}]",
            level=0
        )

        # Clean up the TaskData objects and force a cyclic collection
        del self._task_objects
        gc.collect()

        self._align_design_response()

    def _align_design_response(self):

        # Make sure that the task data files have the correct columns
        for d in self._task_design:
            d.trim_genes(
                remove_constant_genes=False,
                trim_gene_list=self._regulators
            )

        for r in self._task_response:
            r.trim_genes(
                remove_constant_genes=False,
                trim_gene_list=self._targets
            )

    def emit_results(
        self,
        betas,
        rescaled_betas,
        gold_standard,
        priors_data,
        full_model=None,
        full_exp_var=None
    ):
        """
        Output result report(s) for workflow run.

        This is called when `.startup()` is run.
        It is not necessary to call separately.
        """

        self.create_output_dir()

        rp = self._result_processor_driver(
            betas,
            rescaled_betas,
            filter_method=self.gold_standard_filter_method,
            metric=self.metric,
            task_names=self._task_names
        )

        self.results = rp.summarize_network(
            self.output_dir,
            gold_standard,
            self._task_priors,
            task_gold_standards=self._task_gold_standards,
            full_model_betas=full_model,
            full_model_var_exp=full_exp_var
        )

        self.task_results = rp.tasks_networks

        return self.results


def create_task_data_object(workflow_class="single-cell"):
    return create_task_data_class(workflow_class=workflow_class)()


def create_task_data_class(workflow_class="single-cell"):
    """
    Factory function for building task-specific workflows

    :param workflow_class: Task workflow class to build,
        defaults to "single-cell"
    :type workflow_class: str, optional
    :return: TaskData class built from the parent class
    :rtype: TaskData
    """

    task_parent = workflow._factory_build_inferelator(
        regression="base",
        workflow=workflow_class
    )

    class TaskData(task_parent):
        """
        TaskData is a workflow object which only loads and preprocesses
        data from files.
        """

        task_name = None
        tasks_from_metadata = False
        meta_data_task_column = None

        task_workflow_class = str(workflow_class)

        str_attrs = copy.copy(TASK_STR_ATTRS)

        def __str__(self):
            """
            Create a printable report of the settings in this TaskData object

            :return: Settings in str_attrs in a printable string
            :rtype: str
            """

            task_str = f"{self.task_name}:"
            task_str += f"\n\tWorkflow Class: {self.task_workflow_class}\n"
            for attr in self.str_attrs:
                task_str += f"\t{attr}: {getattr(self, attr, 'NA')}\n"

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
            Load all the data and then return a list of references
            to TaskData objects

            There will be multiple objects returned if
            tasks_from_metadata is set.

            If tasks_from_metadata is not set, the list contains
            only this task (self)

            :return: List of TaskData objects with loaded data
            :rtype: list(TaskData)
            """
            Debug.vprint(f"Loading data for task {self.task_name}")
            super(TaskData, self).get_data()

            self.data.name = self.task_name

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

            warnings.warn(
                "Task-specific `num_bootstraps` and `random_seed` are not "
                "supported; set on parent workflow."
            )

        def process_priors_and_gold_standard(
            self,
            gold_standard=None,
            cv_flag=None,
            cv_axis=None,
            shuffle_priors=None,
            add_prior_noise=None
        ):
            """
            Make sure that the priors for this task are correct

            This will remove circularity from the task priors based
            on the parent gold standard
            """

            gold_standard = self.gold_standard if gold_standard is None else gold_standard
            cv_flag = self.split_gold_standard_for_crossvalidation if cv_flag is None else cv_flag
            cv_axis = self.cv_split_axis if cv_axis is None else cv_axis
            shuffle_priors = self.shuffle_prior_axis if shuffle_priors is None else shuffle_priors
            add_prior_noise = self.add_prior_noise if add_prior_noise is None else add_prior_noise

            # Remove circularity from the gold standard
            if cv_flag:
                self.priors_data, _ = self.prior_manager._remove_prior_circularity(
                    self.priors_data,
                    gold_standard,
                    split_axis=cv_axis
                )

            if self.tf_names is not None:
                self.priors_data = self.prior_manager.filter_to_tf_names_list(
                    self.priors_data,
                    self.tf_names
                )

            # Filter priors and expression to a list of genes
            self.filter_to_gene_list()

            # Shuffle prior labels
            if shuffle_priors is not None:
                self.priors_data = self.prior_manager.shuffle_priors(
                    self.priors_data,
                    shuffle_priors,
                    self.random_seed
                )

            if add_prior_noise is not None:
                self.priors_data = self.prior_manager.add_prior_noise(
                    self.priors_data,
                    add_prior_noise,
                    self.random_seed
                )

            if min(self.priors_data.shape) == 0:
                raise ValueError(
                    f"Priors for task {self.task_name} have an axis of length 0"
                )

        def separate_tasks_by_metadata(self, meta_data_column=None):
            """
            Take a single expression matrix and break it into multiple
            dataframes based on meta_data. Return a list of TaskData
            objects which have the task-specific data loaded into them

            :param meta_data_column: Meta_data column which corresponds
                to task ID
            :type meta_data_column: str
            :return new_task_objects: List of the TaskData objects with
                only one task's data each
            :rtype: list(TaskData)

            """

            # Check to make sure data is loaded
            if self.data is None:
                raise ValueError(
                    "No data has been loaded; call `get_data` before `separate_tasks_by_metadata`"
                )

            if meta_data_column is None:
                meta_data_column = self.meta_data_task_column

            # Check to make sure meta_data_column is valid
            if meta_data_column is None:
                raise ValueError("tasks_from_metadata is set but meta_data_task_column is not")

            elif meta_data_column not in self.data.meta_data:
                raise ValueError(
                    f"meta_data_task_column is not found in task {str(self)}"
                )

            tasks = self.data.meta_data[meta_data_column].unique().tolist()

            Debug.vprint(
                f"Creating {len(tasks)} tasks from metadata column {meta_data_column}",
                level=0
            )

            # Remove data references from self
            data = self.data
            self.data = None

            def _make_task_subobject(task):
                # Copy this object
                task_obj = copy.deepcopy(self)

                 # Reset expression matrix, metadata, and task_name in the copy
                task_obj.data = data.subset_copy(
                    row_index=data.meta_data[meta_data_column] == task
                )

                # Set task name
                task_obj.data.name = task
                task_obj.task_name = task

                return task_obj

            new_task_objects = [_make_task_subobject(t) for t in tasks]

            Debug.vprint(
                f"Separated data into {len(new_task_objects)} tasks",
                level=0
            )

            return new_task_objects

    return TaskData
