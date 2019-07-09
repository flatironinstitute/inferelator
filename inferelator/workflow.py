"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.

The base workflow has functions for loading and managing data
But does not have any functions for regression or analysis
"""
from __future__ import unicode_literals, print_function

from inferelator import utils
from inferelator import default
from inferelator.preprocessing.priors import ManagePriors
from inferelator.preprocessing.metadata_parser import MetadataHandler
from inferelator.regression.base_regression import RegressionWorkflow

from inferelator.distributed.inferelator_mp import MPControl

import inspect
import numpy as np
import os
import datetime
import pandas as pd


class WorkflowBase(object):
    # Paths to the input and output locations
    input_dir = None
    output_dir = None

    # Settings that will be used by pd.read_table to import data files
    file_format_settings = default.DEFAULT_PD_INPUT_SETTINGS
    # A dict, keyed by file name, of settings to override the defaults in file_format_settings
    # Used when input files are perhaps not processed into perfect TSVs
    file_format_overrides = dict()

    # File names for each of the data files which can be used in the inference workflow
    expression_matrix_file = default.DEFAULT_EXPRESSION_FILE
    tf_names_file = default.DEFAULT_TFNAMES_FILE
    meta_data_file = default.DEFAULT_METADATA_FILE
    priors_file = default.DEFAULT_PRIORS_FILE
    gold_standard_file = default.DEFAULT_GOLDSTANDARD_FILE

    # Gene list & associated metadata
    gene_metadata_file = None
    gene_metadata = None
    gene_list_index = default.DEFAULT_GENE_LIST_INDEX_COLUMN

    # Flags to control splitting priors into a prior/gold-standard set
    split_gold_standard_for_crossvalidation = False
    cv_split_ratio = default.DEFAULT_GS_SPLIT_RATIO
    cv_split_axis = default.DEFAULT_GS_SPLIT_AXIS
    shuffle_prior_axis = None

    # Flag to identify orientation of the expression matrix (True for samples x genes & False for genes x samples)
    expression_matrix_columns_are_genes = False  # bool

    # Flag to extract metadata from specific columns of the expression matrix instead of a separate file
    extract_metadata_from_expression_matrix = default.DEFAULT_EXTRACT_METADATA_FROM_EXPR  # bool
    expression_matrix_metadata = default.DEFAULT_EXPRESSION_MATRIX_METADATA  # str

    # The random seed for sampling, etc
    random_seed = default.DEFAULT_RANDOM_SEED

    # The number of inference bootstraps to run
    num_bootstraps = default.DEFAULT_NUM_BOOTSTRAPS

    # Computed data structures [G: Genes, K: Predictors, N: Conditions
    expression_matrix = None  # expression_matrix dataframe [G x N]
    tf_names = None  # tf_names list [k,]
    meta_data = None  # meta data dataframe [G x ?]
    priors_data = None  # priors data dataframe [G x K]
    gold_standard = None  # gold standard dataframe [G x K]

    # Multiprocessing controller
    initialize_mp = True
    multiprocessing_controller = None

    # Prior manager
    prior_manager = ManagePriors

    def __init__(self):
        # Get environment variables
        self.get_environmentals()

    def initialize_multiprocessing(self):
        """
        Register the multiprocessing controller if set and run .connect()
        """
        if self.multiprocessing_controller is not None:
            MPControl.set_multiprocess_engine(self.multiprocessing_controller)
        MPControl.connect()

    def get_environmentals(self):
        """
        Load environmental variables into class variables
        """
        for k, v in utils.slurm_envs(default.SBATCH_VARS_FOR_WORKFLOW).items():
            setattr(self, k, v)

    def startup(self):
        """
        Startup by preprocessing all data into a ready format for regression.
        """
        if self.initialize_mp and not MPControl.is_initialized:
            self.initialize_multiprocessing()
        self.startup_run()
        self.startup_finish()

    def startup_run(self):
        """
        Execute any data preprocessing necessary before regression. Startup_run is mostly for reading in data
        """
        raise NotImplementedError  # implement in subclass

    def startup_finish(self):
        """
        Execute any data preprocessing necessary before regression. Startup_finish is mostly for preprocessing data
        prior to regression
        """
        raise NotImplementedError  # implement in subclass

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        raise NotImplementedError  # implement in subclass

    def get_data(self):
        """
        Read data files in to data structures.
        """

        self.read_expression()
        self.read_tfs()
        self.read_metadata()
        self.read_genes()
        self.read_priors()

        # Transpose expression data to [Genes x Samples] if the columns_are_genes flag is set
        self.transpose_expression_matrix()

    def transpose_expression_matrix(self):
        # Transpose expression data
        if self.expression_matrix_columns_are_genes:
            self.expression_matrix = self.expression_matrix.transpose()
            utils.Debug.vprint("Transposing expression matrix to {sh}".format(sh=self.expression_matrix.shape), level=2)

    def read_expression(self, file=None):
        """
        Read expression matrix file into expression_matrix
        """
        file = file if file is not None else self.expression_matrix_file
        utils.Debug.vprint("Loading expression data file {file}".format(file=file), level=1)
        self.expression_matrix = self.input_dataframe(file)

    def read_tfs(self, file=None):
        """
        Read tf names file into tf_names
        """

        # Load the class variable if no file is passed
        file = self.tf_names_file if file is None else file
        utils.Debug.vprint("Loading TF feature names from file {file}".format(file=file), level=1)
        # Read in a dataframe with no header or index
        tfs = self.input_dataframe(file, header=None, index_col=None)

        # Cast the dataframe into a list
        assert tfs.shape[1] == 1
        self.tf_names = tfs.values.flatten().tolist()

    def read_metadata(self, file=None):
        """
        Read metadata file into meta_data or make fake metadata
        """

        file = file if file is not None else self.meta_data_file

        # If the metadata is embedded in the expression matrix, extract it
        if self.extract_metadata_from_expression_matrix:
            utils.Debug.vprint("Slicing metadata from expression matrix", level=1)
            self.expression_matrix, self.meta_data = self.dataframe_split(self.expression_matrix,
                                                                          self.expression_matrix_metadata)
        elif file is not None:
            utils.Debug.vprint("Loading metadata file {file}".format(file=file), level=1)
            self.meta_data = self.input_dataframe(file, index_col=None)
        else:
            utils.Debug.vprint("No metadata provided. Creating a generic metadata", level=0)
            self.meta_data = MetadataHandler.get_handler().create_default_meta_data(self.expression_matrix)

    def read_genes(self, file=None):
        """
        Read in a list of genes which should be modeled for network inference
        """

        file = file if file is not None else self.gene_metadata_file

        if file is not None:
            utils.Debug.vprint("Loading Gene metadata from file {file}".format(file=file), level=1)
            self.gene_metadata = self.input_dataframe(self.gene_metadata_file, index_col=None)
        else:
            return

        if self.gene_list_index is None or self.gene_list_index not in self.gene_metadata.columns:
            raise ValueError("The gene list file must have headers and workflow.gene_list_index must be a valid column")

    def read_priors(self, priors_file=None, gold_standard_file=None):
        """
        Read in the priors and gold standard files
        """
        priors_file = priors_file if priors_file is not None else self.priors_file
        gold_standard_file = gold_standard_file if gold_standard_file is not None else self.gold_standard_file

        if priors_file is not None:
            utils.Debug.vprint("Loading prior data from file {file}".format(file=priors_file), level=1)
            self.priors_data = self.input_dataframe(priors_file)
        if gold_standard_file is not None:
            utils.Debug.vprint("Loading gold_standard data from file {file}".format(file=gold_standard_file), level=1)
            self.gold_standard = self.input_dataframe(gold_standard_file)

        if self.priors_data is None and self.gold_standard is None:
            raise ValueError("No gold standard or priors have been provided")

    def process_priors_and_gold_standard(self):
        """
        Run split, shuffle, etc called for by the workflow flags
        This also filters the expression matrix to the list of genes to model
        """

        # Split gold standard for cross-validation
        if self.split_gold_standard_for_crossvalidation:
            self.priors_data, self.gold_standard = self.prior_manager.cross_validate_gold_standard(self.priors_data,
                                                                                                   self.gold_standard,
                                                                                                   self.cv_split_axis,
                                                                                                   self.cv_split_ratio,
                                                                                                   self.random_seed)

        # Filter priors to a list of regulators
        if self.tf_names is not None:
            self.priors_data = self.prior_manager.filter_to_tf_names_list(self.priors_data, self.tf_names)

        # Filter priors and expression to a list of genes
        if self.gene_metadata is not None and self.gene_list_index is not None:
            gene_list = self.gene_metadata[self.gene_list_index].tolist()
            self.priors_data, self.expression_matrix = self.prior_manager.filter_to_gene_list(self.priors_data,
                                                                                              self.expression_matrix,
                                                                                              gene_list)

        # Shuffle prior labels
        if self.shuffle_prior_axis is not None:
            self.priors_data = self.prior_manager.shuffle_priors(self.priors_data,
                                                                 self.shuffle_prior_axis,
                                                                 self.random_seed)

        # Check for duplicates or whatever
        self.priors_data, self.gold_standard = self.prior_manager.validate_priors_gold_standard(self.priors_data,
                                                                                                self.gold_standard)

    def align_priors_and_expression(self):
        """
        Align prior to the expression matrix
        """
        self.priors_data = self.prior_manager.align_priors_to_expression(self.priors_data, self.expression_matrix)

    def input_path(self, filename):
        """
        Join filename to input_dir
        """

        return os.path.abspath(os.path.expanduser(os.path.join(self.input_dir, filename)))

    def input_dataframe(self, filename, **kwargs):
        """
        Read a file in as a pandas dataframe
        """

        # Set defaults for index_col and header
        kwargs['index_col'] = kwargs.pop('index_col', 0)
        kwargs['header'] = kwargs.pop('header', 0)

        # Use any kwargs for this function and any file settings from default
        file_settings = self.file_format_settings.copy()
        file_settings.update(kwargs)

        # Update the file settings with anything that's in file-specific overrides
        if filename in self.file_format_overrides:
            file_settings.update(self.file_format_overrides[filename])

        utils.Debug.vprint("Loading data file: {a}".format(a=self.input_path(filename)), level=2)
        # Load a dataframe
        return pd.read_csv(self.input_path(filename), **file_settings)

    def append_to_path(self, var_name, to_append):
        """
        Add a string to an existing path variable in class
        """
        path = getattr(self, var_name, None)
        if path is None:
            raise ValueError("Cannot append {to_append} to {var_name} (Which is None)".format(to_append=to_append,
                                                                                              var_name=var_name))
        setattr(self, var_name, os.path.join(path, to_append))


    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(self.response.shape[1])
        random_state = np.random.RandomState(seed=self.random_seed)
        return random_state.choice(col_range, size=(self.num_bootstraps, self.response.shape[1])).tolist()

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass

    @staticmethod
    def is_master():
        """
        Return True if this is the master thread
        """
        return MPControl.is_master

    def create_output_dir(self):
        """
        Set a default output_dir if nothing is set. Create the path if it doesn't exist.
        """
        if self.output_dir is None:
            new_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_dir = self.make_path_safe(os.path.join(self.input_dir, new_path))
        else:
            self.output_dir = self.make_path_safe(self.output_dir)

        try:
            os.makedirs(os.path.expanduser(self.output_dir))
        except FileExistsError:
            pass

    @staticmethod
    def make_path_safe(path):
        """
        Expand relative paths to absolute paths. Pass None through.
        :param path: str
        :return: str
        """
        if path is not None:
            return os.path.abspath(os.path.expanduser(path))
        else:
            return None

    @staticmethod
    def dataframe_split(data_frame, remove_columns):
        """
        Take a dataframe and extract specific columns. Return the dataframe, minus those columns, and a second
        dataframe which is only those columns.
        :param data_frame: pd.DataFrame
        :param meta_columns: list(str)
        :return data_frame, data_frame_two: pd.DataFrame, pd.DataFrame
        """

        data_frame_two = data_frame.loc[:, remove_columns].copy()
        data_frame = data_frame.drop(remove_columns, axis=1)
        return data_frame, data_frame_two


def create_inferelator_workflow(regression=RegressionWorkflow, workflow=WorkflowBase):
    """
    This is the factory method to create workflow classes that combine preprocessing and postprocessing (from workflow)
    with a regression method (from regression)

    :param regression: RegressionWorkflow subclass
        A class object which implements the run_regression and run_bootstrap methods for a specific regression strategy
    :param workflow: WorkflowBase subclass
        A class object which implements the necessary data loading and preprocessing to create design & response data
        for the regression strategy, and then the postprocessing to turn regression betas into a network
    :return RegressWorkflow:
        This returns an uninstantiated class which is the multi-inheritance result of both the regression workflow and
        the preprocessing/postprocessing workflow
    """

    # Decide which preprocessing/postprocessing workflow to use
    # String arguments are parsed for convenience in the run script
    if utils.is_string(workflow):
        if workflow == "base":
            workflow_class = WorkflowBase
        elif workflow == "tfa":
            from inferelator.tfa_workflow import TFAWorkFlow
            workflow_class = TFAWorkFlow
        elif workflow == "amusr" or workflow == "multitask":
            from inferelator.amusr_workflow import MultitaskLearningWorkflow
            workflow_class = MultitaskLearningWorkflow
        elif workflow == "single-cell":
            from inferelator.single_cell_workflow import SingleCellWorkflow
            workflow_class = SingleCellWorkflow
        else:
            raise ValueError("{val} is not a string that can be mapped to a workflow class".format(val=workflow))
    # Or just use a workflow class directly
    elif inspect.isclass(workflow) and issubclass(workflow, WorkflowBase):
        workflow_class = workflow
    else:
        raise ValueError("Workflow must be a string that maps to a workflow class or an actual workflow class")

    # Decide which regression workflow to use
    # Return just the workflow if regression is set to None
    if regression is None:
        return workflow_class
    # String arguments are parsed for convenience in the run script
    elif utils.is_string(regression):
        if regression == "bbsr":
            from inferelator.regression.bbsr_python import BBSRRegressionWorkflow
            regression_class = BBSRRegressionWorkflow
        elif regression == "elasticnet":
            from inferelator.regression.elasticnet_python import ElasticNetWorkflow
            regression_class = ElasticNetWorkflow
        elif regression == "amusr":
            from inferelator.regression.amusr_regression import AMUSRRegressionWorkflow
            regression_class = AMUSRRegressionWorkflow
        else:
            raise ValueError("{val} is not a string that can be mapped to a regression class".format(val=regression))
    # Or just use a regression class directly
    elif inspect.isclass(regression) and issubclass(regression, RegressionWorkflow):
        regression_class = regression
    else:
        raise ValueError("Regression must be a string that maps to a regression class or an actual regression class")

    class RegressWorkflow(regression_class, workflow_class):
        regression_type = regression_class

    return RegressWorkflow


def inferelator_workflow(regression=RegressionWorkflow, workflow=WorkflowBase):
    """
    Create and instantiate a workflow

    :param regression: RegressionWorkflow subclass
        A class object which implements the run_regression and run_bootstrap methods for a specific regression strategy
    :param workflow: WorkflowBase subclass
        A class object which implements the necessary data loading and preprocessing to create design & response data
        for the regression strategy, and then the postprocessing to turn regression betas into a network
    :return RegressWorkflow:
        This returns an initialized object which is the multi-inheritance result of both the regression workflow and
        the preprocessing/postprocessing workflow
    """
    return create_inferelator_workflow(regression=regression, workflow=workflow)()
