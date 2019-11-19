"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.

The base workflow has functions for loading and managing data
But does not have any functions for regression or analysis
"""
from __future__ import unicode_literals, print_function
from builtins import FileExistsError

import datetime
import inspect
import os
import warnings
import copy

import numpy as np
import pandas as pd

from inferelator import default
from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.distributed.inferelator_mp import MPControl
from inferelator.preprocessing.metadata_parser import MetadataHandler
from inferelator.preprocessing.priors import ManagePriors
from inferelator.regression.base_regression import RegressionWorkflow
from inferelator.postprocessing.results_processor import ResultsProcessor

DEFAULT_PANDAS_TSV_SETTINGS = dict(sep="\t", index_col=0, header=0)


class WorkflowBaseLoader(object):
    """
    WorkflowBaseLoader is the class to load raw data. It does no processing.
    """
    # Paths to the input and output locations
    input_dir = None
    output_dir = None

    # File names for each of the data files which can be used in the inference workflow
    expression_matrix_file = None
    tf_names_file = None
    meta_data_file = None
    priors_file = None
    gold_standard_file = None

    # Metadata handler
    metadata_handler = "branching"

    # Gene list & associated metadata
    gene_metadata_file = None
    gene_metadata = None
    gene_list_index = None

    # Loaded data structures [G: Genes, K: Predictors, N: Conditions
    expression_matrix = None  # expression_matrix dataframe [G x N]
    tf_names = None  # tf_names list [k,]
    meta_data = None  # meta data dataframe [G x ?]
    priors_data = None  # priors data dataframe [G x K]
    gold_standard = None  # gold standard dataframe [G x K]

    # Flag to identify orientation of the expression matrix (True for samples x genes & False for genes x samples)
    expression_matrix_columns_are_genes = False  # bool

    # Flag to extract metadata from specific columns of the expression matrix instead of a separate file
    extract_metadata_from_expression_matrix = False  # bool
    expression_matrix_metadata = None  # str

    # Flag to indicate the inferelator should be run without existing network data
    use_no_prior = False  # bool
    use_no_gold_standard = False  # bool

    # Settings that will be used by pd.read_table to import data files
    _file_format_settings = None

    def __init__(self):
        if self._file_format_settings is None:
            self._file_format_settings = dict()

    def set_file_paths(self, input_dir=None, output_dir=None, expression_matrix_file=None, tf_names_file=None,
                       meta_data_file=None, priors_file=None, gold_standard_file=None, gene_metadata_file=None):
        """
        Set the file paths necessary for the inferelator to run
        :param input_dir: str
        :param output_dir: str
        :param expression_matrix_file: str
        :param tf_names_file: str
        :param meta_data_file: str
        :param priors_file: str
        :param gold_standard_file: str
        :param gene_metadata_file: str
        """

        self._set_with_warning("input_dir", input_dir)
        self._set_with_warning("output_dir", output_dir)
        self._set_file_name("expression_matrix_file", expression_matrix_file)
        self._set_file_name("tf_names_file", tf_names_file)
        self._set_file_name("meta_data_file", meta_data_file)
        self._set_file_name("priors_file", priors_file)
        self._set_file_name("gold_standard_file", gold_standard_file)
        self._set_file_name("gene_metadata_file", gene_metadata_file)

    def set_file_properties(self, extract_metadata_from_expression_matrix=None, expression_matrix_metadata=None,
                            expression_matrix_columns_are_genes=None, gene_list_index=None, metadata_handler=None):
        """
        Set properties associated with the input data files
        :param extract_metadata_from_expression_matrix: str
        :param expression_matrix_metadata: str
        :param expression_matrix_columns_are_genes: str
        :param gene_list_index: str
        :param metadata_handler: str
        """

        self._set_without_warning("extract_metadata_from_expression_matrix", extract_metadata_from_expression_matrix)
        self._set_without_warning("expression_matrix_columns_are_genes", expression_matrix_columns_are_genes)

        self._set_with_warning("expression_matrix_metadata", expression_matrix_metadata)
        self._set_with_warning("gene_list_index", gene_list_index)
        self._set_with_warning("metadata_handler", metadata_handler)

    def set_network_data_flags(self, use_no_prior=None, use_no_gold_standard=None):
        """
        Set flags to skip using existing network data. Note that these flags will be ignored if network data is
        provided
        :param use_no_prior: bool
        :param use_no_gold_standard: bool
        """

        warnings.warn("Omitting prior network data is not recommended. Use at your own risk.")

        self._set_without_warning("use_no_prior", use_no_prior)
        self._set_without_warning("use_no_gold_standard", use_no_gold_standard)

    def set_file_loading_arguments(self, file_name, **kwargs):
        """
        Update the settings for a given file name
        :param file_name: str
        """

        # Check and see if file_name is actually an object attribute holding a file name. Use that if so.
        file_name = self._get_file_name_from_attribute(file_name)
        if file_name is None:
            return

        self._file_format_settings[file_name].update(kwargs)
        self.print_file_loading_arguments(file_name)

    def print_file_loading_arguments(self, file_name):
        """
        Print the settings for a given file name
        :param file_name: str
        """

        # Check and see if file_name is actually an object attribute holding a file name. Use that if so.
        file_name = self._get_file_name_from_attribute(file_name)
        if file_name is None:
            return

        msg = "File {f} has the following settings:".format(f=file_name)
        msg += "\n\t".join([str(k) + " = " + str(v) for k, v in self._file_format_settings[file_name].items()])
        utils.Debug.vprint(msg, level=0)

    def _get_file_name_from_attribute(self, file_name):
        """
        Check and see if a file name is an object attribute that holds a file namee
        :param file_name: str
        :return file_name: str
        """
        # Check and see if file_name is actually an object attribute holding a file name. Use that if so.
        if file_name not in self._file_format_settings:
            if hasattr(self, file_name) and getattr(self, file_name) in self._file_format_settings:
                file_name = getattr(self, file_name)
            else:
                utils.Debug.vprint("File {f} is unknown".format(f=file_name), level=0)
                return None
        return file_name

    def _set_file_name(self, attr_name, file_name):
        """
        Set a file name. Create a set of default loading parameters and store them in _file_format_settings.
        Also print a warning if the file doesn't exist
        :param attr_name: str
        :param file_name: str
        """

        if file_name is None:
            return

        if file_name not in self._file_format_settings:
            self._file_format_settings[file_name] = copy.copy(DEFAULT_PANDAS_TSV_SETTINGS)

        if not os.path.isfile(self.input_path(file_name)):
            utils.Debug.vprint("File {f} does not exist".format(f=file_name))

        self._set_with_warning(attr_name, file_name)

    def _set_without_warning(self, attr_name, value):
        """
        Set an attribute name.
        :param attr_name: str
        :param value:
        """

        if value is None:
            return

        setattr(self, attr_name, value)

    def _set_with_warning(self, attr_name, value):
        """
        Set an attribute name. Warn if it's already not None
        :param attr_name: str
        :param value:
        """

        if value is None:
            return

        current_value = getattr(self, attr_name)
        if current_value is not None:
            warnings.warn("Setting {a}: replacing value {vo} with value {vn}".format(a=attr_name,
                                                                                     vo=current_value,
                                                                                     vn=value))

        setattr(self, attr_name, value)

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
        self._transpose_expression_matrix()

        # Validate that necessary input settings exist
        self.validate_data()

    def _transpose_expression_matrix(self):
        # Transpose expression data
        if self.expression_matrix_columns_are_genes:
            self.expression_matrix = utils.transpose_dataframe(self.expression_matrix)
            utils.Debug.vprint("Transposing expression matrix to {sh}".format(sh=self.expression_matrix.shape), level=2)

    def input_dataframe(self, filename, **kwargs):
        """
        Read a file in as a pandas dataframe
        """
        utils.Debug.vprint("Loading data file: {a}".format(a=self.input_path(filename)), level=2)

        # Use any kwargs for this function and any file settings from default
        if self._file_format_settings is not None and filename in self._file_format_settings:
            file_settings = self._file_format_settings[filename]
        else:
            file_settings = copy.copy(DEFAULT_PANDAS_TSV_SETTINGS)

        file_settings.update(kwargs)

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

    def read_expression(self, file=None):
        """
        Read expression matrix file into expression_matrix
        """
        file = file if file is not None else self.expression_matrix_file
        utils.Debug.vprint("Loading expression data file {file}".format(file=file), level=1)
        self.expression_matrix = self.input_dataframe(file)

        try:
            check.dataframe_is_finite(self.expression_matrix)
        except ValueError as err:
            utils.Debug.vprint("Expression Matrix " + str(err), level=0)

    def read_tfs(self, file=None):
        """
        Read tf names file into tf_names
        """

        # Load the class variable if no file is passed
        file = self.tf_names_file if file is None else file

        if file is not None:
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
            metadata_processor = MetadataHandler.get_handler(self.metadata_handler)
            self.meta_data = metadata_processor.create_default_meta_data(self.expression_matrix)

    def read_genes(self, file=None):
        """
        Read in a list of genes which should be modeled for network inference
        """

        file = file if file is not None else self.gene_metadata_file

        if file is not None:
            utils.Debug.vprint("Loading Gene metadata from file {file}".format(file=file), level=1)
            self.gene_metadata = self.input_dataframe(self.gene_metadata_file, index_col=None)

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

    def validate_data(self):
        """
        Make sure that the data that's loaded is acceptable
        """

        # Validate that the gene_metadata can be properly read, if loaded
        if self.gene_metadata is not None and self.gene_list_index not in self.gene_metadata.columns:
            raise ValueError("The gene list file must have headers and workflow.gene_list_index must be a valid column")

        # Create a null prior if the flag is set
        if self.use_no_prior and self.priors_data is not None:
            warnings.warn("The use_no_prior flag will be ignored because prior data exists")
        elif self.use_no_prior:
            utils.Debug.vprint("A null prior is has been created", level=0)
            self.priors_data = self._create_null_prior(self.expression_matrix, self.tf_names)

        # Create a null gold standard if the flag is set
        if self.use_no_gold_standard and self.gold_standard is not None:
            warnings.warn("The use_no_gold_standard flag will be ignored because gold standard data exists")
        elif self.use_no_gold_standard:
            utils.Debug.vprint("A null gold standard has been created", level=0)
            self.gold_standard = self._create_null_prior(self.expression_matrix, self.tf_names)

        # Validate that some network information exists and has been loaded
        if self.priors_data is None and self.gold_standard is None:
            raise ValueError("No gold standard or priors have been provided")

    def input_path(self, filename):
        """
        Join filename to input_dir
        """

        # Raise an error if filename is None
        if filename is None:
            raise ValueError("Cannot create a path to a filename set as None")

        # Return an absolute path unchanged
        elif os.path.isabs(filename):
            return filename

        # If input_dir is set, join file name to it and return that
        elif self.input_dir is not None:
            return self.make_path_safe(os.path.join(self.input_dir, filename))

        # If input_dir is not set, convert the filename to absolute and return it
        else:
            return self.make_path_safe(filename)

    @staticmethod
    def _create_null_prior(expression_data, tf_names):
        """
        Create a prior data matrix that is all 0s
        :param expression_data: pd.DataFrame
        :param tf_names: list
        :return priors: pd.DataFrame
        """
        return pd.DataFrame(0, index=expression_data.columns, columns=tf_names)

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
        :param remove_columns: list(str)
        :return data_frame, data_frame_two: pd.DataFrame, pd.DataFrame
        """

        data_frame_two = data_frame.loc[:, remove_columns].copy()
        data_frame = data_frame.drop(remove_columns, axis=1)
        return data_frame, data_frame_two


class WorkflowBase(WorkflowBaseLoader):
    # Flags to control splitting priors into a prior/gold-standard set
    split_gold_standard_for_crossvalidation = False
    cv_split_ratio = None
    cv_split_axis = 0
    shuffle_prior_axis = None

    # The random seed for sampling, etc
    random_seed = default.DEFAULT_RANDOM_SEED

    # The number of inference bootstraps to run
    num_bootstraps = default.DEFAULT_NUM_BOOTSTRAPS

    # Multiprocessing controller
    initialize_mp = True
    multiprocessing_controller = None

    # Prior manager
    prior_manager = ManagePriors

    # Result processing & model metrics
    result_processor_driver = ResultsProcessor
    gold_standard_filter_method = default.DEFAULT_GS_FILTER_METHOD
    metric = "aupr"

    # Output results in an InferelatorResults object
    results = None

    def __init__(self):
        super(WorkflowBase, self).__init__()
        # Get environment variables
        self.get_environmentals()

    def set_crossvalidation_parameters(self, split_gold_standard_for_crossvalidation=None, cv_split_ratio=None,
                                       cv_split_axis=None):
        """
        Set parameters for crossvalidation
        :param split_gold_standard_for_crossvalidation: bool
        :param cv_split_ratio: float
        :param cv_split_axis: int (0, 1)
        """

        self._set_without_warning("split_gold_standard_for_crossvalidation", split_gold_standard_for_crossvalidation)
        self._set_with_warning("cv_split_ratio", cv_split_ratio)
        self._set_with_warning("cv_split_axis", cv_split_axis)

        if not split_gold_standard_for_crossvalidation and (cv_split_axis is not None or cv_split_ratio is not None):
            warnings.warn("The split_gold_standard_for_crossvalidation flag is not set. Other options may be ignored")

    def set_shuffle_parameters(self, shuffle_prior_axis=None):
        """
        Set parameters for shuffling labels on a prior axis
        :param shuffle_prior_axis:
        """
        self._set_with_warning("shuffle_prior_axis", shuffle_prior_axis)

    def set_postprocessing_parameters(self, gold_standard_filter_method=None, metric=None):
        """
        Set parameters for the postprocessing engine
        :param gold_standard_filter_method:
        :param metric:
        """

        self._set_with_warning("gold_standard_filter_method", gold_standard_filter_method)
        self._set_with_warning("metric", metric)

    def set_run_parameters(self, num_bootstraps=None, random_seed=None):
        """
        Set parameters used during runtime
        """

        self._set_without_warning("num_bootstraps", num_bootstraps)
        self._set_without_warning("random_seed", random_seed)

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
        else:
            self.tf_names = self.priors_data.columns.tolist()

        # Filter priors and expression to a list of genes
        self.filter_to_gene_list()

        # Shuffle prior labels
        if self.shuffle_prior_axis is not None:
            self.priors_data = self.prior_manager.shuffle_priors(self.priors_data,
                                                                 self.shuffle_prior_axis,
                                                                 self.random_seed)

        # Check for duplicates or whatever
        self.priors_data, self.gold_standard = self.prior_manager.validate_priors_gold_standard(self.priors_data,
                                                                                                self.gold_standard)

    def filter_to_gene_list(self):
        """
        Filter the priors and expression matrix to just genes in gene_metadata
        """
        if self.gene_metadata is not None and self.gene_list_index is not None:
            gene_list = self.gene_metadata[self.gene_list_index].tolist()
            self.priors_data, self.expression_matrix = self.prior_manager.filter_to_gene_list(self.priors_data,
                                                                                              self.expression_matrix,
                                                                                              gene_list)

    def align_priors_and_expression(self):
        """
        Align prior to the expression matrix
        """
        self.priors_data = self.prior_manager.align_priors_to_expression(self.priors_data, self.expression_matrix)

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

    def create_task(self, **kwargs):
        """
        Create a task data object
        """
        raise NotImplementedError("This workflow does not support multiple tasks")


def _factory_build_inferelator(regression=RegressionWorkflow, workflow=WorkflowBase):
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
        if regression == "base":
            from inferelator.regression.base_regression import BaseRegression
            regression_class = BaseRegression
        elif regression == "bbsr":
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
    return _factory_build_inferelator(regression=regression, workflow=workflow)()
