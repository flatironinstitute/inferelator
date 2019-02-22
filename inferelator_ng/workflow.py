"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.
"""

from inferelator_ng import utils
from inferelator_ng.utils import Validator as check
from inferelator_ng import default
from inferelator_ng.prior_gs_split_workflow import split_for_cv, remove_prior_circularity
import numpy as np
import os
import datetime
import pandas as pd

import gzip
import bz2


class WorkflowBase(object):
    # Common configuration parameters
    input_dir = None
    file_format_settings = default.DEFAULT_PD_INPUT_SETTINGS
    file_format_overrides = dict()
    expression_matrix_file = default.DEFAULT_EXPRESSION_FILE
    tf_names_file = default.DEFAULT_TFNAMES_FILE
    meta_data_file = default.DEFAULT_METADATA_FILE
    priors_file = default.DEFAULT_PRIORS_FILE
    gold_standard_file = default.DEFAULT_GOLDSTANDARD_FILE
    output_dir = None
    random_seed = default.DEFAULT_RANDOM_SEED
    num_bootstraps = default.DEFAULT_NUM_BOOTSTRAPS

    # Flags to control splitting priors into a prior/gold-standard set
    split_priors_for_gold_standard = False
    split_gold_standard_for_crossvalidation = False
    cv_split_ratio = default.DEFAULT_GS_SPLIT_RATIO
    cv_split_axis = default.DEFAULT_GS_SPLIT_AXIS

    # Computed data structures [G: Genes, K: Predictors, N: Conditions
    expression_matrix = None  # expression_matrix dataframe [G x N]
    tf_names = None  # tf_names list [k,]
    meta_data = None  # meta data dataframe [G x ?]
    priors_data = None  # priors data dataframe [G x K]
    gold_standard = None  # gold standard dataframe [G x K]

    # Hold the KVS information
    rank = 0
    kvs = None
    tasks = None

    def __init__(self, initialize_mp=True):
        # Connect to KVS and get environment variables
        if initialize_mp:
            self.initialize_multiprocessing()
        self.get_environmentals()

    def initialize_multiprocessing(self):
        """
        Override this if you want to use something besides KVS for multiprocessing.
        """
        from inferelator_ng.kvs_controller import KVSController
        self.kvs = KVSController()

    def get_environmentals(self):
        """
        Load environmental variables into class variables
        """
        for k, v in utils.slurm_envs().items():
            setattr(self, k, v)

    def startup(self):
        """
        Startup by preprocessing all data into a ready format for regression.
        """
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
        self.set_gold_standard_and_priors()

    def read_expression(self, file=None):
        """
        Read expression matrix file into expression_matrix
        """
        if file is None:
            file = self.expression_matrix_file
        self.expression_matrix = self.input_dataframe(file)

    def read_tfs(self, file=None):
        """
        Read tf names file into tf_names
        """
        if file is None:
            file = self.tf_names_file

        tfs = self.input_dataframe(file, index_col=None)
        assert tfs.shape[1] == 1
        self.tf_names = tfs.values.flatten().tolist()

    def read_metadata(self, file=None):
        """
        Read metadata file into meta_data or make fake metadata
        """
        if file is None:
            file = self.meta_data_file

        try:
            self.meta_data = self.input_dataframe(file, index_col=None)
        except IOError:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)

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

        try:
            check.index_values_unique(self.priors_data.index)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate gene(s) in prior index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

        try:
            check.index_values_unique(self.priors_data.columns)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate tf(s) in prior index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

    def split_priors_into_gold_standard(self):
        """
        Break priors_data in half and give half to the gold standard
        """

        if self.gold_standard is not None:
            utils.Debug.vprint("Existing gold standard is being replaced by a split from the prior", level=0)
        self.priors_data, self.gold_standard = split_for_cv(self.priors_data,
                                                            self.cv_split_ratio,
                                                            split_axis=self.cv_split_axis,
                                                            seed=self.random_seed)

        utils.Debug.vprint("Prior split into a prior {pr} and a gold standard {gs}".format(pr=self.priors_data.shape,
                                                                                           gs=self.gold_standard.shape),
                           level=0)

    def cross_validate_gold_standard(self):
        """
        Sample the gold standard for crossvalidation, and then remove the new gold standard from the priors
        """

        utils.Debug.vprint("Resampling prior {pr} and gold standard {gs}".format(pr=self.priors_data.shape,
                                                                                 gs=self.gold_standard.shape), level=0)
        _, self.gold_standard = split_for_cv(self.gold_standard,
                                             self.cv_split_ratio,
                                             split_axis=self.cv_split_axis,
                                             seed=self.random_seed)
        self.priors_data, self.gold_standard = remove_prior_circularity(self.priors_data, self.gold_standard,
                                                                        split_axis=self.cv_split_axis)
        utils.Debug.vprint("Selected prior {pr} and gold standard {gs}".format(pr=self.priors_data.shape,
                                                                               gs=self.gold_standard.shape), level=0)

    def input_path(self, filename, mode='r'):
        """
        Join filename to input_dir
        """

        if filename.endswith(".gz"):
            opener = gzip.open
        elif filename.endswith(".bz2"):
            opener = bz2.BZ2File
        else:
            opener = open

        return opener(os.path.abspath(os.path.join(self.input_dir, filename)), mode=mode)

    def input_dataframe(self, filename, index_col=0):
        """
        Read a file in as a pandas dataframe
        """

        file_settings = self.file_format_settings.copy()
        if filename in self.file_format_overrides:
            file_settings.update(self.file_format_overrides[filename])

        with self.input_path(filename) as fh:
            return pd.read_table(fh, index_col=index_col, **file_settings)

    def append_to_path(self, var_name, to_append):
        """
        Add a string to an existing path variable in class
        """
        path = getattr(self, var_name, None)
        if path is None:
            raise ValueError("Cannot append to None")
        setattr(self, var_name, os.path.join(path, to_append))

    @staticmethod
    def create_default_meta_data(expression_matrix):
        """
        Create a meta_data dataframe from basic defaults
        """
        metadata_rows = expression_matrix.columns.tolist()
        metadata_defaults = {"isTs": "FALSE", "is1stLast": "e", "prevCol": "NA", "del.t": "NA", "condName": None}
        data = {}
        for key in metadata_defaults.keys():
            data[key] = pd.Series(data=[metadata_defaults[key] if metadata_defaults[key] else i for i in metadata_rows])
        return pd.DataFrame(data)

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors, that are in the tf_names list
        """
        expressed_targets = self.expression_matrix.index
        expressed_or_prior = expressed_targets.union(self.priors_data.columns)
        keeper_regulators = expressed_or_prior.intersection(self.tf_names)

        if len(keeper_regulators) == 0 or len(expressed_targets) == 0:
            raise ValueError("Filtering will result in a priors with at least one axis of 0 length")

        self.priors_data = self.priors_data.loc[expressed_targets, keeper_regulators]
        self.priors_data = pd.DataFrame.fillna(self.priors_data, 0)

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(self.response.shape[1])
        random_state = np.random.RandomState(seed=self.random_seed)
        return random_state.choice(col_range, size=(self.num_bootstraps, self.response.shape[1])).tolist()

    def emit_results(self):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass

    def is_master(self):
        """
        Return True if this is the rank-0 (master) thread
        """

        if self.rank == 0:
            return True
        else:
            return False

    def create_output_dir(self):
        """
        Set a default output_dir if nothing is set. Create the path if it doesn't exist.
        """
        if self.output_dir is None:
            self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass
