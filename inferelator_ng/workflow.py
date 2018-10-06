"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.
"""

from inferelator_ng import utils
import numpy as np
import os
import pandas as pd

PD_INPUT_SETTINGS = dict(sep="\t", header=0)

class WorkflowBase(object):
    # Common configuration parameters
    input_dir = None
    expression_matrix_file = "expression.tsv"
    tf_names_file = "tf_names.tsv"
    meta_data_file = "meta_data.tsv"
    priors_file = "gold_standard.tsv"
    gold_standard_file = "gold_standard.tsv"
    output_dir = None
    random_seed = 42
    expression_matrix_transpose = False

    # Computed data structures
    expression_matrix = None  # expression_matrix dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe

    # Hold the KVS information
    rank = 0
    kvs = None
    tasks = None

    # Startup configuration parameters. Set async_start to true to enable a staggered startup. startup_run will be
    # executed asynchronously and then startup_finish will be run together.
    async_start = False
    async_chunk = 2

    def __init__(self):
        # Connect to KVS and get environment variables
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
        if self.async_start:
            self.kvs.async_settings(chunk=self.async_chunk).execute_async(self.startup_run)
        else:
            self.startup_run()
        self.startup_finish()

    def startup_run(self):
        """
        Execute any data preprocessing necessary before regression. This should include only steps that can be
        run asynchronously by each separate client
        """
        raise NotImplementedError  # implement in subclass

    def startup_finish(self):
        """
        Execute any data preprocessing necessary before regression. This should include steps that need to be run
        synchronously. It will be executed after startup_run.
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

    def read_expression(self, file=None, transpose=None):
        """
        Read expression matrix file into expression_matrix
        """
        if file is None:
            file = self.expression_matrix_file
        self.expression_matrix = self.input_dataframe(file)

        if transpose is None:
            transpose = self.expression_matrix_transpose
        if transpose:
            self.expression_matrix = self.expression_matrix.T

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
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

    def input_path(self, filename):
        """
        Join filename to input_dir
        """
        return os.path.abspath(os.path.join(self.input_dir, filename))

    def input_dataframe(self, filename, index_col=0):
        """
        Read a file in as a pandas dataframe
        """
        return pd.read_table(self.input_path(filename), index_col=index_col, **PD_INPUT_SETTINGS)

    def append_to_path(self, var_name, to_append):
        """
        Add a string to an existing path variable in class
        """
        path = getattr(self, var_name, None)
        if path is None:
            raise ValueError("Cannot append to None")
        setattr(self, var_name, os.path.join(path, to_append))

    def create_default_meta_data(self, expression_matrix):
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
        exp_genes = self.expression_matrix.index.tolist()
        all_regs_with_data = list(
            set.union(set(self.expression_matrix.index.tolist()), set(self.priors_data.columns.tolist())))
        tf_names = list(set.intersection(set(self.tf_names), set(all_regs_with_data)))
        self.priors_data = self.priors_data.loc[exp_genes, tf_names]
        self.priors_data = pd.DataFrame.fillna(self.priors_data, 0)

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(self.response.shape[1])
        return np.random.choice(col_range, size=(self.num_bootstraps, self.response.shape[1])).tolist()

    def emit_results(self):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass

    def is_master(self):

        if self.rank == 0:
            return True
        else:
            return False
