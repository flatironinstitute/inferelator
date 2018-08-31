"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.
"""

from kvsstcp import KVSClient
from . import utils
import numpy as np
import os
import pandas as pd


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

    def __init__(self):
        # Connect to KVS and get environment variables
        self.kvs = KVSClient()
        for k, v in utils.slurm_envs().items():
            setattr(self, k, v)

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

    def read_expression(self):
        self.expression_matrix = self.input_dataframe(self.expression_matrix_file)
        if self.expression_matrix_transpose:
            self.expression_matrix = self.expression_matrix.T

    def read_tfs(self):
        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = utils.read_tf_names(tf_file)

    def read_metadata(self):
        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False, strict=False)
        if self.meta_data is None:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)

    def set_gold_standard_and_priors(self):
        self.priors_data = self.input_dataframe(self.priors_file)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

    def input_path(self, filename):
        return os.path.abspath(os.path.join(self.input_dir, filename))

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

    def input_file(self, filename, strict=True):
        path = self.input_path(filename)
        if os.path.exists(path):
            return open(path)
        elif not strict:
            return None
        raise ValueError("no such file " + repr(path))

    def input_dataframe(self, filename, strict=True, has_index=True):
        f = self.input_file(filename, strict)
        if f is not None:
            return utils.df_from_tsv(f, has_index)
        else:
            assert not strict
            return None

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
