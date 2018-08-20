import os
import datetime
from . import utils
import numpy as np
import pandas as pd
from kvsstcp.kvsclient import KVSClient

"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.
"""

# Get the following environment variables and put them into the workflow object
# Workflow_variable_name, casting function, default (if the env isn't set or the casting fails for whatever reason)
SBATCH_VARS = {'RUNDIR': ('output_dir', str, None),
               'DATADIR': ('input_dir', str, None),
               'SLURM_PROCID': ('rank', int, 0),
               'SLURM_NTASKS_PER_NODE': ('cores', int, 10)
               }


class WorkflowBase(object):
    # File paths
    input_dir = None
    output_dir = None
    expression_matrix_file = "expression.tsv"
    tf_names_file = "tf_names.tsv"
    meta_data_file = "meta_data.tsv"
    priors_file = "gold_standard.tsv"
    gold_standard_file = "gold_standard.tsv"

    # Required configuration parameters
    random_seed = 42
    cores = 10
    rank = 0

    # Computed data structures
    expression_matrix = None  # expression_matrix dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe

    # Connect to KVS
    kvs = KVSClient()

    def __init__(self):
        self.get_sbatch_variables()

    def get_sbatch_variables(self):
        """
        Get environment variables and set them as class variables
        """
        for os_var, (cv, mt, de) in SBATCH_VARS.items():
            try:
                val = mt(os.environ[os_var])
                utils.Debug.vprint("Setting {var} to {val}".format(var=cv, val=val), level=2)
            except (KeyError, TypeError):
                val = de
            setattr(self, cv, val)

    def append_to_path(self, var_name, to_append):
        """
        Add a string to an existing path variable in class
        """
        path = getattr(self, var_name, None)
        if path is None:
            raise ValueError("Cannot append to None")
        setattr(self, var_name, os.path.join(path, to_append))

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        raise NotImplementedError  # implement in subclass

    def preprocess_data(self):
        """
        Read data files in to data structures.
        """
        np.random.seed(self.random_seed)

        self.expression_matrix = self.input_dataframe(self.expression_matrix_file)
        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = utils.read_tf_names(tf_file)

        # Read metadata, creating a default non-time series metadata file if none is provided
        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False, strict=False)
        if self.meta_data is None:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)
        self.priors_data = self.input_dataframe(self.priors_file)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

        self.filter_expression_and_priors()

    def is_master(self):
        if self.rank == 0:
            return True
        else:
            return False

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

    def input_path(self, filename):
        if self.input_dir is None:
            return os.path.abspath(os.path.join('.', filename))
        else:
            return os.path.abspath(os.path.join(self.input_dir, filename))

    def validate_output_path(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass

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

        utils.Debug.vprint("Filter_expression_and_priors complete, priors data {}".format(self.priors_data.shape))

    def get_bootstraps(self, size, num_bootstraps):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(size)
        return [[np.random.choice(col_range) for x in col_range] for y in range(num_bootstraps)]

    def emit_results(self, *args, **kwargs):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass
