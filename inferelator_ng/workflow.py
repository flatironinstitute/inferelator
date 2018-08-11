"""
Base implementation for high level workflow.

The goal of this design is to make it easy to share
code among different variants of the Inferelator workflow.
"""

"""
Add doc string here.
"""

from . import utils
import numpy as np
import os
import random
import pandas as pd

SBATCH_VARS = {'RUNDIR': 'output_dir', 'DATADIR': 'input_dir', 'SLURM_PROC_ID': 'rank'}
SBATCH_VAR_TYPE = {'RUNDIR': str, 'DATADIR': str, 'SLURM_PROC_ID': int}
SBATCH_DEFAULTS = {'RUNDIR': None, 'DATADIR': None, 'SLURM_PROC_ID': 0}

class WorkflowBase(object):

    # Common configuration parameters
    input_dir = None
    expression_matrix_file = "expression.tsv"
    tf_names_file = "tf_names.tsv"
    meta_data_file = "meta_data.tsv"
    priors_file = "gold_standard.tsv"
    gold_standard_file = "gold_standard.tsv"
    random_seed = 42

    # Computed data structures
    expression_matrix = None  # expression_matrix dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe

    def __init__(self):
        self.get_sbatch_variables()

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        raise NotImplementedError  # implement in subclass

    def get_data(self):
        """
        Read data files in to data structures.
        """
        self.expression_matrix = self.input_dataframe(self.expression_matrix_file)
        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = utils.read_tf_names(tf_file)
        
        # Read metadata, creating a default non-time series metadata file if none is provided
        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False, strict=False)
        if self.meta_data is None:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)
        self.set_gold_standard_and_priors()

    def set_gold_standard_and_priors(self):
        self.priors_data = self.input_dataframe(self.priors_file)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

    def input_path(self, filename):
        if self.input_dir is None:
            return os.path.abspath(os.path.join('.', filename))
        else:
            return os.path.abspath(os.path.join(self.input_dir, filename))

    def create_default_meta_data(self, expression_matrix):
        metadata_rows = expression_matrix.columns.tolist()
        metadata_defaults = {"isTs":"FALSE", "is1stLast":"e", "prevCol":"NA", "del.t":"NA", "condName":None}
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

    def input_dataframe(self, filename, strict=True, has_index =True):
        f = self.input_file(filename, strict)
        if f is not None:
            return utils.df_from_tsv(f, has_index)
        else:
            assert not strict
            return None

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.filter_expression_and_priors()
        print('Creating design and response matrix ... ')
        self.design_response_driver.delTmin = self.delTmin
        self.design_response_driver.delTmax = self.delTmax
        self.design_response_driver.tau = self.tau
        self.design_response_driver.return_half_tau = True
        (self.design, self.response, self.half_tau_response) = self.design_response_driver.run(self.expression_matrix,
                                                                                               self.meta_data)

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors, that are in the tf_names list
        """
        exp_genes = self.expression_matrix.index.tolist()
        all_regs_with_data = list(set.union(set(self.expression_matrix.index.tolist()), set(self.priors_data.columns.tolist())))
        tf_names = list(set.intersection(set(self.tf_names), set(all_regs_with_data)))
        self.priors_data = self.priors_data.loc[exp_genes, tf_names]
        self.priors_data = pd.DataFrame.fillna(self.priors_data, 0)

        utils.Debug.vprint("Filter_expression_and_priors complete, priors data {}".format(self.priors_data.shape))

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(self.response.shape[1])
        return [[np.random.choice(col_range) for x in col_range] for y in range(self.num_bootstraps)]


    def get_sbatch_variables(self):
        import pprint
        pprint.PrettyPrinter().pprint(os.environ)
        for os_var in SBATCH_VARS:
            try:
                val = SBATCH_VAR_TYPE[os_var](os.environ[os_var])
                utils.Debug.vprint("Setting {var} to {val}".format(var=SBATCH_VARS[os_var], val=val), level=0)
            except KeyError:
                val = SBATCH_DEFAULTS[os_var]
            setattr(self, SBATCH_VARS[os_var], val)


    def emit_results(self, *args, **kwargs):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass
