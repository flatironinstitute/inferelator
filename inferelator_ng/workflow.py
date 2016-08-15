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

class WorkflowBase(object):

    # Common configuration parameters
    input_dir = None
    exp_mat_file = "expression.tsv"
    tf_names_file = "tf_names.tsv"
    meta_data_file = "meta_data.tsv"
    priors_file = "gold_standard.tsv"
    gold_standard_file = "gold_standard.tsv"
    random_seed = 42

    # Computed data structures
    exp_mat = None  # exp_mat dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe

    def __init__(self):
        # Do nothing (all configuration is external to init)
        pass

    def run(self):
        "Execute workflow, after all configuration."
        self.get_data()
        np.random.seed(self.random_seed)
        self.compute_common_data()
        priors = list(self.get_priors())
        bootstraps = list(self.get_bootstraps())
        for prior in priors:
            results = [prior.run_bootstrap(self, bootstrap_parameters)
                       for bootstrap_parameters in bootstraps]
            prior.combine_bootstrap_results(self, results)
        self.emit_results(priors)

    def get_data(self):
        """
        Read data files in to data structures.
        """
        self.exp_mat = self.input_dataframe(self.exp_mat_file)
        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = utils.read_tf_names(tf_file)
        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False)
        self.priors_data = self.input_dataframe(self.priors_file)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

    def input_path(self, filename):
        return os.path.abspath(os.path.join(self.input_dir, filename))

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

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        raise NotImplementedError  # implement in subclass

    def get_priors(self):
        """
        Generate sequence of priors objects for run.
        """
        raise NotImplementedError  # implement in subclass

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objectss for run.
        """
        raise NotImplementedError  # implement in subclass

    def emit_results(self, priors):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass
