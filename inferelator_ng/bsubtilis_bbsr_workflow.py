"""
Run BSubtilis Inference. 
"""

from . import utils
import numpy as np
import os
from workflow import WorkflowBase

class Bsubtilis_Bbsr_Workflow(WorkflowBase):

    # Common configuration parameters
    input_dir = 'bsubtilis'
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

    def input_path(self, filename):
        return os.path.abspath(os.path.join(self.input_dir, filename))

    def input_file(self, filename, strict=True):
        path = self.input_path(filename)
        if os.path.exists(path):
            return open(path)
        elif not strict:
            return None
        raise ValueError("no such file " + repr(path))

    def input_dataframe(self, filename, strict=True):
        f = self.input_file(filename, strict)
        if f is not None:
            return utils.df_from_tsv(f)
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
