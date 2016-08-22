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
        # Do nothing (all configuration is external to init)
        pass

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
        (self.design, self.response) = self.design_response_driver.run(self.expression_matrix, self.meta_data)

        # compute half_tau_response
        print('Setting up TFA specific response matrix ... ')
        self.design_response_driver.tau = self.tau / 2
        (self.design, self.half_tau_response) = self.design_response_driver.run(self.expression_matrix, self.meta_data)

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors, that are in the tf_names list
        """
        common_genes = list(set.intersection(set(self.expression_matrix.index.tolist()), set(self.priors_data.index.tolist())))
        self.priors_data = self.priors_data.loc[common_genes, self.tf_names]
        self.expression_matrix = self.expression_matrix.loc[common_genes,]

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objects for run.
        """
        col_range = range(self.response.shape[1])
        return [[np.random.choice(col_range) for x in col_range] for y in range(self.num_bootstraps)]


    def emit_results(self):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass
