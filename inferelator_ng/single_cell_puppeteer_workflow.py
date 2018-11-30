from __future__ import print_function

import os
import csv
import datetime

import numpy as np
import pandas as pd

from inferelator_ng import single_cell_workflow
from inferelator_ng import tfa_workflow
from inferelator_ng import results_processor
from inferelator_ng import utils

# The variable names that get set in the main workflow, but need to get copied to the puppets
SHARED_CLASS_VARIABLES = ['tf_names', 'gene_list', 'num_bootstraps', 'normalize_counts_to_one',
                          'normalize_batch_medians', 'magic_imputation', 'batch_correction_lookup',
                          'modify_activity_from_metadata', 'metadata_expression_lookup', 'gene_list_lookup']

DEFAULT_SIZE_SAMPLING = [1]
DEFAULT_GOLD_STANDARD_CUTOFF = [5]
DEFAULT_SEED_RANGE = range(42, 45)
DEFAULT_BASE_SEED = 42


class NoOutputRP(results_processor.ResultsProcessor):

    def summarize_network(self, output_dir, gold_standard, priors, threshold=0.95, output_file_name=None):
        combined_confidences = self.compute_combined_confidences()
        (recall, precision) = self.calculate_precision_recall(combined_confidences, gold_standard)
        aupr = self.calculate_aupr(recall, precision)
        interactions = (combined_confidences > threshold).sum().sum()
        if output_file_name is not None:
            self.threshold_and_summarize()
            resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
            self.save_network_to_tsv(combined_confidences, resc_betas_median, priors, output_dir=output_dir,
                                     output_file_name=output_file_name)
        return aupr, interactions


def make_puppet_workflow(workflow_type):
    class SingleCellPuppetWorkflow(single_cell_workflow.SingleCellWorkflow, workflow_type):
        """
        Standard workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and saves the model AUPR without outputting anything
        """

        network_file_path = None
        network_file_name = None

        def __init__(self, kvs, rank, expr_data, meta_data, prior_data, gs_data):
            self.kvs = kvs
            self.rank = rank
            self.expression_matrix = expr_data
            self.meta_data = meta_data
            self.priors_data = prior_data
            self.gold_standard = gs_data

        def startup_run(self):
            self.compute_activity()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):
            if self.is_master():
                results = NoOutputRP(betas, rescaled_betas)
                self.aupr, self.n_interact = results.summarize_network(self.network_file_path, gold_standard, priors,
                                                                       output_file_name=self.network_file_name)
            else:
                self.aupr, self.n_interact = None, None

    return SingleCellPuppetWorkflow


class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, tfa_workflow.TFAWorkFlow):
    seeds = DEFAULT_SEED_RANGE
    regression_type = tfa_workflow.BBSR_TFA_Workflow

    # Output TSV controllers
    write_network = True  # bool
    writer = None  # csv.csvwriter
    header = ["Seed", "AUPR", "Num_Interacting"]  # list[]
    output_file_name = "aupr.tsv"  # str

    def run(self):
        self.startup()
        self.create_writer()
        auprs = self.modeling_method()

    def compute_activity(self):
        pass

    def single_cell_normalize(self):
        pass

    def modeling_method(self):
        raise NotImplementedError("No method to create models was provided")

    def create_writer(self):
        if self.is_master():
            self.create_output_dir()
            self.writer = csv.writer(open(os.path.join(self.output_dir, self.output_file_name), mode="wb", buffering=0),
                                     delimiter="\t", quoting=csv.QUOTE_NONE)
            self.writer.writerow(self.header)

    def create_output_dir(self):
        if self.is_master():
            if self.output_dir is None:
                self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass

    def new_puppet(self, expr_data, meta_data, seed=DEFAULT_BASE_SEED, priors_data=None, gold_standard=None):
        if gold_standard is None:
            gold_standard = self.gold_standard
        if priors_data is None:
            priors_data = self.priors_data
        puppet = make_puppet_workflow(self.regression_type)(self.kvs, self.rank, expr_data, meta_data,
                                                            priors_data, gold_standard)
        self.assign_class_vars(puppet)
        puppet.random_seed = seed
        if self.write_network:
            puppet.network_file_path = self.output_dir
            puppet.network_file_name = "network_s{seed}.tsv".format(seed=seed)
        return puppet

    def assign_class_vars(self, obj):
        """
        Transfer class variables from this object to a target object
        """
        for varname in SHARED_CLASS_VARIABLES:
            try:
                setattr(obj, varname, getattr(self, varname))
            except AttributeError:
                utils.Debug.vprint("Variable {var} not assigned to parent".format(var=varname))


class SingleCellSizeSampling(SingleCellPuppeteerWorkflow):
    sizes = DEFAULT_SIZE_SAMPLING
    header = ["Size", "Num_Sampled", "Seed", "AUPR", "Num_Interacting"]

    def modeling_method(self, *args, **kwargs):
        return self.get_aupr_for_subsampled_data(*args, **kwargs)

    def get_aupr_for_subsampled_data(self, expr_data=None, meta_data=None):
        if expr_data is None:
            expr_data = self.expression_matrix
        if meta_data is None:
            meta_data = self.meta_data

        aupr_data = []
        for s_ratio in self.sizes:
            new_size = int(s_ratio * self.expression_matrix.shape[1])
            for seed in self.seeds:
                np.random.seed(seed)
                new_idx = np.random.choice(expr_data.shape[1], size=new_size)
                puppet = self.new_puppet(expr_data.iloc[:, new_idx], meta_data.iloc[new_idx, :], seed=seed)
                if self.write_network:
                    puppet.network_file_name = "network_{size}_s{seed}.tsv".format(size=s_ratio, seed=seed)
                puppet.run()
                size_aupr = (s_ratio, new_size, seed, puppet.aupr, puppet.n_interact)
                aupr_data.extend(size_aupr)
                if self.is_master():
                    self.writer.writerow(size_aupr)
        return aupr_data


class SingleCellDropoutConditionSampling(SingleCellPuppeteerWorkflow):
    drop_column = None
    header = ["Dropout", "Seed", "AUPR", "Num_Interacting"]

    def modeling_method(self, *args, **kwargs):
        return self.auprs_for_condition_dropout()

    def auprs_for_condition_dropout(self):
        aupr_data = [self.auprs_for_index("all", pd.Series(True, index=self.meta_data.index))]
        for r_name, r_idx in self.condition_dropouts().items():
            aupr_data.extend(self.auprs_for_index(r_name, r_idx))
        return aupr_data

    def condition_dropouts(self):
        condition_indexes = dict()
        for cond in self.meta_data[self.drop_column].unique().tolist():
            condition_indexes[cond] = self.meta_data[self.drop_column] == cond
        return condition_indexes

    def auprs_for_index(self, r_name, r_idx):
        local_expr_data = self.expression_matrix.loc[:, r_idx]
        local_meta_data = self.meta_data.loc[r_idx, :]

        aupr_data = []
        for seed in self.seeds:
            puppet = self.new_puppet(local_expr_data, local_meta_data, seed)
            if self.write_network:
                puppet.network_file_name = "network_{drop}_s{seed}.tsv".format(drop=r_name, seed=seed)
            puppet.run()
            drop_data = (r_name, seed, puppet.aupr, puppet.n_interact)
            aupr_data.append(drop_data)
            if self.is_master():
                self.writer.writerow(drop_data)
        return aupr_data
