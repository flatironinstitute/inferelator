from __future__ import print_function

import os
import datetime

import numpy as np

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
DEFAULT_SEED_RANGE = range(42,45)



class NoOutputRP(results_processor.ResultsProcessor):

    def summarize_network(self, output_dir, gold_standard, priors):
        combined_confidences = self.compute_combined_confidences()
        (recall, precision) = self.calculate_precision_recall(combined_confidences, gold_standard)
        return self.calculate_aupr(recall, precision)


def make_puppet_workflow(workflow_type):
    class SingleCellPuppetWorkflow(single_cell_workflow.SingleCellWorkflow, workflow_type):
        """
        Standard workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and saves the model AUPR without outputting anything
        """

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
                self.aupr = NoOutputRP(betas, rescaled_betas).summarize_network(None, gold_standard, priors)
            else:
                self.aupr = None

    return SingleCellPuppetWorkflow


class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, tfa_workflow.TFAWorkFlow):
    seeds = DEFAULT_SEED_RANGE
    regression_type = tfa_workflow.BBSR_TFA_Workflow
    header = ["Seed", "AUPR"]

    def run(self):
        self.startup()
        aupr_data = self.modeling_method()
        self.emit_results(aupr_data)

    def compute_activity(self):
        pass

    def single_cell_normalize(self):
        pass

    def modeling_method(self):
        raise NotImplementedError("No method to create models was provided")

    def emit_results(self, auprs, file_name="aupr.tsv"):

        if self.is_master():
            if self.output_dir is None:
                self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass
            with open(os.path.join(self.output_dir, file_name), mode="w") as out_fh:
                print("\t".join(self.header), file=out_fh)
                for tup in auprs:
                    print("\t".join(map(str, tup)), file=out_fh)

    def get_aupr_for_seeds(self, expr_data, meta_data, regression_type, priors_data=None, gold_standard=None):
        if gold_standard is None:
            gold_standard = self.gold_standard
        if priors_data is None:
            priors_data = self.priors_data

        aupr_data = []
        for seed in self.seeds:
            puppet = make_puppet_workflow(regression_type)(self.kvs, self.rank, expr_data, meta_data,
                                                           priors_data, gold_standard)
            self.assign_class_vars(puppet)
            puppet.random_seed = seed
            puppet.run()
            aupr_data.append((seed, puppet.aupr))
        return aupr_data

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
    header = ["Size", "Seed", "AUPR"]

    def modeling_method(self, *args, **kwargs):
        return self.get_aupr_for_resized_data(*args, **kwargs)

    def get_aupr_for_resized_data(self, expr_data=None, meta_data=None, regression_type=None):
        if expr_data is None:
            expr_data = self.expression_matrix
        if meta_data is None:
            meta_data = self.meta_data
        if regression_type is None:
            regression_type = self.regression_type

        aupr_data = []
        for s_ratio in self.sizes:
            new_size = int(s_ratio * self.expression_matrix.shape[1])
            new_idx = np.random.choice(expr_data.shape[1], size=new_size)
            auprs = self.get_aupr_for_seeds(expr_data.iloc[:, new_idx],
                                            meta_data.iloc[new_idx, :],
                                            regression_type=regression_type)
            aupr_data.extend([(new_size, se, au) for (se, au) in auprs])
        return aupr_data


class SingleCellDropoutConditionSampling(SingleCellPuppeteerWorkflow):
    drop_column = None

    def modeling_method(self, *args, **kwargs):
        return self.auprs_for_condition_dropout(*args, **kwargs)

    def auprs_for_condition_dropout(self):
        aupr_data = []
        idx = self.condition_dropouts()
        for r_name, r_idx in idx.items():
            auprs = self.auprs_for_index(r_idx)
            aupr_data.extend([(r_name, se, au) for (se, au) in auprs])
        return aupr_data

    def condition_dropouts(self):
        condition_indexes = dict()
        for cond in self.meta_data[self.drop_column].unique().tolist():
            condition_indexes[cond] = self.meta_data[self.drop_column] == cond
        return condition_indexes

    def auprs_for_index(self, idx):
        local_expr_data = self.expression_matrix.iloc[:, idx]
        local_meta_data = self.meta_data.iloc[idx, :]
        return self.get_aupr_for_seeds(local_expr_data, local_meta_data, self.regression_type)