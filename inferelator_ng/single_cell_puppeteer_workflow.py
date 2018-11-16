import os
import datetime

import numpy as np

from inferelator_ng import single_cell_workflow
from inferelator_ng import tfa_workflow
from inferelator_ng import workflow
from inferelator_ng import results_processor


class NoOutputRP(results_processor.ResultsProcessor):

    def summarize_network(self, output_dir, gold_standard, priors):
        combined_confidences = self.compute_combined_confidences()
        (recall, precision) = self.calculate_precision_recall(combined_confidences, gold_standard)
        return self.calculate_aupr(recall, precision)


def make_puppet_workflow(workflow_type):
    class SingleCellPuppetWorkflow(single_cell_workflow.SingleCellWorkflow, workflow_type):
        """
        Standard BBSR TFA Workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and saves the model AUPR without outputting anything
        """

        def __init__(self, kvs, rank, expr_data, meta_data, prior_data, gs_data, tf_names, size=None):
            self.kvs = kvs
            self.rank = rank
            self.expression_matrix = expr_data
            self.meta_data = meta_data
            self.priors_data = prior_data
            self.gold_standard = gs_data
            self.tf_names = tf_names
            self.size = size

        def startup_run(self):
            self.compute_activity()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):
            if self.is_master():
                self.aupr = NoOutputRP(betas, rescaled_betas).summarize_network(None, gold_standard, priors)
            else:
                self.aupr = None

        def get_bootstraps(self):
            if self.size is not None:
                return np.random.choice(self.response.shape[1], size=(self.num_bootstraps, self.size)).tolist()
            else:
                return workflow.WorkflowBase.get_bootstraps(self)

    return SingleCellPuppetWorkflow


class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, tfa_workflow.TFAWorkFlow):
    seeds = range(42, 45)
    sizes = [1]
    drop_column = None
    regression_type = tfa_workflow.BBSR_TFA_Workflow

    def compute_activity(self):
        pass

    def single_cell_normalize(self):
        pass

    def run(self):
        self.startup()

        aupr_data = dict()
        if self.drop_column is not None:
            idx = self.condition_dropouts()
            for r_name, r_idx in idx.items():
                aupr_data[r_name] = self.auprs_for_index(r_idx)

        aupr_data["ALL"] = self.auprs(self.expression_matrix, self.meta_data, self.regression_type)
        self.emit_results(aupr_data)

    def auprs_for_index(self, idx):
        local_expr_data = self.expression_matrix.iloc[:, idx]
        local_meta_data = self.meta_data.iloc[idx, :]
        return self.auprs(local_expr_data, local_meta_data, self.regression_type)

    def emit_results(self, auprs, file_name="aupr.tsv"):

        if self.is_master():
            if self.output_dir is None:
                self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass
            with open(os.path.join(self.output_dir, file_name), mode="w") as out_fh:
                for conds, size_seed in auprs:
                    for line in size_seed:
                        print(conds + "\t" + "\t".join(line), file=out_fh)

    def auprs(self, expr_data, meta_data, regression_type):
        aupr_data = list()
        for s_ratio in self.sizes:
            new_size = self.calc_size(s_ratio)
            for seed in self.seeds:
                puppet = make_puppet_workflow(regression_type)(self.kvs, self.rank, expr_data, meta_data,
                                                               self.priors_data, self.gold_standard, self.tf_names)
                puppet.random_seed = seed
                puppet.run()
                aupr_data.append((new_size, seed, puppet.aupr))
        return aupr_data

    def condition_dropouts(self):
        condition_indexes = dict()
        for cond in self.meta_data[self.drop_column].unique().tolist():
            condition_indexes[cond] = self.meta_data[self.drop_column] == cond
        return condition_indexes

    def calc_size(self, size):
        return int(size * self.expression_matrix.shape[1])
