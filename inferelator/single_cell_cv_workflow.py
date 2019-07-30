from __future__ import print_function

import numpy as np
import pandas as pd

from inferelator import single_cell_workflow
from inferelator import crossvalidation_workflow
from inferelator import utils
from inferelator import default
from inferelator.utils import Validator as check

from inferelator.distributed.inferelator_mp import MPControl

class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, crossvalidation_workflow.PuppeteerWorkflow):
    seeds = default.DEFAULT_SEED_RANGE

    # Output TSV controllers
    write_network = True  # bool
    csv_writer = None  # csv.csvwriter
    csv_header = ("Seed", "AUPR", "Num_Interacting")  # list[]
    output_file_name = "aupr.tsv"  # str

    # How to sample
    stratified_sampling = False
    stratified_batch_lookup = default.DEFAULT_METADATA_FOR_BATCH_CORRECTION
    sample_with_replacement = True

    cv_workflow_type = single_cell_workflow.SingleCellWorkflow
    cv_regression_type = "bbsr"

    def run(self):
        np.random.seed(self.random_seed)
        self.startup()
        self.create_writer()
        auprs = self.modeling_method()

    def startup_finish(self):
        # Do all the processing stuff in the puppet workflow
        pass

    def process_priors_and_gold_standard(self):
        # Do all the processing stuff in the puppet workflow
        pass

    def modeling_method(self):
        raise NotImplementedError("No method to create models was provided")

    def get_sample_index(self, meta_data=None, sample_ratio=None, sample_size=None,
                         min_size=default.DEFAULT_MINIMUM_SAMPLE_SIZE, stratified_sampling=None):
        """
        Produce an integer index to sample data using .iloc. If the self.stratified_sampling flag is True, sample
        separately from each group, as defined by the self.stratified_batch_lookup column.
        :param meta_data: pd.DataFrame [N x ?]
            Data frame to sample from. Use self.meta_data if this is not set.
        :param sample_ratio: float
            Sample expression_matrix to this proportion of data points
        :param sample_size: int
            Sample expression matrix to this absolute number of data points. If sampling from each stratified group,
            this is the absolute number of data points PER GROUP (not total)
        :return new_size, new_idx: int, np.ndarray
            Return the total number of
        """

        # Sanity check inputs
        assert check.arguments_not_none((sample_ratio, sample_size), num_none=1)
        assert check.argument_numeric(sample_ratio, low=0, allow_none=True)
        assert check.argument_numeric(sample_size, low=0, allow_none=True)

        stratified_sampling = stratified_sampling if stratified_sampling is not None else self.stratified_sampling

        if stratified_sampling:
            # Use the main meta_data if there's nothing given
            meta_data = meta_data if meta_data is not None else self.meta_data

            # Copy and reindex the meta_data so that the index can be used with iloc
            meta_data = meta_data.copy()
            meta_data.index = pd.Index(range(meta_data.shape[0]))
            new_idx = np.ndarray(0, dtype=int)

            # For each factor in the batch column
            for batch in meta_data[self.stratified_batch_lookup].unique().tolist():
                # Get the integer index of the data points in this batch
                batch_idx = meta_data.loc[meta_data[self.stratified_batch_lookup] == batch, :].index.tolist()

                # Decide how many to collect from this batch
                size = sample_size if sample_ratio is None else max(int(len(batch_idx) * sample_ratio), min_size)

                # Resample and append the new sample index to the index array
                new_idx = np.append(new_idx, np.random.choice(batch_idx, size=size,
                                                              replace=self.sample_with_replacement))
            return new_idx
        else:
            # Decide how many to collect from the total expression matrix or the meta_data
            num_samples = self.expression_matrix.shape[1] if meta_data is None else meta_data.shape[0]
            size = sample_size if sample_ratio is None else max(int(sample_ratio * num_samples), min_size)
            return np.random.choice(num_samples, size=size, replace=self.sample_with_replacement)


class SingleCellSizeSampling(SingleCellPuppeteerWorkflow):
    sizes = default.DEFAULT_SIZE_SAMPLING
    csv_header = ["Size", "Num_Sampled", "Seed", "AUPR", "Num_Confident_Int", "Num_Precision_Int"]

    def modeling_method(self, *args, **kwargs):
        return self.get_aupr_for_subsampled_data()

    def get_aupr_for_subsampled_data(self):
        aupr_data = []
        for s_ratio in self.sizes:
            for seed in self.seeds:

                np.random.seed(seed)
                nidx = self.get_sample_index(sample_ratio=s_ratio)
                puppet = self.new_puppet(self.expression_matrix.iloc[:, nidx], self.meta_data.iloc[nidx, :], seed=seed)
                if self.write_network:
                    puppet.network_file_name = "network_{size}_s{seed}.tsv".format(size=s_ratio, seed=seed)
                    puppet.pr_curve_file_name = "pr_curve_{size}_s{seed}.pdf".format(size=s_ratio, seed=seed)
                puppet.run()
                size_aupr = (s_ratio, len(nidx), seed, puppet.aupr, puppet.n_interact, puppet.precision_interact)
                aupr_data.append(size_aupr)
                if self.is_master():
                    self.csv_writer.writerow(size_aupr)
                MPControl.sync_processes("post_CV")
        return aupr_data


class SingleCellDropoutConditionSampling(SingleCellPuppeteerWorkflow):
    csv_header = ["Dropout", "Seed", "AUPR", "Num_Confident_Int", "Num_Precision_Int"]

    # Sampling batches
    sample_batches_to_size = default.DEFAULT_BATCH_SIZE
    stratified_sampling = True
    drop_column = None

    model_dropouts = True
    model_dropins = True

    def modeling_method(self, *args, **kwargs):

        self.factor_indexes = self.factor_singles()
        auprs = []
        if self.model_dropins:
            auprs.extend(self.auprs_for_condition_dropin())
        if self.model_dropouts:
            auprs.extend(self.auprs_for_condition_dropout())
        return auprs

    def auprs_for_condition_dropout(self):
        """
        Run modeling on all data, and then on data where each factor from `drop_column` has been removed
        :return:
        """
        # Run the modeling on all data
        aupr_data = self.auprs_for_index("all_dropout", pd.Series(True, index=self.meta_data.index))

        if self.drop_column is None:
            return aupr_data

        # For all of the factors in `drop_column`, iterate through and remove them one by one, modeling on the rest
        for r_name, r_idx in self.factor_indexes.items():
            aupr_data.extend(self.auprs_for_index(r_name, r_idx))
        return aupr_data

    def auprs_for_condition_dropin(self):
        """
        Run modeling on all data, and then on data where each factor from `drop_column` is sampled separately
        :return:
        """
        # Run the modeling on all data with resizing
        drop_in_sizing = int(self.sample_batches_to_size / len(self.factor_indexes))
        aupr_data = self.auprs_for_index("all_dropin", pd.Series(True, index=self.meta_data.index),
                                         sample_size=drop_in_sizing)

        if self.drop_column is None:
            return aupr_data

        # For all of the factors in `drop_column`, iterate through them and model each separately
        for r_name, r_idx in self.factor_indexes.items():
            aupr_data.extend(self.auprs_for_index(r_name + "_only", r_idx))
        return aupr_data

    def factor_dropouts(self):
        """
        Create a dict of boolean Series, keyed by factor name, with an index boolean pd.Series that removes that factor

        :return factor_indexes: dict{pd.Series}
        """
        factor_indexes = dict()
        for fact in self.meta_data[self.drop_column].unique().tolist():
            factor_indexes[fact] = self.meta_data[self.drop_column] != fact
        return factor_indexes

    def factor_singles(self):
        """
        Create a dict of boolean Series, keyed by factor name, with an index boolean pd.Series that keeps that factor

        :return factor_indexes: dict{pd.Series}
        """
        factor_indexes = dict()
        for fact in self.meta_data[self.drop_column].unique().tolist():
            factor_indexes[fact] = self.meta_data[self.drop_column] == fact
        return factor_indexes

    def auprs_for_index(self, r_name, r_idx, sample_size=None):

        sample_size = sample_size if sample_size is not None else self.sample_batches_to_size

        local_expr_data = self.expression_matrix.loc[:, r_idx]
        local_meta_data = self.meta_data.loc[r_idx, :]

        aupr_data = []
        for seed in self.seeds:
            np.random.seed(seed)
            idx = self.get_sample_index(meta_data=local_meta_data, sample_size=sample_size)
            puppet = self.new_puppet(local_expr_data.iloc[:, idx], local_meta_data.iloc[idx, :], seed)
            if self.write_network:
                puppet.network_file_name = "network_drop_{drop}_s{seed}.tsv".format(drop=r_name, seed=seed)
                puppet.pr_curve_file_name = "pr_curve_{drop}_s{seed}.pdf".format(drop=r_name, seed=seed)
            puppet.run()
            drop_data = (r_name, seed, puppet.aupr, puppet.n_interact, puppet.precision_interact)
            aupr_data.append(drop_data)
            if self.is_master():
                self.csv_writer.writerow(drop_data)
            MPControl.sync_processes("post_CV")
        return aupr_data
