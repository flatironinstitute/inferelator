from __future__ import print_function

import os
import csv

import numpy as np
import pandas as pd

from inferelator_ng import single_cell_workflow
from inferelator_ng import results_processor
from inferelator_ng import utils
from inferelator_ng import default
from inferelator_ng import bbsr_python
from inferelator_ng.utils import Validator as check

# The variable names that get set in the main workflow, but need to get copied to the puppets
SHARED_CLASS_VARIABLES = ['tf_names', 'gene_list', 'num_bootstraps', 'modify_activity_from_metadata',
                          'metadata_expression_lookup', 'gene_list_lookup', 'mi_sync_path', 'count_minimum',
                          'gold_standard_filter_method', 'split_priors_for_gold_standard', 'cv_split_ratio',
                          'split_gold_standard_for_crossvalidation', 'cv_split_axis', 'preprocessing_workflow',
                          'shuffle_prior_axis', 'write_network', 'output_dir']


class NoOutputRP(results_processor.ResultsProcessor):
    """
    Overload the existing results processor to return summary information and to only output files if specifically
    instructed to do so
    """

    network_file_name = None
    pr_curve_file_name = None
    confidence_file_name = None
    threshold_file_name = None

    def summarize_network(self, output_dir, gold_standard, priors, confidence_threshold=default.DEFAULT_CONF,
                          precision_threshold=default.DEFAULT_PREC):
        """
        Take the betas and rescaled beta_errors, construct a network, and test it against the gold standard
        :param output_dir: str
            Path to write files into. Don't write anything if this is None.
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard to test the network against
        :param priors: pd.DataFrame [G x K]
            Prior data
        :param confidence_threshold: float
            Threshold for confidence scores
        :param precision_threshold: float
            Threshold for precision
        :return aupr: float
            Returns the AUPR calculated from the network and gold standard
        :return num_conf: int
            The number of interactions above the confidence threshold
        :return num_prec: int
            The number of interactions above the precision threshold
        """

        pr_calc = results_processor.RankSummaryPR(self.rescaled_betas, gold_standard, filter_method=self.filter_method)
        beta_sign, beta_nonzero = self.summarize(self.betas)
        beta_threshold = self.passes_threshold(beta_nonzero, len(self.betas), self.threshold)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        network_data = {'beta.sign.sum': beta_sign, 'var.exp.median': resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=pr_calc.aupr), level=0)

        # Plot PR curve & Output results to a TSV
        self.write_output_files(pr_calc, output_dir, priors, beta_threshold, network_data)

        num_conf = pr_calc.num_over_conf_threshold(confidence_threshold)
        num_prec = pr_calc.num_over_precision_threshold(precision_threshold)

        return pr_calc.aupr, num_conf, num_prec


# Factory method to spit out a puppet workflow
def create_puppet_workflow(base_class=single_cell_workflow.SingleCellWorkflow, result_processor=NoOutputRP):
    class PuppetClass(base_class):
        """
        Standard workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and returns the model AUPR and edge counts without writing files (unless told to)
        """

        write_network = True
        network_file_name = None

        def __init__(self, kvs, rank, expr_data, meta_data, prior_data, gs_data):
            self.kvs = kvs
            self.rank = rank
            self.expression_matrix = expr_data
            self.meta_data = meta_data
            self.priors_data = prior_data
            self.gold_standard = gs_data

        def startup_run(self):
            if self.split_priors_for_gold_standard:
                self.split_priors_into_gold_standard()
            elif self.split_gold_standard_for_crossvalidation:
                self.cross_validate_gold_standard()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):
            if self.is_master():
                results = result_processor(betas, rescaled_betas, filter_method=self.gold_standard_filter_method)
                if self.write_network:
                    results.network_file_name = self.network_file_name
                    network_file_path = self.output_dir
                else:
                    results.network_file_name = None
                    network_file_path = None
                results.pr_curve_file_name = None
                results.confidence_file_name = None
                results.threshold_file_name = None
                results.write_task_files = False
                results.tasks_names = getattr(self, "tasks_names", None) # For multitask
                results = results.summarize_network(network_file_path, gold_standard, priors)
                self.aupr, self.n_interact, self.precision_interact = results
            else:
                self.aupr, self.n_interact, self.precision_interact = None, None, None

    return PuppetClass

class PuppeteerWorkflow(object):
    """
    This class contains the methods to create new child Workflow objects
    It does not extend WorkflowBase because I hate keeping track of multiinheritance patterns
    """
    write_network = True  # bool
    csv_writer = None  # csv.csvwriter
    csv_header = []  # list[]
    output_file_name = "aupr.tsv"  # str

    puppet_class = single_cell_workflow.SingleCellWorkflow
    puppet_result_processor = NoOutputRP
    regression_type = bbsr_python

    def create_writer(self):
        """
        Create a CSVWriter and stash it in self.writer
        """

        if self.is_master():
            self.create_output_dir()
            self.csv_writer = csv.writer(open(os.path.join(self.output_dir, self.output_file_name),
                                              mode="w", buffering=1), delimiter="\t", lineterminator="\n",
                                         quoting=csv.QUOTE_NONE)
            self.csv_writer.writerow(self.csv_header)

    def new_puppet(self, expr_data, meta_data, seed=default.DEFAULT_RANDOM_SEED, priors_data=None, gold_standard=None):
        """
        Create a new puppet workflow to run the inferelator
        :param expr_data: pd.DataFrame [G x N]
        :param meta_data: pd.DataFrame [N x ?]
        :param seed: int
        :param priors_data: pd.DataFrame [G x K]
        :param gold_standard: pd.DataFrame [G x K]
        :return puppet:
        """

        # Unless told otherwise, use the master priors and master gold standard
        if gold_standard is None:
            gold_standard = self.gold_standard
        if priors_data is None:
            priors_data = self.priors_data

        # Create a new puppet workflow with the factory method and pass in data on instantiation
        puppet = create_puppet_workflow(base_class = self.puppet_class, result_processor = self.puppet_result_processor)
        puppet = puppet(self.kvs, self.rank, expr_data, meta_data, priors_data, gold_standard)

        # Transfer the class variables necessary to get the puppet to dance (everything in SHARED_CLASS_VARIABLES)
        self.assign_class_vars(puppet)

        # Set the random seed into the puppet
        puppet.random_seed = seed

        # Make sure that the puppet knows the correct orientation of the expression matrix
        puppet.expression_matrix_columns_are_genes = False

        # Tell the puppet what to name stuff (if write_network is False then no output will be produced)
        puppet.network_file_name = "network_s{seed}.tsv".format(seed=seed)
        return puppet

    def assign_class_vars(self, obj):
        """
        Transfer class variables from this object to a target object
        """
        for varname in SHARED_CLASS_VARIABLES:
            try:
                setattr(obj, varname, getattr(self, varname))
                utils.Debug.vprint("Variable {var} set to child".format(var=varname), level=2)
            except AttributeError:
                utils.Debug.vprint("Variable {var} not assigned to parent".format(var=varname))

        self.regression_type.patch_workflow(obj)


class SingleCellPuppeteerWorkflow(single_cell_workflow.SingleCellWorkflow, PuppeteerWorkflow):
    seeds = default.DEFAULT_SEED_RANGE

    # Output TSV controllers
    write_network = True  # bool
    csv_writer = None  # csv.csvwriter
    csv_header = ["Seed", "AUPR", "Num_Interacting"]  # list[]
    output_file_name = "aupr.tsv"  # str

    # How to sample
    stratified_sampling = False
    stratified_batch_lookup = default.DEFAULT_METADATA_FOR_BATCH_CORRECTION
    sample_with_replacement = True

    def run(self):
        np.random.seed(self.random_seed)
        self.startup()
        self.create_writer()
        auprs = self.modeling_method()

    def compute_activity(self):
        # Compute activities in the puppet, not in the puppetmaster
        pass

    def single_cell_normalize(self):
        # Normalize and impute in the puppet, not in the puppetmaster
        pass

    def set_gold_standard_and_priors(self):
        # Split priors for a gold standard in the puppet, not in the puppetmaster
        self.priors_data = self.input_dataframe(self.priors_file)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)

    def align_priors_and_expression(self):
        # Align the priors and expression in the puppet, not in the puppetmaster
        pass

    def shuffle_priors(self):
        # Do any shuffles in the puppet, not in the puppetmaster
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
                puppet.run()
                size_aupr = (s_ratio, len(nidx), seed, puppet.aupr, puppet.n_interact, puppet.precision_interact)
                aupr_data.extend(size_aupr)
                if self.is_master():
                    self.csv_writer.writerow(size_aupr)
        return aupr_data


class SingleCellDropoutConditionSampling(SingleCellPuppeteerWorkflow):
    csv_header = ["Dropout", "Seed", "AUPR", "Num_Confident_Int", "Num_Precision_Int"]

    # Sampling batches
    sample_batches_to_size = default.DEFAULT_BATCH_SIZE
    stratified_sampling = True
    drop_column = None

    def modeling_method(self, *args, **kwargs):

        self.factor_indexes = self.factor_singles()
        auprs = self.auprs_for_condition_dropin()
        auprs.extend(self.auprs_for_condition_dropout())
        return auprs

    def auprs_for_condition_dropout(self):
        """
        Run modeling on all data, and then on data where each factor from `drop_column` has been removed
        :return:
        """
        # Run the modeling on all data
        aupr_data = [self.auprs_for_index("all_dropout", pd.Series(True, index=self.meta_data.index))]

        if self.drop_column is None:
            return aupr_data

        # For all of the factors in `drop_column`, iterate through and remove them one by one, modeling on the rest
        for r_name, r_idx in self.factor_indexes.items():
            aupr_data.extend(self.auprs_for_index(r_name, r_idx))
        return aupr_data

    def auprs_for_condition_dropin(self):
        """
        Run modeling on all data, and then on data where each factor from `drop_column` has been removed one
        :return:
        """
        # Run the modeling on all data with resizing
        drop_in_sizing = int(self.sample_batches_to_size / len(self.factor_indexes))
        aupr_data = [self.auprs_for_index("all_dropin", pd.Series(True, index=self.meta_data.index),
                                          sample_size=drop_in_sizing)]

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
            puppet.run()
            drop_data = (r_name, seed, puppet.aupr, puppet.n_interact, puppet.precision_interact)
            aupr_data.append(drop_data)
            if self.is_master():
                self.csv_writer.writerow(drop_data)
        return aupr_data