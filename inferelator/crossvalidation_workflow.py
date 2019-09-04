from __future__ import print_function

import os
import csv

from inferelator import utils, default
from inferelator.regression import base_regression
from inferelator.postprocessing import results_processor
from inferelator import workflow

# The variable names that get set in the main workflow, but need to get copied to the puppets
SHARED_CLASS_VARIABLES = ['tf_names', 'gene_metadata', 'gene_list_index', 'num_bootstraps', 'mi_sync_path',
                          'count_minimum', 'gold_standard_filter_method', 'cv_split_ratio',
                          'split_gold_standard_for_crossvalidation', 'cv_split_axis', 'preprocessing_workflow',
                          'shuffle_prior_axis', 'write_network', 'output_dir', 'tfa_driver', 'drd_driver',
                          'result_processor_driver', 'prior_manager', 'meta_data_task_column']


class PuppeteerWorkflow(object):
    """
    This class contains the methods to create new child Workflow objects
    It needs to be multi-inherited with a Workflow class (this needs to be the left side)
    This does not extend WorkflowBase because multiinheritence from subclasses of the same super is a NIGHTMARE
    """
    write_network = True  # bool
    csv_writer = None  # csv.csvwriter
    csv_header = ()  # list[]
    output_file_name = "aupr.tsv"  # str

    # Workflow types for the crossvalidation
    cv_regression_type = base_regression.RegressionWorkflow
    cv_workflow_type = workflow.WorkflowBase
    cv_result_processor_type = results_processor.ResultsProcessor

    def create_writer(self):
        """
        Create a CSVWriter and stash it in self.writer
        """

        if self.is_master():
            self.create_output_dir()
            self.csv_writer = csv.writer(open(os.path.expanduser(os.path.join(self.output_dir, self.output_file_name)),
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
        puppet = create_puppet_workflow(base_class=self.cv_workflow_type,
                                        regression_class=self.cv_regression_type,
                                        result_processor_class=self.cv_result_processor_type)
        puppet = puppet(expr_data, meta_data, priors_data, gold_standard)

        # Transfer the class variables necessary to get the puppet to dance (everything in SHARED_CLASS_VARIABLES)
        self.assign_class_vars(puppet)

        # Set the random seed into the puppet
        puppet.random_seed = seed

        # Tell the puppet what to name stuff (if write_network is False then no output will be produced)
        puppet.network_file_name = "network_s{seed}.tsv".format(seed=seed)
        puppet.pr_curve_file_name = "pr_curve_s{seed}.pdf".format(seed=seed)
        return puppet

    def assign_class_vars(self, obj):
        """
        Transfer class variables from this object to a target object
        """
        for varname in SHARED_CLASS_VARIABLES:
            try:
                setattr(obj, varname, getattr(self, varname))
                utils.Debug.vprint("Variable {var} set to child".format(var=varname), level=3)
            except AttributeError:
                utils.Debug.vprint("Variable {var} not assigned to parent".format(var=varname), level=2)


# Factory method to spit out a puppet workflow
def create_puppet_workflow(regression_class=base_regression.RegressionWorkflow,
                           base_class=workflow.WorkflowBase,
                           result_processor_class=None):

    puppet_parent = workflow.create_inferelator_workflow(regression=regression_class, workflow=base_class)

    class PuppetClass(puppet_parent):
        """
        Standard workflow except it takes all the data as references to __init__ instead of as filenames on disk or
        as environment variables, and returns the model AUPR and edge counts without writing files (unless told to)
        """

        write_network = True
        network_file_name = None
        pr_curve_file_name = None
        initialize_mp = False

        def __init__(self, expr_data, meta_data, prior_data, gs_data):
            self.expression_matrix = expr_data
            self.meta_data = meta_data
            self.priors_data = prior_data
            self.gold_standard = gs_data

        def startup_run(self):
            # Skip all of the data loading
            self.process_priors_and_gold_standard()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):

            if self.is_master():

                # Create a processor
                rp = self.result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                                  metric=self.metric)

                # Assign task names if they're a thing
                rp.tasks_names = getattr(self, "tasks_names", None)

                # Process into results
                self.results = rp.summarize_network(None, gold_standard, priors)

                # Write the network if that flag is set
                if self.write_network:
                    self.results.clear_output_file_names()
                    self.results.network_file_name = self.network_file_name
                    self.results.curve_file_name = self.pr_curve_file_name
                    self.results.write_result_files(self.make_path_safe(self.output_dir))
            else:
                self.results = None

    if result_processor_class is not None:
        PuppetClass.result_processor_driver = result_processor_class

    return PuppetClass
