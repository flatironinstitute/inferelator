from __future__ import print_function

import os
import csv

from inferelator import utils, default
from inferelator.regression import base_regression
from inferelator.postprocessing import results_processor
from inferelator.postprocessing import model_performance
from inferelator import workflow


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

        pr_calc = model_performance.RankSummaryPR(self.rescaled_betas, gold_standard, filter_method=self.filter_method)
        beta_sign, beta_nonzero = self.summarize(self.betas)
        beta_threshold = self.passes_threshold(beta_nonzero, len(self.betas), self.threshold)
        resc_betas_mean, resc_betas_median = self.mean_and_median(self.rescaled_betas)
        network_data = {'beta.sign.sum': beta_sign, 'var.exp.median': resc_betas_median}

        utils.Debug.vprint("Model AUPR:\t{aupr}".format(aupr=pr_calc.aupr), level=0)

        # Plot PR curve & Output results to a TSV
        self.network_data = self.write_output_files(pr_calc, output_dir, priors, beta_threshold, network_data)

        num_conf = pr_calc.num_over_conf_threshold(confidence_threshold)
        num_prec = pr_calc.num_over_precision_threshold(precision_threshold)

        return pr_calc.aupr, num_conf, num_prec


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

    cv_regression_type = base_regression.RegressionWorkflow
    cv_workflow_type = workflow.WorkflowBase
    cv_result_processor_type = NoOutputRP

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
                           result_processor_class=NoOutputRP):

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
                results = result_processor_class(betas, rescaled_betas, filter_method=self.gold_standard_filter_method)
                if self.write_network:
                    results.network_file_name = self.network_file_name
                    results.pr_curve_file_name = self.pr_curve_file_name
                    network_file_path = self.make_path_safe(self.output_dir)
                else:
                    results.network_file_name = None
                    results.pr_curve_file_name = None
                    network_file_path = None
                results.confidence_file_name = None
                results.threshold_file_name = None
                results.write_task_files = False
                results.tasks_names = getattr(self, "tasks_names", None)  # For multitask
                summary = results.summarize_network(network_file_path, gold_standard, priors)

                self.network = results.network_data

                if isinstance(summary, tuple):
                    self.aupr, self.n_interact, self.precision_interact = summary
                else:
                    self.aupr = summary
            else:
                self.aupr, self.n_interact, self.precision_interact = None, None, None

    return PuppetClass
