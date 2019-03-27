from inferelator_ng import utils, default
from inferelator_ng.postprocessing import results_processor
from inferelator_ng import workflow

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
def create_puppet_workflow(base_class=workflow.WorkflowBase, result_processor=NoOutputRP):
    class PuppetClass(base_class):
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
            if self.split_priors_for_gold_standard:
                self.split_priors_into_gold_standard()
            elif self.split_gold_standard_for_crossvalidation:
                self.cross_validate_gold_standard()

        def emit_results(self, betas, rescaled_betas, gold_standard, priors):
            if self.is_master():
                results = result_processor(betas, rescaled_betas, filter_method=self.gold_standard_filter_method)
                if self.write_network:
                    results.network_file_name = self.network_file_name
                    results.pr_curve_file_name = self.pr_curve_file_name
                    network_file_path = self.output_dir
                else:
                    results.network_file_name = None
                    results.pr_curve_file_name = None
                    network_file_path = None
                results.confidence_file_name = None
                results.threshold_file_name = None
                results.write_task_files = False
                results.tasks_names = getattr(self, "tasks_names", None) # For multitask
                results = results.summarize_network(network_file_path, gold_standard, priors)
                self.aupr, self.n_interact, self.precision_interact = results
            else:
                self.aupr, self.n_interact, self.precision_interact = None, None, None

    return PuppetClass