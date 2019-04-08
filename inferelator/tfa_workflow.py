"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
from inferelator import workflow
from inferelator.preprocessing import design_response_translation  # added python design_response
from inferelator.preprocessing.tfa import TFA
from inferelator.postprocessing.results_processor import ResultsProcessor
from inferelator import utils
from inferelator import default


class TFAWorkFlow(workflow.WorkflowBase):
    # Design/response parameters
    delTmin = default.DEFAULT_DELTMIN
    delTmax = default.DEFAULT_DELTMAX
    tau = default.DEFAULT_TAU

    # Regression data
    design = None
    response = None
    half_tau_response = None

    # TFA implementation
    tfa_driver = TFA

    # Design-Response Driver implementation
    drd_driver = design_response_translation.PythonDRDriver

    # Result Processor implementation
    result_processor_driver = ResultsProcessor
    # Result processing parameters
    gold_standard_filter_method = default.DEFAULT_GS_FILTER_METHOD

    def run(self):
        """
        Execute workflow, after all configuration.
        """

        # Set the random seed (for bootstrap selection)
        np.random.seed(self.random_seed)

        # Call the startup workflow
        self.startup()

        # Run regression after startup
        betas, rescaled_betas = self.run_regression()

        # Write the results out to a file
        return self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def startup_run(self):
        self.get_data()

    def startup_finish(self):
        self.compute_common_data()
        self.compute_activity()

    def run_regression(self):
        raise NotImplementedError

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        TFA_calculator = self.tfa_driver(self.priors_data, self.design, self.half_tau_response)
        self.design = TFA_calculator.compute_transcription_factor_activity()
        self.half_tau_response = None

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        if self.is_master():
            self.create_output_dir()
            rp = self.result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method)
            rp.summarize_network(self.output_dir, gold_standard, priors)
            return rp.network_data

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.filter_expression_and_priors()
        drd = self.drd_driver(return_half_tau=True)
        utils.Debug.vprint('Creating design and response matrix ... ')
        drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
        self.design, self.response, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        self.expression_matrix = None
