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
        self.process_priors_and_gold_standard()

    def startup_finish(self):
        self.align_priors_and_expression()
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
        # If there is a tfa driver, run it to calculate TFA from the prior & expression data
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        tfa_calculator = self.tfa_driver(self.priors_data, self.design, self.half_tau_response)
        self.design = tfa_calculator.compute_transcription_factor_activity()
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

        if self.drd_driver is not None:
            # If there is a design-response driver, run it to create design and response
            drd = self.drd_driver(return_half_tau=True)
            utils.Debug.vprint('Creating design and response matrix ... ')
            drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
            self.design, self.response, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        else:
            # If there is no design-response driver set, use the expression data for design and response
            self.design = self.expression_matrix
            self.response, self.half_tau_response = self.expression_matrix, self.expression_matrix

        self.expression_matrix = None
