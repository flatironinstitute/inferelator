"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
from inferelator_ng import workflow
from inferelator_ng import design_response_translation  # added python design_response
from inferelator_ng.tfa import TFA
from inferelator_ng.results_processor import ResultsProcessor
from inferelator_ng import bbsr_python
from inferelator_ng import utils
from inferelator_ng import default


class TFAWorkFlow(workflow.WorkflowBase):
    # Design/response parameters
    delTmin = default.DEFAULT_DELTMIN
    delTmax = default.DEFAULT_DELTMAX
    tau = default.DEFAULT_TAU

    # Result processing parameters
    gold_standard_filter_method = default.DEFAULT_GS_FILTER_METHOD

    # Regression implementation
    regression_type = bbsr_python

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
        self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def startup_run(self):
        self.set_regression_type()
        self.get_data()

    def startup_finish(self):
        self.compute_common_data()
        self.compute_activity()

    def set_regression_type(self):
        self.regression_type.patch_workflow(self)

    def run_regression(self):
        betas = []
        rescaled_betas = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            utils.Debug.vprint('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps), level=0)
            np.random.seed(self.random_seed + idx)
            current_betas, current_rescaled_betas = self.run_bootstrap(bootstrap)
            if self.is_master():
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)

        return betas, rescaled_betas

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.design, self.half_tau_response)
        self.design = TFA_calculator.compute_transcription_factor_activity()
        self.half_tau_response = None

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        if self.is_master():
            self.create_output_dir()
            self.results_processor = ResultsProcessor(betas, rescaled_betas,
                                                      filter_method=self.gold_standard_filter_method)
            self.results_processor.summarize_network(self.output_dir, gold_standard, priors)

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.filter_expression_and_priors()
        drd = design_response_translation.PythonDRDriver(return_half_tau=True)
        utils.Debug.vprint('Creating design and response matrix ... ')
        drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
        self.design, self.response, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        self.expression_matrix = None
