"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
import os
from inferelator_ng import workflow
from inferelator_ng import design_response_translation  # added python design_response
from inferelator_ng.tfa import TFA
from inferelator_ng.results_processor import ResultsProcessor
from inferelator_ng import mi
from inferelator_ng import bbsr_python
import datetime
from inferelator_ng import utils


class BBSR_TFA_Workflow(workflow.WorkflowBase):
    # Design/response parameters
    delTmin = 0
    delTmax = 120
    tau = 45

    # Regression parameters
    num_bootstraps = 2

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
        self.get_data()
        self.compute_common_data()
        self.compute_activity()

    def startup_finish(self):
        pass

    def run_regression(self):
        betas = []
        rescaled_betas = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            utils.Debug.vprint('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps), level=0)
            current_betas, current_rescaled_betas = self.run_bootstrap(bootstrap)
            if self.is_master():
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)

        return betas, rescaled_betas

    def run_bootstrap(self, bootstrap):
        X = self.design.iloc[:, bootstrap]
        Y = self.response.iloc[:, bootstrap]
        utils.Debug.vprint('Calculating MI, Background MI, and CLR Matrix', level=0)
        (clr_matrix, mi_matrix) = mi.MIDriver(kvs=self.kvs, rank=self.rank).run(X, Y)
        utils.Debug.vprint('Calculating betas using BBSR', level=0)
        ownCheck = utils.ownCheck(self.kvs, self.rank, chunk=25)
        return bbsr_python.BBSR_runner().run(X, Y, clr_matrix, self.priors_data, self.kvs, self.rank, ownCheck)

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
            if self.output_dir is None:
                self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.makedirs(self.output_dir)
            except OSError:
                pass
            self.results_processor = ResultsProcessor(betas, rescaled_betas)
            self.results_processor.summarize_network(self.output_dir, gold_standard, priors)

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.filter_expression_and_priors()
        drd = design_response_translation.PythonDRDriver()
        utils.Debug.vprint('Creating design and response matrix ... ')
        drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
        self.design, self.response = drd.run(self.expression_matrix, self.meta_data)
        drd.tau = self.tau / 2
        self.design, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        self.expression_matrix = None
