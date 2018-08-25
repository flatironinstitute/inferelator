"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
import os
from inferelator_ng import workflow
from inferelator_ng import design_response_translation #added python design_response
from inferelator_ng.tfa import TFA
from inferelator_ng.results_processor import ResultsProcessor
from inferelator_ng import mi
from inferelator_ng import bbsr_python
import datetime
from kvsstcp.kvsclient import KVSClient
import pandas as pd
from . import utils

# Connect to the key value store service (its location is found via an
# environment variable that is set when this is started vid kvsstcp.py
# --execcmd).
kvs = KVSClient()
# Find out which process we are (assumes running under SLURM).
rank = int(os.environ['SLURM_PROCID'])

class BBSR_TFA_Workflow(workflow.WorkflowBase):

    delTmin = 0
    delTmax = 120
    tau = 45

    num_bootstraps = 2
    output_dir = None

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)

        self.get_data()
        self.compute_common_data()
        self.compute_activity()
        betas = []
        rescaled_betas = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))
            X = self.activity.ix[:, bootstrap]
            Y = self.response.ix[:, bootstrap]
            print('Calculating MI, Background MI, and CLR Matrix')
            (clr_matrix, mi_matrix) = mi.MIDriver(kvs=kvs, rank=rank).run(X, Y)
            print('Calculating betas using BBSR')
            ownCheck = utils.ownCheck(kvs, rank, chunk=25)
            current_betas,current_rescaled_betas = bbsr_python.BBSR_runner().run(X, Y, clr_matrix, self.priors_data,kvs,rank, ownCheck)
            if self.is_master():
                betas.append(current_betas)
                rescaled_betas.append(current_rescaled_betas)

        self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        print('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.design, self.half_tau_response)
        self.activity = TFA_calculator.compute_transcription_factor_activity()

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
        design_response_driver = design_response_translation.PythonDRDriver()
        print('Creating design and response matrix ... ')
        design_response_driver.delTmin = self.delTmin
        design_response_driver.delTmax = self.delTmax
        design_response_driver.tau = self.tau
        (self.design, self.response) = design_response_driver.run(self.expression_matrix, self.meta_data)

        # compute half_tau_response
        print('Setting up TFA specific response matrix ... ')
        design_response_driver.tau = self.tau / 2
        (self.design, self.half_tau_response) = design_response_driver.run(self.expression_matrix, self.meta_data)
