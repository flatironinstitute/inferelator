"""
Run BSubtilis Network Inference with TFA BBSR. 
"""

import numpy as np
import os
from workflow import WorkflowBase
import design_response_R
from tfa import TFA
import mi_R
import bbsr_R
import datetime

class Bsubtilis_Bbsr_Workflow(WorkflowBase):

    def __init__(self):
        # Do nothing (all configuration is external to init)
        pass

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)
        self.mi_clr_driver = mi_R.MIDriver()
        self.brd = bbsr_R.BBSR_driver()
        self.get_data()
        self.compute_common_data()
        self.compute_activity()
        self.results = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            print 'Bootstrap {} of {}'.format(idx, self.num_bootstraps)
            X = self.activity.ix[:, bootstrap]
            Y = self.response.ix[:, bootstrap]
            print 'Calculating MI, Background MI, and CLR Matrix'
            (self.clr_matrix, self.mi_matrix) = self.mi_clr_driver.run(X, Y)
            print 'Calculating betas using BBSR'
            (betas, resc) = self.brd.run(X, Y, self.clr_matrix, self.priors_data)
            self.results.append((betas, resc))
        self.emit_results()

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        self.priors_data = self.filter_prior_matrix(self.priors_data, self.expression_matrix, self.tf_names)
        print 'Creating design and response matrix ... '
        drd = design_response_R.DRDriver()
        drd.delTmin = self.delTmin
        drd.delTmax = self.delTmax
        drd.tau = self.tau
        (self.design, self.response) = drd.run(self.expression_matrix, self.meta_data)

        # compute half_tau_response
        print 'Setting up TFA specific response matrix ... '
        drd.tau = self.tau / 2
        (self.design, self.half_tau_response) = drd.run(self.expression_matrix, self.meta_data)

    def filter_prior_matrix(self, priors_data, expression_matrix, tf_names):
        """
        Filter the prior matrix to only include columns in the tfs names list
        and rows in the expression matrix row names list
        """
        return self.priors_data.loc[expression_matrix.index.tolist(), tf_names]

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        print 'Computing Transcription Factor Activity ... '
        TFA_calculator = TFA(self.priors_data, self.design, self.half_tau_response)
        self.activity = TFA_calculator.compute_transcription_factor_activity()

    def emit_results(self):
        """
        Output result report(s) for workflow run.
        """
        output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(output_dir)
        for idx, result in enumerate(self.results):
            result[0].to_csv(os.path.join(output_dir, 'betas_{}.tsv'.format(idx)), sep = '\t')
            result[1].to_csv(os.path.join(output_dir,'resc_{}.tsv'.format(idx)), sep = '\t')
