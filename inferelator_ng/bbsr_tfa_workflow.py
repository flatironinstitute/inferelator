"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
import os
from . import workflow
import inferelator_ng.design_response_translation as design_response_translation #added python design_response
from inferelator_ng.tfa import TFA
from inferelator_ng.results_processor import ResultsProcessor
import inferelator_ng.mi_clr_python as mi_clr
import inferelator_ng.bbsr_python as bbsr_python
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

#This is a WorkflowBase with BBSR & TFA specific addons
class BBSR_TFA_Workflow(workflow.WorkflowBase):

    # Computed data structures
    expression_matrix = None  # expression_matrix dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe

    def run(self):
        """
        Execute workflow, after all configuration.
        """

        #Sets the random seed in np.random
        np.random.seed(self.random_seed)

        #This is the controller for running mutual information (which is still in R?)
        self.mi_clr_driver = mi_clr.MIDriver()

        #This is the controller to do bayes subset regression
        self.regression_driver = bbsr_python.BBSR_runner()

        #This is the controller for the design response matrix builder
        self.design_response_driver = design_response_translation.PythonDRDriver() #this is the python switch

        #Callback to WorkflowBase.get_data
        self.get_data()

        #expression_matrix = pd.DataFrame (from pd.read_csv) - rows are genes, columns are experiments
        #tf_names = list (from list(pd.read_csv))
        #meta_data = pd.DataFrame (from pd.read_csv) - row for each experiment
        #priors_data = pd.DataFrame - row for each gene, column for each TF
        #gold_standard = pd.DataFrame - row for each gene, column for each TF

        #All these are now in memory, having been read in from the tsvs

        self.compute_common_data()

        #design = pd.DataFrame - rows are genes, columns are experiments
        #response = pd.DataFrame - rows are genes, columns are experiments
        #half_tau_response = pd.DataFrame

        #These are the computed design response matrixes

        #Calculate the TFAs by matrix pseudoinverse
        self.compute_activity()

        #activity = pd.DataFrame - rows are TFs, columns are experiments

        betas = []
        rescaled_betas = []

        #Bootstrap sample size is the number of experiments

        for idx, bootstrap in enumerate(self.get_bootstraps()):

            print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))

            #X and Y are resampled based on bootstrap (index generated from np.random.choice(num_cols))
            X = self.activity.ix[:, bootstrap]
            Y = self.response.ix[:, bootstrap]


            print('Calculating MI, Background MI, and CLR Matrix')

            #Calculate CLR & MI if we're proc 0 or get CLR & MI from the KVS if we're not
            if 0 == rank:
                (clr_matrix, mi_matrix) = self.mi_clr_driver.run(X, Y)
                kvs.put('mi %d'%idx, (clr_matrix, mi_matrix))
            else:
                (clr_matrix, mi_matrix) = kvs.view('mi %d'%idx)


            print('Calculating betas using BBSR')

            #Create the generator to handle interprocess communication through KVS
            ownCheck = utils.ownCheck(kvs, rank, chunk=25)

            #Run the BBSR on this bootstrap
            current_betas,current_rescaled_betas = self.regression_driver.run(X, Y, clr_matrix, self.priors_data,kvs,rank, ownCheck)

            if rank: continue
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
        if 0 == rank:
            output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(output_dir)
            self.results_processor = ResultsProcessor(betas, rescaled_betas)
            self.results_processor.summarize_network(output_dir, gold_standard, priors)
