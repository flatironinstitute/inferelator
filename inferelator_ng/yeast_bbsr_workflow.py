"""
Run Yeast Network Inference with TFA BBSR. 
"""

import numpy as np
import pandas as pd
import os
from workflow import WorkflowBase
from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
import design_response_R
from tfa import TFA
from results_processor import ResultsProcessor
import mi_R
import bbsr_R
import datetime

class Yeast_Bbsr_Workflow(BBSR_TFA_Workflow):

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors, that are in the tf_names list
        """
        exp_genes = self.expression_matrix.index.tolist()
        all_regs_with_data = list(set.union(set(self.expression_matrix.index.tolist()), set(self.priors_data.columns.tolist())))
        tf_names = list(set.intersection(set(self.tf_names), set(all_regs_with_data)))
        self.priors_data = self.priors_data.loc[exp_genes, tf_names]
        self.priors_data = pd.DataFrame.fillna(self.priors_data, 0)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(output_dir)
        self.results_processor = ResultsProcessor(betas, rescaled_betas)
        self.results_processor.summarize_network(output_dir, abs(gold_standard), priors)

