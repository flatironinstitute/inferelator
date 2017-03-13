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

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run, using a binary version of the gold standard
        """
        return super(Yeast_Bbsr_Workflow, self).emit_results(betas, rescaled_betas, abs(gold_standard), priors)
