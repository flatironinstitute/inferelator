"""
Run BSubtilis Network Inference with TFA BBSR. 
"""

import numpy as np
import os
from workflow import WorkflowBase

class Bsubtilis_Bbsr_Workflow(WorkflowBase):

    # Common configuration parameters
    input_dir = 'bsubtilis'

    def __init__(self):
        # Do nothing (all configuration is external to init)
        pass

    def input_path(self, filename):
        return os.path.abspath(os.path.join(self.input_dir, filename))

    def input_file(self, filename, strict=True):
        path = self.input_path(filename)
        if os.path.exists(path):
            return open(path)
        elif not strict:
            return None
        raise ValueError("no such file " + repr(path))

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """
        raise NotImplementedError  # implement in subclass

    def get_priors(self):
        """
        Generate sequence of priors objects for run.
        """
        raise NotImplementedError  # implement in subclass

    def get_bootstraps(self):
        """
        Generate sequence of bootstrap parameter objectss for run.
        """
        raise NotImplementedError  # implement in subclass

    def emit_results(self, priors):
        """
        Output result report(s) for workflow run.
        """
        raise NotImplementedError  # implement in subclass
