# Set threading control variables if they're not already set

import os
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")

from inferelator.workflow import inferelator_workflow
from inferelator.crossvalidation_workflow import CrossValidationManager
from inferelator.utils import inferelator_verbose_level
from inferelator.distributed.inferelator_mp import MPControl

from inferelator.workflows import (
    amusr_workflow,
    single_cell_workflow,
    tfa_workflow,
    velocity_workflow
)

from inferelator.regression.base_regression import PreprocessData
