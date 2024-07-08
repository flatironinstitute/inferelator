__version__ = '0.6.3'

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
