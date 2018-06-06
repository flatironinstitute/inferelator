from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng.prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase

class BBSR_TFA_Workflow_with_Prior_GS_split(BBSR_TFA_Workflow, PriorGoldStandardSplitWorkflowBase):
    """ 
        The class BBSR_TFA_Workflow_with_Prior_GS_split is a case of multiple inheritance,
        as it inherits both from BBSR_TFA_Workflow and PriorGoldStandardSplitWorkflowBase      
    """

workflow = BBSR_TFA_Workflow_with_Prior_GS_split()
# Common configuration parameters
workflow.input_dir = 'data/bsubtilis'
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.run() 
