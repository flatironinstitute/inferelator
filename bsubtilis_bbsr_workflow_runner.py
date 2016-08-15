from inferelator_ng.bsubtilis_bbsr_workflow import Bsubtilis_Bbsr_Workflow

workflow = Bsubtilis_Bbsr_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/bsubtilis'
workflow.num_bootstraps = 20
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.run() 
