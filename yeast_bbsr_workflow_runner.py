from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/yeast'
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 0001
workflow.run() 

