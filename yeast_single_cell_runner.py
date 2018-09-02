from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels["verbose"])

#Build the workflow
workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data'
workflow.append_to_path('input_dir', 'yeast')
workflow.expression_matrix_file = "expr_transpose_100k_5000umi.tsv"
workflow.expression_matrix_transpose = True
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.async_start = True
workflow.async_chunk = 3
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 1

#Run the workflow
workflow.run()