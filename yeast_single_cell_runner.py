from inferelator_ng.single_cell_bbsr_tfa_workflow import Single_Cell_BBSR_TFA_Workflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels["verbose"])

# Build the workflow
workflow = Single_Cell_BBSR_TFA_Workflow()
# Common configuration parameters
workflow.append_to_path('input_dir', 'yeast')
workflow.expression_matrix_file = 'bootstrap_10k_1500umi.tsv'
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 42

# Run the workflow
workflow.preprocess_data()
workflow.run()
