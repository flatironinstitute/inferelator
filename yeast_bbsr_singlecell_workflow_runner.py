from inferelator_ng.single_cell_workflow import SingleCellWorkflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels["verbose"])

#Build the workflow
workflow = SingleCellWorkflow()
# Common configuration parameters
workflow.input_dir = 'data'
workflow.append_to_path('input_dir', 'yeast')
workflow.expression_matrix_file = '101718_SS_Subset_Data.tsv.gz'
workflow.priors_file = "yeast-motif-prior.tsv"
workflow.num_bootstraps = 2
workflow.random_seed = 1

#Run the workflow
workflow.run()