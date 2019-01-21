from inferelator_ng import single_cell_workflow
from inferelator_ng import single_cell
from inferelator_ng import utils

utils.Debug.set_verbose_level(1)

"""Create a new workflow"""
workflow = single_cell_workflow.SingleCellBBSRWorkflow()    # Use BBSR for regularization
#workflow = single_cell_workflow.SingleCellMENWorkflow()    # Use elastic net for regularization


"""Set basic parameters"""
workflow.num_bootstraps = 2
workflow.random_seed = 1

"""
Set a temp directory for sharing MI arrays between processes. This must be writable by ALL processes on ALL nodes.
If None, pickle and share through KVS (a sneaky memory leak with KVSClient can be a problem with many bootstraps)
"""
workflow.mi_sync_path = None


"""Set location of input and output files (this overrides any environment variables set by SLURM)"""
workflow.input_dir = 'data'
# workflow.output_dir = None


"""Append to a path variable (this way we can add a subdirectory to a path set by SLURM)"""
workflow.append_to_path('input_dir', 'yeast')


"""Set file names"""

#workflow.expression_matrix_file = "expression.tsv"
#workflow.tf_names_file = "tf_names.tsv"
#workflow.meta_data_file = "meta_data.tsv"
#workflow.priors_file = "gold_standard.tsv"
#workflow.gold_standard_file = "gold_standard.tsv"
#workflow.gene_list_file = "genes.tsv"


"""Extract metadata from the expression_matrix_file if necessary"""
workflow.extract_metadata_from_expression_matrix = True     # Boolean flag to extract metadata
# workflow.expression_matrix_metadata = ['Condition']       # A list of columns to extract

"""Set parameters for the gold standard and results processing module"""

workflow.split_priors_into_gold_standard_ratio = 0.5            # What fraction of the priors should be gold standard
workflow.gold_standard_filter_method = 'keep_all_gold_standard' # Process results over the entire gold standard
#workflow.gold_standard_filter_method = 'overlap'               # Process results only where we have model predictions


"""Set threshold for modeling on a specific gene"""
workflow.count_minimum = 0.025                              # This is per-cell

"""
Set the preprocessing workflow in a list
Each list element is a 2-tuple with a function from single_cell and dict with keyword arguments needed
They will be executed in order
"""

workflow.preprocessing_workflow.append((single_cell.normalize_medians_for_batch, dict(batch_factor_column='Condition')))
#workflow.preprocessing_workflow.append((single_cell.normalize_expression_to_one, dict()))
#workflow.preprocessing_workflow.append((single_cell.normalize_multiBatchNorm, dict()))
workflow.preprocessing_workflow.append((single_cell.log2_data, dict()))
#workflow.preprocessing_workflow.append((single_cell.impute_magic_expression, dict()))
#workflow.preprocessing_workflow.append((single_cell.impute_SIMLR_expression, dict()))

"""
Use the metadata to modify transcription factor activity
This requires a meta_data column to use to identify transcription factors to modify
"""
workflow.modify_activity_from_metadata = False          # Boolean flag
# workflow.metadata_expression_lookup = None

"""Execute modeling workflow"""
workflow.run()