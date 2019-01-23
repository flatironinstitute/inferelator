from inferelator_ng import single_cell_workflow
from inferelator_ng import single_cell
from inferelator_ng import utils

utils.Debug.set_verbose_level(1)

"""Create a new workflow"""

## Single-task inference
workflow = single_cell_workflow.SingleCellWorkflow()

## Multi-task inference
# from inferelator_ng import amusr_workflow
# workflow = amusr_workflow.SingleCellMultiTask

"""Set basic parameters"""
workflow.num_bootstraps = 2
workflow.random_seed = 1

"""Set regression type"""

## Bayes Best Subset Regression (BBSR)
from inferelator_ng import bbsr_python

workflow.regression_type = bbsr_python

## Elastic Net
# from inferelator_ng import elasticnet_python
# workflow.regression_type = elasticnet_python

## Multitask Regression (it is not necessary to actually set this - this is the default for SingleCellMultiTask)
# from inferelator_ng import amusr_regression
# workflow.regression_type = amusr_regression

"""
Set a temp directory for sharing MI arrays between processes. This must be writable by ALL processes on ALL nodes.
If None, pickle and share through KVS (a sneaky memory leak with KVSClient can be a problem with many bootstraps).
This will only have an effect for BBSR regression (gradient descent doesn't need MI)
"""
workflow.mi_sync_path = None

"""Set location of input and output files (this overrides any environment variables)"""

# Input data path is taken from the environment variable DATADIR unless overriden here
# workflow.input_dir = 'data'

# Output data path is taken from the environment variable RUNDIR unless overriden here
# workflow.output_dir = None


"""Append to a path variable (this way we can add a subdirectory to a path set by environment variable)"""
workflow.append_to_path('input_dir', 'yeast')

"""Set file names"""

# workflow.expression_matrix_file = "expression.tsv"
# workflow.tf_names_file = "tf_names.tsv"
# workflow.meta_data_file = "meta_data.tsv"
# workflow.priors_file = "gold_standard.tsv"
# workflow.gold_standard_file = "gold_standard.tsv"
# workflow.gene_list_file = "genes.tsv"


"""Extract metadata from the expression_matrix_file if necessary"""
workflow.extract_metadata_from_expression_matrix = True  # Boolean flag to extract metadata
# workflow.expression_matrix_metadata = ['Condition']       # A list of columns to extract

"""Set parameters for cross-validation"""

# Split the gold standard and keep the proportion in cv_split_ratio
# This also removes any genes in the gold standard from the prior
workflow.split_gold_standard_for_crossvalidation = True
workflow.cv_split_ratio = 0.5

# An alternative option is to split the priors and replace the gold standard with it
# This is provided mostly for legacy reasons. Use the other thing.
# workflow.split_priors_for_gold_standard = False

"""Set parameters for the gold standard and results processing module"""

# Process results over the entire gold standard
workflow.gold_standard_filter_method = 'keep_all_gold_standard'

# Process results only where we have model predictions (this might be cheating...)
# workflow.gold_standard_filter_method = 'overlap'


"""Set threshold for modeling on a specific gene"""

# This should only be set for data which is strictly > 0 (like UMI counts)
# This is the minimum average value for a feature - any features with less than this will get discarded
# Note that setting this to 0 is pointless; there is already a filter to remove any genes with variance of 0
workflow.count_minimum = 0.025

"""
Set the preprocessing steps for the workflow. Preprocessing is implemented in inferelator_ng.single_cell.
Preprocessing steps will be executed in the order that they are set here
"""

# This preprocessing step normalizes the median UMI count for each batch by multiplying every cell in the batch by some
# correction factor. The final result is that the UMI count medians for each batch are the same
workflow.add_preprocess_step(single_cell.normalize_medians_for_batch, batch_factor_column='Condition')
# This preprocessing step normalizes the UMI count for each cell within a batch so that they are equal to the median
# UMI count for the batch. If combined with normalize_medians_for_batch, this results in all cells in the data having
# equal total UMI counts
workflow.add_preprocess_step(single_cell.normalize_sizes_within_batch, batch_factor_column='Condition')
# Log the data
workflow.add_preprocess_step(single_cell.log2_data)
# workflow.add_preprocess_step(single_cell.log10_data)
# workflow.add_preprocess_step(single_cell.ln_data)

"""
Use the metadata to modify transcription factor activity
This requires a meta_data column to use to identify transcription factors to modify
"""

workflow.modify_activity_from_metadata = False  # Boolean flag
# workflow.metadata_expression_lookup = None

"""Execute modeling workflow"""
workflow.run()
