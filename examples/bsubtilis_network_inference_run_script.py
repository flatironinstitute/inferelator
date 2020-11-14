# Load modules
from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager

# Set verbosity level to "Talky"
inferelator_verbose_level(1)

# Set the location of the input data and the desired location of the output files

DATA_DIR = '../data/bsubtilis'
OUTPUT_DIR = '~/bsubtilis_inference/'

EXPRESSION_FILE_NAME = 'expression.tsv.gz'
PRIORS_FILE_NAME = 'gold_standard.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'gold_standard.tsv.gz'
META_DATA_FILE_NAME = 'meta_data.tsv'
TF_LIST_FILE_NAME = 'tf_names.tsv'

CV_SEEDS = list(range(42, 52))

# Multiprocessing uses the pathos implementation of multiprocessing (with dill instead of cPickle)
# This is suited for a single computer but will not work on a distributed cluster

n_cores_local = 10
local_engine = True

# Multiprocessing needs to be protected with the if __name__ == 'main' pragma
if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.set_processes(n_cores_local)
    MPControl.connect()


# Define the general run parameters
def set_up_workflow(wkf):
    wkf.set_file_paths(input_dir=DATA_DIR,
                       output_dir=OUTPUT_DIR,
                       tf_names_file=TF_LIST_FILE_NAME,
                       meta_data_file=META_DATA_FILE_NAME,
                       priors_file=PRIORS_FILE_NAME,
                       gold_standard_file=GOLD_STANDARD_FILE_NAME)
    wkf.set_expression_file(tsv=EXPRESSION_FILE_NAME)
    wkf.set_file_properties(expression_matrix_columns_are_genes=False)
    wkf.set_run_parameters(num_bootstraps=5)
    wkf.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
    return wkf


# Inference with BBSR (crossvalidation)
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a worker
worker = inferelator_workflow(regression="bbsr", workflow="tfa")
worker = set_up_workflow(worker)
worker.append_to_path("output_dir", "bbsr")

# Create a crossvalidation wrapper
cv_wrap = CrossValidationManager(worker)

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Run
cv_wrap.run()

# Inference with Elastic Net (crossvalidation)
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a worker
worker = inferelator_workflow(regression="elasticnet", workflow="tfa")
worker = set_up_workflow(worker)
worker.append_to_path("output_dir", "elastic_net")

# Set L1 ratio to 1 (This is now LASSO regression instead of Elastic Net)
# Parameters set with this function are passed to sklearn.linear_model.ElasticNetCV
worker.set_regression_parameters(l1_ratio=1, max_iter=2000)

# Create a crossvalidation wrapper and pass it the worker during __init__
cv_wrap = CrossValidationManager(worker)

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Run
cv_wrap.run()

# Final network
worker = inferelator_workflow(regression="bbsr", workflow="tfa")
worker = set_up_workflow(worker)
worker.append_to_path('output_dir', 'final')
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=False, cv_split_ratio=None)
worker.set_run_parameters(num_bootstraps=2, random_seed=100)
final_network = worker.run()
