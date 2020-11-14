# Load modules
from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager

# Set verbosity level to "Talky"
inferelator_verbose_level(1)


# Set the location of the input data and the desired location of the output files

DATA_DIR = '../data/bsubtilis'
OUTPUT_DIR = '~/bsubtilis_inference/'

PRIORS_FILE_NAME = 'gold_standard.tsv.gz'
GOLD_STANDARD_FILE_NAME = 'gold_standard.tsv.gz'
TF_LIST_FILE_NAME = 'tf_names.tsv'

BSUBTILIS_1_EXPRESSION = 'GSE67023_expression.tsv.gz'
BSUBTILIS_1_METADATA = 'GSE67023_meta_data.tsv'

BSUBTILIS_2_EXPRESSION = 'expression.tsv.gz'
BSUBTILIS_2_METADATA = 'meta_data.tsv'

CV_SEEDS = list(range(42, 52))

# Multiprocessing uses the pathos implementation of multiprocessing (with dill instead of cPickle)
# This is suited for a single computer but will not work on a distributed cluster

n_cores_local = 10
local_engine = True

# Multiprocessing needs to be protected with the if __name__ == 'main' pragma
if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = n_cores_local
    MPControl.connect()

# Inference on B. subtilis data set 1 (GSE67023) with BBSR
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a crossvalidation wrapper
cv_wrap = CrossValidationManager()

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="bbsr", workflow="tfa")

worker.set_file_paths(input_dir=DATA_DIR,
                      output_dir=OUTPUT_DIR,
                      expression_matrix_file=BSUBTILIS_1_EXPRESSION,
                      tf_names_file=TF_LIST_FILE_NAME,
                      meta_data_file=BSUBTILIS_1_METADATA,
                      priors_file=PRIORS_FILE_NAME,
                      gold_standard_file=GOLD_STANDARD_FILE_NAME)
worker.set_file_properties(expression_matrix_columns_are_genes=False)

worker.set_run_parameters(num_bootstraps=5)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "bsubtilis_1")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker

# Run
cv_wrap.run()

# Inference on B. subtilis data set 2 (GSE27219) with BBSR
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a crossvalidation wrapper
cv_wrap = CrossValidationManager()

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="bbsr", workflow="tfa")

worker.set_file_paths(input_dir=DATA_DIR,
                      output_dir=OUTPUT_DIR,
                      expression_matrix_file=BSUBTILIS_2_EXPRESSION,
                      tf_names_file=TF_LIST_FILE_NAME,
                      meta_data_file=BSUBTILIS_2_METADATA,
                      priors_file=PRIORS_FILE_NAME,
                      gold_standard_file=GOLD_STANDARD_FILE_NAME)
worker.set_file_properties(expression_matrix_columns_are_genes=False)

worker.set_run_parameters(num_bootstraps=5)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "bsubtilis_2")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker

# Run
cv_wrap.run()

# Inference on individual data sets with BBSR
# A final network is generated from the two separate networks
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a crossvalidation wrapper
cv_wrap = CrossValidationManager()

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="bbsr-by-task", workflow="multitask")
worker.set_file_paths(input_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                      gold_standard_file=GOLD_STANDARD_FILE_NAME)

# Create tasks
task1 = worker.create_task(task_name="Bsubtilis_1",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_1_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_1_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task1.set_file_properties(expression_matrix_columns_are_genes=False)

task2 = worker.create_task(task_name="Bsubtilis_2",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_2_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_2_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)

worker.set_run_parameters(num_bootstraps=5)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "bsubtilis_1_2_STL")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker

# Run
cv_wrap.run()

# Inference on individual data sets with AMuSR
# Using the crossvalidation wrapper
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
# Each run is seeded differently (and therefore has different holdouts)

# Create a crossvalidation wrapper
cv_wrap = CrossValidationManager()

# Assign variables for grid search
cv_wrap.add_gridsearch_parameter('random_seed', CV_SEEDS)

# Create a worker
worker = inferelator_workflow(regression="amusr", workflow="multitask")
worker.set_file_paths(input_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                      gold_standard_file=GOLD_STANDARD_FILE_NAME)

# Create tasks
task1 = worker.create_task(task_name="Bsubtilis_1",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_1_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_1_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task1.set_file_properties(expression_matrix_columns_are_genes=False)

task2 = worker.create_task(task_name="Bsubtilis_2",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_2_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_2_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)

worker.set_run_parameters(num_bootstraps=5)
worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True, cv_split_ratio=0.2)
worker.append_to_path("output_dir", "bsubtilis_1_2_MTL")

# Assign the worker to the crossvalidation wrapper
cv_wrap.workflow = worker

# Run
cv_wrap.run()

# Final network
# Create a worker
worker = inferelator_workflow(regression="amusr", workflow="multitask")
worker.set_file_paths(input_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                      gold_standard_file=GOLD_STANDARD_FILE_NAME)
# Create tasks
task1 = worker.create_task(task_name="Bsubtilis_1",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_1_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_1_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task1.set_file_properties(expression_matrix_columns_are_genes=False)

task2 = worker.create_task(task_name="Bsubtilis_2",
                           input_dir=DATA_DIR,
                           expression_matrix_file=BSUBTILIS_2_EXPRESSION,
                           tf_names_file=TF_LIST_FILE_NAME,
                           meta_data_file=BSUBTILIS_2_METADATA,
                           priors_file=PRIORS_FILE_NAME,
                           workflow_type="tfa")
task2.set_file_properties(expression_matrix_columns_are_genes=False)

worker.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=False, cv_split_ratio=None)
worker.append_to_path("output_dir", "MTL_Final")
worker.set_run_parameters(num_bootstraps=50, random_seed=100)
final_network = worker.run()
