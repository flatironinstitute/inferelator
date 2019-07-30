# Load modules

from inferelator import utils
from inferelator.distributed.inferelator_mp import MPControl

from inferelator import workflow

# Set verbosity level to "Talky"
utils.Debug.set_verbose_level(1)

# Set the location of the input data and the desired location of the output files

DATA_DIR = '../data/yeast'
OUTPUT_DIR = '~/yeast_inference/'

EXPRESSION_FILE_NAME = 'expression.tsv.gz'
META_DATA_FILE_NAME = 'meta_data.tsv'
GOLD_STANDARD_FILE_NAME = 'gold_standard.tsv'
TF_LIST_FILE_NAME = 'tf_names_restrict.tsv'

GENE_METADATA_FILE_NAME = 'orfs.tsv'
GENE_METADATA_COLUMN = 'SystematicName'

CV_SEEDS = list(range(42,52))

# Multiprocessing uses the pathos implementation of multiprocessing (with dill instead of cPickle)
# This is suited for a single computer but will not work on a distributed cluster

n_cores_local = 3
local_engine = True

if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = n_cores_local
    MPControl.connect()

# Define the general run parameters

def set_up_workflow(wkf):
    wkf.input_dir = DATA_DIR
    wkf.output_dir = OUTPUT_DIR
    wkf.expression_matrix_file = EXPRESSION_FILE_NAME
    wkf.meta_data_file = META_DATA_FILE_NAME
    wkf.tf_names_file = TF_LIST_FILE_NAME
    wkf.gold_standard_file = GOLD_STANDARD_FILE_NAME
    wkf.gene_metadata_file = GENE_METADATA_FILE_NAME
    wkf.gene_list_index = GENE_METADATA_COLUMN
    wkf.expression_matrix_columns_are_genes = False
    wkf.num_bootstraps = 5
    return wkf

# Inference with BBSR (crossvalidation)
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
for random_seed in CV_SEEDS:
    worker = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
    worker = set_up_workflow(worker)
    worker.split_gold_standard_for_crossvalidation = True
    worker.cv_split_ratio = 0.2
    worker.random_seed = random_seed
    worker.append_to_path('output_dir', 'bbsr_cv_' + str(random_seed))
    worker.run()
    del worker

# Inference with Elastic Net (crossvalidation)
# Run the regression 10 times and hold 20% of the gold standard out of the priors for testing each time
for random_seed in CV_SEEDS:
    worker = workflow.inferelator_workflow(regression="elasticnet", workflow="tfa")
    worker = set_up_workflow(worker)
    worker.split_gold_standard_for_crossvalidation = True
    worker.cv_split_ratio = 0.2
    worker.random_seed = random_seed
    worker.append_to_path('output_dir', 'elasticnet_cv_' + str(random_seed))
    worker.run()
    del worker

# Final network
worker = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
worker = set_up_workflow(worker)
worker.append_to_path('output_dir', 'final')
worker.split_gold_standard_for_crossvalidation = False
worker.cv_split_ratio = None
worker.num_bootstraps = 50
worker.random_seed = 100
final_network = worker.run()
del worker