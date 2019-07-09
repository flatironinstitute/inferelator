# Load modules

from inferelator import utils
from inferelator.distributed.inferelator_mp import MPControl
from inferelator import workflow

# Set verbosity level to "Talky"
utils.Debug.set_verbose_level(1)

# Set the location of the input data and the desired location of the output files

DATA_DIR = '../data/th17'
OUTPUT_DIR = '~/miraldi_2019/'

EXPRESSION_FILE_NAME = 'th17_RNAseq254_DESeq2_VSDcounts.txt.gz'
GOLD_STANDARD_FILE_NAME = 'KC1p5_mmOverlap.tsv.gz'
PRIOR_FILE_NAME = 'ATAC_Th17_merged.tsv.gz'
GENE_METADATA_FILE_NAME = 'union_Th17genesMM10_Th17vTh0_FC0p58_FDR10.txt'
GENE_METADATA_COLUMN_NAME = 'Gene'
TF_LIST_FILE_NAME = 'potRegs_names.txt'

# Start Multiprocessing Engine
# Default to a single computer. Setting up a cluster is left as an exercise to the reader.

n_cores_dask = 80
activate_path = '~/.local/anaconda3/bin/activate'

# The if __name__ is __main__ pragma protects against runaway multiprocessing
# Dask requires a slurm controller in an HPC environment.
# The conda or venv activate script is necessary to set the worker environment
# This code does NOT set the environment for the current process, only for workers

if __name__ == '__main__':
    MPControl.set_multiprocess_engine("dask-cluster")
    MPControl.client.minimum_cores = n_cores_dask
    MPControl.client.maximum_cores = n_cores_dask
    MPControl.client.walltime = '48:00:00'
    MPControl.client.add_worker_env_line('module load slurm')
    MPControl.client.add_worker_env_line('module load gcc/8.3.0')
    MPControl.client.add_worker_env_line('source ' + activate_path)
    MPControl.client.cluster_controller_options.append("-p ccb")
    MPControl.connect()

    wkf = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
    wkf.input_dir = DATA_DIR
    wkf.output_dir = OUTPUT_DIR
    wkf.expression_matrix_file = EXPRESSION_FILE_NAME
    wkf.priors_file = PRIOR_FILE_NAME
    wkf.meta_data_file = None
    wkf.gold_standard_file = GOLD_STANDARD_FILE_NAME
    wkf.gene_metadata_file = GENE_METADATA_FILE_NAME
    wkf.gene_list_index = GENE_METADATA_COLUMN_NAME
    wkf.tf_names_file = TF_LIST_FILE_NAME
    wkf.expression_matrix_columns_are_genes = False
    wkf.extract_metadata_from_expression_matrix = False
    wkf.split_gold_standard_for_crossvalidation = True
    wkf.cv_split_ratio = 0.2
    wkf.num_bootstraps = 5
    network = wkf.run()