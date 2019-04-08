"""
This is an example run script containing all of the options that can be set. You should choose which options are
best for your application.
"""

# Choose a level of information that will be printed to stdout
# -1 will silence almost all output
# 0 will provide information about the stage of processing that is occurring
# 1 will provide information about processing and some information about filtering
# 2+ will provide fine-grained information about many steps during preprocessing
# (Settings above 1 are not recommended for standard usage)

from inferelator import utils

utils.Debug.set_verbose_level(1)

# Select a processing engine
from inferelator.distributed.inferelator_mp import MPControl

# The simplest engine is the "local" engine which will use one core
# Pros: Runs without installing any additional libraries
# Cons: Runs very slowly

# MPControl.set_multiprocess_engine("local")

# Running on a single computer can be made more efficient with the "multiprocessing" engine
# This engine relies on the pathos implementation of multiprocessing
#
# The number of desired cores can be set
# Pros: Runs on a single computer very efficiently
# Cons: Is not compatible with multiple clustered nodes

MPControl.set_multiprocess_engine("multiprocessing")
MPControl.client.processes = 3

# Also available for a single computer is the "dask-local" engine
# This engine relies on the dask library
#
# Pros: Runs on a single computer
# Cons: Has more overhead than pathos (slower)

# MPControl.set_multiprocess_engine("dask-local")
# MPControl.client.processes = 3

# Scaling to a cluster requires one of the cluster-specific engines, like the "kvs" engine
# This engine relies on the KVSSTCP library
# The SBATCH_VARS variable in the inferelator.default package may need to be adjusted for the specific cluster used
#
# Pros: Can be set up on most clusters without reconfiguration
# Cons: Is memory inefficient. Is not robust to errors.

# MPControl.set_multiprocess_engine("kvs")

# The "dask-cluster" engine maximizes scalability
# This engine relies on the dask and dask_jobqueue libraries
#
# Pros: Scales to handle the largest datasets
# Cons: May require considerable effort to match to the cluster configuration

# MPControl.set_multiprocess_engine("dask-cluster")
# MPControl.client.minimum_cores = 400
# MPControl.client.maximum_cores = 400
# MPControl.client.walltime = '48:00:00'

# Select a workflow and regression method
from inferelator import workflow

# BBSR for bulk data
wflow = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")

# BBSR for single-cell data
# wflow = workflow.inferelator_workflow(regression="bbsr", workflow="single-cell")

# Multitask learning
# wflow = workflow.inferelator_workflow(regression="amusr", workflow="amusr")

# Set the filenames for input data
wflow.input_dir = 'data/yeast'
wflow.expression_matrix_file = 'expression.tsv'
wflow.tf_names_file = "tf_names.tsv"
wflow.priors_file = "yeast-motif-prior.tsv"
wflow.gold_standard_file = "gold_standard.tsv"

# Set the timeseries processing variables
wflow.delTmax = 110
wflow.delTmin = 0
wflow.tau = 45

# Set the number of bootstraps and the random seed
wflow.num_bootstraps = 2
wflow.random_seed = 42

# Run the workflow
network = wflow.run()