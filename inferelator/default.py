"""
This file sets default constants that are used throughout the package.
Objects set as defaults are set in classes (so that we don't have to import anything here).
Mostly I did this so I could easily reuse defaults and change them to match my data structure.
Don't look at me like that.
"""

"""Default Environmental Variable Lookup"""

# This is a dict, keyed by the class setattr variable name, of tuples (env name, coercion function, default value)
SBATCH_VARS = dict(output_dir=('RUNDIR', str, None),
                   input_dir=('DATADIR', str, None),
                   rank=('SLURM_PROCID', int, 0),
                   cores=('SLURM_NTASKS_PER_NODE', int, 1),
                   tasks=('SLURM_NTASKS', int, 1),
                   node=('SLURM_NODEID', int, 0),
                   num_nodes=('SLURM_JOB_NUM_NODES', int, 1))

SBATCH_VARS_FOR_KVS = ["rank", "cores", "tasks", "node", "num_nodes"]
SBATCH_VARS_FOR_WORKFLOW = ["output_dir", "input_dir"]

"""Default Data File Settings"""

DEFAULT_PD_INPUT_SETTINGS = dict(sep="\t")
DEFAULT_EXPRESSION_FILE = "expression.tsv"
DEFAULT_TFNAMES_FILE = "tf_names.tsv"
DEFAULT_METADATA_FILE = "meta_data.tsv"
DEFAULT_PRIORS_FILE = "gold_standard.tsv"
DEFAULT_GOLDSTANDARD_FILE = "gold_standard.tsv"

"""Default TFAWorkflow Parameters"""
DEFAULT_DELTMIN = 0
DEFAULT_DELTMAX = 120
DEFAULT_TAU = 45
DEFAULT_GS_FILTER_METHOD = 'keep_all_gold_standard'

"""Defaults For Regression"""
# Default number of predictors to include in the model
DEFAULT_nS = 10

# Default weight for priors & Non-priors
# If prior_weight is the same as no_prior_weight:
#   Priors will be included in the pp matrix before the number of predictors is reduced to nS
#   They won't get special treatment in the model though
DEFAULT_prior_weight = 1
DEFAULT_no_prior_weight = 1

# Throw away the priors which have a CLR that is 0 before the number of predictors is reduced by BIC
DEFAULT_filter_priors_for_clr = False
