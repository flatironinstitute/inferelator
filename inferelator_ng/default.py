"""
This file sets default constants that are used throughout the package.
Objects set as defaults are set in classes (so that we don't have to import anything here).
Mostly I did this so I could easily reuse defaults and change them to match my data structure.
Don't look at me like that.
"""


"""Default Data File Settings"""

DEFAULT_PD_INPUT_SETTINGS = dict(sep="\t", header=0)
DEFAULT_EXPRESSION_FILE = "expression.tsv"
DEFAULT_TFNAMES_FILE = "tf_names.tsv"
DEFAULT_METADATA_FILE = "meta_data.tsv"
DEFAULT_PRIORS_FILE = "gold_standard.tsv"
DEFAULT_GOLDSTANDARD_FILE = "gold_standard.tsv"


"""Default WorkflowBase Parameters"""
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_BOOTSTRAPS = 2
DEFAULT_GS_SPLIT_RATIO = None
DEFAULT_GS_SPLIT_AXIS = 0


"""Default TFAWorkflow Parameters"""
DEFAULT_DELTMIN = 0
DEFAULT_DELTMAX = 120
DEFAULT_TAU = 45
DEFAULT_GS_FILTER_METHOD = 'overlap'

"""Default SingleCellWorkflow Parameters"""

DEFAULT_EXPRESSION_DATA_IS_SAMPLES_BY_GENES = True

DEFAULT_EXPRESSION_MATRIX_METADATA = ['Genotype', 'Genotype_Group', 'Replicate', 'Condition', 'tenXBarcode']
DEFAULT_EXTRACT_METADATA_FROM_EXPR = False
DEFAULT_MODIFY_TFA_FROM_METADATA = False
DEFAULT_METADATA_FOR_TFA_ADJUSTMENT = 'Genotype_Group'
DEFAULT_METADATA_FOR_BATCH_CORRECTION = 'Condition'

DEFAULT_GENE_LIST_FILE = None
DEFAULT_GENE_LIST_INDEX_COLUMN = 'SystematicName'
DEFAULT_GENE_LIST_LOOKUP_COLUMN = 'Name'

DEFAULT_COUNT_MINIMUM = None

"""Default SingleCellPuppeteerWorkflow Parameters"""

# DEFAULTS FOR Puppeteer
DEFAULT_SEED_RANGE = range(42, 45)

# DEFAULTS FOR SizeSelectPuppeteer
DEFAULT_SIZE_SAMPLING = [1]
DEFAULT_MINIMUM_SAMPLE_SIZE = 10

# DEFAULTS FOR SingleCellDropoutConditionSampling
DEFAULT_BATCH_SIZE = 500

# DEFAULTS FOR NoOutputRP
DEFAULT_CONF = 0.95
DEFAULT_PREC = 0.5