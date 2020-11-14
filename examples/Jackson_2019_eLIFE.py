# Load modules
from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl, CrossValidationManager

# Set verbosity level to "Talky"
inferelator_verbose_level(1)

# Set the location of the input data and the desired location of the output files

DATA_DIR = '../data/yeast'
OUTPUT_DIR = '~/jackson_2019/'

EXPRESSION_FILE_NAME = '103118_SS_Data.tsv.gz'
GENE_METADATA_FILE_NAME = 'orfs.tsv'
METADATA_COLUMNS = ['TF', 'strain', 'date', 'restriction', 'mechanism', 'time']

YEASTRACT_PRIOR = "YEASTRACT_20190713_BOTH.tsv"

TF_NAMES = "tf_names_gold_standard.txt"
YEASTRACT_TF_NAMES = "tf_names_yeastract.txt"

n_cores_local = 10
local_engine = True

# Multiprocessing uses the pathos implementation of multiprocessing (with dill instead of cPickle)
# This is suited for a single computer, but will likely be too slow for the example here

if __name__ == '__main__' and local_engine:
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = n_cores_local
    MPControl.connect()

# Define the general run parameters used for all figures


def set_up_workflow(wkf):
    wkf.set_file_paths(input_dir=DATA_DIR,
                       output_dir=OUTPUT_DIR,
                       expression_matrix_file='103118_SS_Data.tsv.gz',
                       gene_metadata_file='orfs.tsv',
                       gold_standard_file='gold_standard.tsv',
                       priors_file='gold_standard.tsv',
                       tf_names_file=TF_NAMES)
    wkf.set_file_properties(extract_metadata_from_expression_matrix=True,
                            expression_matrix_metadata=METADATA_COLUMNS,
                            expression_matrix_columns_are_genes=True,
                            gene_list_index="SystematicName")
    wkf.set_crossvalidation_parameters(split_gold_standard_for_crossvalidation=True,
                                       cv_split_ratio=0.5)
    wkf.set_run_parameters(num_bootstraps=5)
    wkf.set_count_minimum(0.05)
    wkf.add_preprocess_step("log2")
    return wkf


def set_up_fig5a(wkf):
    cv_wrap = CrossValidationManager(wkf)
    cv_wrap.add_gridsearch_parameter('random_seed', list(range(42, 52)))
    return cv_wrap


# Figure 5A: No Imputation
worker = set_up_workflow(inferelator_workflow(regression="bbsr", workflow="single-cell"))
worker.append_to_path('output_dir', 'figure_5a_no_impute')

set_up_fig5a(worker).run()

# Figure 5A: Shuffled Priors
worker = set_up_workflow(inferelator_workflow(regression="bbsr", workflow="single-cell"))
worker.set_shuffle_parameters(shuffle_prior_axis=0)
worker.append_to_path('output_dir', 'figure_5a_shuffled')

set_up_fig5a(worker).run()
