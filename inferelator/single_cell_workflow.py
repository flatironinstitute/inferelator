"""
Run Single Cell Network Inference
"""
import pandas as pd
import numpy as np

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator import tfa_workflow
from inferelator.preprocessing import single_cell
from inferelator import default


class SingleCellWorkflow(tfa_workflow.TFAWorkFlow):
    # Gene list
    gene_metadata_file = default.DEFAULT_GENE_LIST_FILE
    gene_metadata = None
    gene_list_index = default.DEFAULT_GENE_LIST_INDEX_COLUMN

    # Single-cell expression data manipulations
    count_minimum = default.DEFAULT_COUNT_MINIMUM  # float
    expression_matrix_columns_are_genes = default.DEFAULT_EXPRESSION_DATA_IS_SAMPLES_BY_GENES  # bool
    extract_metadata_from_expression_matrix = default.DEFAULT_EXTRACT_METADATA_FROM_EXPR  # bool
    expression_matrix_metadata = default.DEFAULT_EXPRESSION_MATRIX_METADATA  # str

    # Preprocessing workflow holder
    preprocessing_workflow = None

    # Shuffle priors for a negative control
    shuffle_prior_axis = None

    def startup_finish(self):
        # If the expression matrix is [G x N], transpose it for preprocessing
        if not self.expression_matrix_columns_are_genes:
            self.expression_matrix = self.expression_matrix.transpose()

        # Preprocess the single-cell data based on the preprocessing steps added to the workflow
        self.single_cell_normalize()

        # Transpose the expression data to [G x N], filter to gene_list, and align the priors to the expression data
        self.filter_expression_and_priors()

        # Compute TFA
        self.compute_activity()

    def read_metadata(self, file=None):
        # If the metadata is embedded in the expression matrix, extract it
        # Otherwise call the super read_metadata
        if self.extract_metadata_from_expression_matrix:
            self.meta_data = self.expression_matrix.loc[:, self.expression_matrix_metadata].copy()
            self.expression_matrix = self.expression_matrix.drop(self.expression_matrix_metadata, axis=1)
        else:
            super(SingleCellWorkflow, self).read_metadata(file=file)

    def filter_expression_and_priors(self):
        # Transpose the expression matrix to convert from [N x G] to [G x N]
        self.expression_matrix = self.expression_matrix.transpose()

        # If gene_list_file is set, read a list of genes in and then filter the expression and priors to this list
        if self.gene_metadata is None and self.gene_metadata_file is not None:
            self.read_genes()

        if self.gene_metadata is not None:
            genes = self.gene_metadata[self.gene_list_index]
            utils.Debug.vprint("Filtering expression and priors to {le} genes from list".format(le=len(genes)), level=1)
            self.expression_matrix = self.expression_matrix.loc[self.expression_matrix.index.intersection(genes)]
            utils.Debug.vprint("Expression data filtered to {sh}".format(sh=self.expression_matrix.shape), level=1)
            self.priors_data = self.priors_data.loc[self.priors_data.index.intersection(genes)]
            utils.Debug.vprint("Priors data filtered to {sh}".format(sh=self.priors_data.shape), level=1)

        self.align_priors_and_expression()
        self.shuffle_priors()

    def align_priors_and_expression(self):
        # Make sure that the priors align to the expression matrix
        self.priors_data = self.priors_data.reindex(index=self.expression_matrix.index).fillna(value=0)

        # Trim to the tf_names list
        tf_keepers = pd.Index(self.tf_names).intersection(pd.Index(self.priors_data.columns))
        self.priors_data = self.priors_data.loc[:, tf_keepers]

    def single_cell_normalize(self):
        """
        Single cell normalization. Requires expression_matrix to be all numeric, and to be [N x G].
        Executes all preprocessing workflow steps from the preprocessing_workflow list that's set by the
        add_preprocess_step() class function
        """

        assert check.dataframe_is_numeric(self.expression_matrix)

        self.expression_matrix, self.meta_data = single_cell.filter_genes_for_count(self.expression_matrix,
                                                                                    self.meta_data,
                                                                                    count_minimum=self.count_minimum)

        if np.sum(~np.isfinite(self.expression_matrix.values), axis=None) > 0:
            raise ValueError("NaN values are present prior to normalization in the expression matrix")

        if self.preprocessing_workflow is not None:
            for sc_func, sc_kwargs in self.preprocessing_workflow:
                sc_kwargs['random_seed'] = self.random_seed
                self.expression_matrix, self.meta_data = sc_func(self.expression_matrix, self.meta_data, **sc_kwargs)

        if np.sum(~np.isfinite(self.expression_matrix.values), axis=None) > 0:
            raise ValueError("NaN values have been introduced into the expression matrix by normalization")

    def read_genes(self):
        """
        Read in a list of genes which should be modeled for network inference
        """

        self.gene_metadata = self.input_dataframe(self.gene_metadata_file, index_col=None)

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        TFA_calculator = self.tfa_driver(self.priors_data, self.expression_matrix, self.expression_matrix)
        self.design = TFA_calculator.compute_transcription_factor_activity()
        self.response = self.expression_matrix
        self.expression_matrix = None

    def add_preprocess_step(self, fun, **kwargs):
        if self.preprocessing_workflow is None:
            self.preprocessing_workflow = []
        self.preprocessing_workflow.append((fun, kwargs))
