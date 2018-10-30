"""
Run Single Cell Network Inference with TFA BBSR
"""
import pandas as pd
import gzip
import types

from inferelator_ng import bbsr_tfa_workflow
from inferelator_ng.tfa import TFA
from inferelator_ng import utils

EXPRESSION_MATRIX_METADATA = ['Genotype', 'Genotype_Group', 'Replicate', 'Condition', 'tenXBarcode']
GENE_LIST_INDEX_COLUMN = 'SystematicName'
GENE_LIST_LOOKUP_COLUMN = 'Name'
METADATA_FOR_TFA_ADJUSTMENT = 'Genotype_Group'


class SingleCellWorkflow(bbsr_tfa_workflow.BBSR_TFA_Workflow):
    # Gene list
    gene_list_file = None
    gene_list = None
    gene_list_index = GENE_LIST_INDEX_COLUMN

    # Single-cell expression data manipulations
    expression_matrix_transpose = True
    extract_metadata_from_expression_matrix = True
    expression_matrix_metadata = EXPRESSION_MATRIX_METADATA
    minimum_reads_per_thousand_cells = 1

    # Normalization method flags
    library_normalization = True
    magic_imputation = True

    # TFA modification flags
    modify_activity_from_metadata = True
    metadata_expression_lookup = METADATA_FOR_TFA_ADJUSTMENT
    gene_list_lookup = GENE_LIST_LOOKUP_COLUMN

    def startup_run(self):

        # If the metadata is embedded in the expression matrix, monkeypatch a new read_metadata() function in
        # to properly extract it
        if self.extract_metadata_from_expression_matrix:
            def read_metadata(self):
                self.meta_data = self.expression_matrix.loc[:, self.expression_matrix_metadata].copy()
                self.expression_matrix = self.expression_matrix.drop(self.expression_matrix_metadata, axis=1)

            self.read_metadata = types.MethodType(read_metadata, self)

        # Load the usual data files for inferelator regression
        self.get_data()

        # Filter expression and priors to align
        self.filter_expression_and_priors()
        self.single_cell_normalize()
        self.compute_activity()

    def filter_expression_and_priors(self):

        # Transpose the expression matrix (if it's N x G instead of G x N)
        if self.expression_matrix_transpose:
            self.expression_matrix = self.expression_matrix.transpose()

        # If gene_list_file is set, read a list of genes in and then filter the expression and priors to this list
        if self.gene_list_file is not None:
            self.read_genes()
            genes = self.gene_list[self.gene_list_index]
            self.expression_matrix = self.expression_matrix.loc[self.expression_matrix.index.intersection(genes)]
            self.priors_data = self.priors_data.loc[self.priors_data.index.intersection(genes)]

        self.expression_matrix = self.expression_matrix.loc[~(self.expression_matrix.sum(axis=1) == 0)]
        # Make sure that the priors align to the expression matrix
        self.priors_data = self.priors_data.reindex(index=self.expression_matrix.index).fillna(value=0)

    def single_cell_normalize(self):

        # Normalize UMI counts per cell (0-1 so that sum(counts) = 1 for each cell)
        if self.library_normalization:
            utils.Debug.vprint('Normalizing UMI counts per cell ... ')
            self.normalize_expression()
        if self.magic_imputation:
            utils.Debug.vprint('Imputing data with MAGIC ... ')
            self.magic_expression()

    def read_genes(self):

        with self.input_path(self.gene_list_file) as genefh:
            self.gene_list = pd.read_table(genefh, **self.file_format_settings)

    def normalize_expression(self):
        umi = self.expression_matrix.sum(axis=0)
        self.expression_matrix = self.expression_matrix.divide(umi, axis=1)

    def magic_expression(self):
        import magic
        self.expression_matrix = magic.MAGIC().fit_transform(self.expression_matrix)

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.expression_matrix, self.expression_matrix)
        self.design = TFA_calculator.compute_transcription_factor_activity()
        self.response = self.expression_matrix
        self.expression_matrix = None

        if self.modify_activity_from_metadata:
            self.apply_metadata_to_activity()

    def scale_activity(self):
        """
        Rescale activity to between 0 and 1
        :return:
        """
        self.design = self.design - self.design.min(axis=0)
        self.design = self.design / self.design.max(axis=0)

    def apply_metadata_to_activity(self):
        """
        Set design values according to metadata
        :return:
        """

        utils.Debug.vprint('Modifying Transcription Factor Activity ... ')

        # Get the genotypes from the metadata and map them to expression data names
        self.meta_data[self.metadata_expression_lookup] = self.meta_data[self.metadata_expression_lookup].str.upper()
        genotypes = self.meta_data[self.metadata_expression_lookup].unique().tolist()
        genes = self.gene_list.loc[self.gene_list[self.gene_list_lookup].isin(genotypes), :]

        # Convert the dataframe into a dict that can be used with pd.df.map()
        gene_map = dict(zip(genes[self.gene_list_lookup].tolist(), genes[self.gene_list_index].tolist()))

        # Replace the genotypes with the gene name to modify
        self.meta_data[self.metadata_expression_lookup] = self.meta_data[self.metadata_expression_lookup].map(gene_map)

        # Map the replacement function back into the design matrix
        for idx, row in self.meta_data.iterrows():
            if pd.isnull(row[self.metadata_expression_lookup]):
                continue
            new_value = self.tfa_adj_func(row[self.metadata_expression_lookup])
            self.design.loc[row[self.metadata_expression_lookup], idx] = new_value

    def tfa_adj_func(self, gene):
        return self.design.loc[gene, :].min()
