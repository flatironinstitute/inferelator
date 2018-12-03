"""
Run Single Cell Network Inference
"""
import pandas as pd
import numpy as np
import types

from inferelator_ng.tfa import TFA
from inferelator_ng import utils
from inferelator_ng import tfa_workflow
from inferelator_ng import elasticnet_python
from inferelator_ng import single_cell

EXPRESSION_MATRIX_METADATA = ['Genotype', 'Genotype_Group', 'Replicate', 'Condition', 'tenXBarcode']
GENE_LIST_INDEX_COLUMN = 'SystematicName'
GENE_LIST_LOOKUP_COLUMN = 'Name'
METADATA_FOR_TFA_ADJUSTMENT = 'Genotype_Group'
METADATA_FOR_BATCH_CORRECTION = 'Condition'


class SingleCellWorkflow(object):
    # Gene list
    gene_list_file = None
    gene_list = None
    gene_list_index = GENE_LIST_INDEX_COLUMN

    # Single-cell expression data manipulations
    count_minimum = None  # float
    expression_matrix_transpose = True  # bool
    extract_metadata_from_expression_matrix = False  # bool
    expression_matrix_metadata = EXPRESSION_MATRIX_METADATA  # str

    # Normalization method flags
    normalize_counts_to_one = False  # bool
    normalize_batch_medians = False  # bool
    log_two_plus_one = False  # bool
    ln_plus_one = False  # bool
    magic_imputation = False  # bool
    batch_correction_lookup = METADATA_FOR_BATCH_CORRECTION  # str

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

    def startup_finish(self):
        # Filter expression and priors to align
        self.single_cell_normalize()
        self.filter_expression_and_priors()
        self.compute_activity()

    def filter_expression_and_priors(self):

        # Transpose the expression matrix (if it's N x G instead of G x N)
        if self.expression_matrix_transpose:
            self.expression_matrix = self.expression_matrix.transpose()

        if self.count_minimum is not None:
            keep_genes = self.expression_matrix.sum(axis=1) >= (self.count_minimum * self.expression_matrix.shape[0])
            self.expression_matrix = self.expression_matrix.loc[keep_genes, :]

        # If gene_list_file is set, read a list of genes in and then filter the expression and priors to this list
        if self.gene_list_file is not None:
            self.read_genes()
            genes = self.gene_list[self.gene_list_index]
            self.expression_matrix = self.expression_matrix.loc[self.expression_matrix.index.intersection(genes)]
            self.priors_data = self.priors_data.loc[self.priors_data.index.intersection(genes)]

        # Only keep stuff from the expression matrix that's got counts
        self.expression_matrix = self.expression_matrix.loc[~(self.expression_matrix.sum(axis=1) == 0)]

        # Make sure that the priors align to the expression matrix
        self.priors_data = self.priors_data.reindex(index=self.expression_matrix.index).fillna(value=0)

        # Trim to the tf_names list
        tf_keepers = list(set(self.tf_names).intersection(set(self.priors_data.columns.tolist())))
        self.priors_data = self.priors_data.loc[:, tf_keepers]

    def single_cell_normalize(self):

        if self.normalize_counts_to_one and self.normalize_batch_medians:
            raise ValueError("One normalization method at a time")
        if self.log_two_plus_one and self.ln_plus_one:
            raise ValueError("One logging method at a time")

        if self.expression_matrix.isnull().values.any():
            raise ValueError("NaN values are present prior to normalization in the expression matrix")

        # Normalize UMI counts per cell (0-1 so that sum(counts) = 1 for each cell)
        if self.normalize_counts_to_one:
            utils.Debug.vprint('Normalizing UMI counts per cell ... ')
            self.expression_normalize_to_one()

        # Batch normalize so that all batches have the same median UMI count
        if self.normalize_batch_medians:
            utils.Debug.vprint('Normalizing multiple batches ... ')
            self.batch_normalize_medians()

        # log2(x+1) all data
        if self.log_two_plus_one:
            utils.Debug.vprint('Logging data ... ')
            self.log2_data()

        # ln2(x+1) all data points
        if self.ln_plus_one:
            utils.Debug.vprint('Logging data ... ')
            self.ln_data()

        # Use MAGIC (van Dijk et al Cell, 2018, 10.1016/j.cell.2018.05.061) to impute data
        if self.magic_imputation:
            utils.Debug.vprint('Imputing data with MAGIC ... ')
            self.magic_expression()

        if self.expression_matrix.isnull().values.any():
            raise ValueError("NaN values have been introduced into the expression matrix by normalization")

    def read_genes(self):

        with self.input_path(self.gene_list_file) as genefh:
            self.gene_list = pd.read_table(genefh, **self.file_format_settings)

    def expression_normalize_to_one(self):
        # Get UMI counts for each cell
        umi = single_cell.umi(self.expression_matrix)
        if (umi == 0).values.any():
            raise ValueError("There are cells with no counts in the expression matrix")

        # Divide each cell's raw count data by the total number of UMI counts for that cell
        self.expression_matrix = self.expression_matrix.astype(float)
        self.expression_matrix = self.expression_matrix.divide(umi, axis=1)

    def batch_normalize_medians(self, batch_factor_column=METADATA_FOR_BATCH_CORRECTION):
        # Get UMI counts for each cell
        umi = single_cell.umi(self.expression_matrix)

        # Create a new dataframe with the UMI counts and the factor to batch correct on
        median_umi = pd.DataFrame({'umi': umi, batch_factor_column: self.meta_data[batch_factor_column]})

        # Group and take the median UMI count for each batch
        median_umi = median_umi.groupby(batch_factor_column).agg('median')

        # Convert to a correction factor
        median_umi['umi'] = median_umi['umi'] / median_umi['umi'].median()

        # Apply the correction factor to all the data batch-wise. Do this with numpy because pandas is a glacier.
        new_expression_data = np.ndarray((0, self.expression_matrix.shape[1]), dtype=np.dtype(float))
        new_meta_data = pd.DataFrame(columns=self.meta_data.columns)
        for batch, corr_factor in median_umi.iterrows():
            rows = self.meta_data[batch_factor_column] == batch
            new_expression_data = np.vstack((new_expression_data,
                                             self.expression_matrix.loc[rows, :].values / corr_factor['umi']))
            new_meta_data = pd.concat([new_meta_data, self.meta_data.loc[rows, :]])
        self.expression_matrix = pd.DataFrame(new_expression_data,
                                              index=new_meta_data.index,
                                              columns=self.expression_matrix.columns)
        self.meta_data = new_meta_data

    def log2_data(self):
        self.expression_matrix = np.log2(self.expression_matrix + 1)

    def ln_data(self):
        self.expression_matrix = np.log1p(self.expression_matrix)

    def magic_expression(self):
        import magic
        np.random.seed(self.random_seed)
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
        if self.gene_list is not None:
            genes = self.gene_list.loc[self.gene_list[self.gene_list_lookup].isin(genotypes), :]
        else:
            genes = self.design.index.isin(genotypes)

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


class SingleCellBBSRWorkflow(SingleCellWorkflow, tfa_workflow.BBSR_TFA_Workflow):
    pass


class SingleCellMENWorkflow(SingleCellWorkflow, elasticnet_python.MEN_Workflow):
    pass
