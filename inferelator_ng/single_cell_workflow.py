"""
Run Single Cell Network Inference with TFA BBSR
"""
import pandas as pd
import gzip

from inferelator_ng import bbsr_tfa_workflow
from inferelator_ng.tfa import TFA
from inferelator_ng import utils

EXPRESSION_MATRIX_METADATA = ['Genotype', 'Genotype_Group', 'Replicate', 'Condition', 'tenXBarcode']

class SingleCellWorkflow(bbsr_tfa_workflow.BBSR_TFA_Workflow):

    expression_matrix_metadata = EXPRESSION_MATRIX_METADATA

    def startup_run(self):
        self.get_data()
        self.expression_matrix = self.expression_matrix.transpose()
        self.filter_expression_and_priors()
        self.normalize_expression()
        self.compute_activity()

    def read_metadata(self, file=None):
        self.meta_data = self.expression_matrix.loc[:, self.expression_matrix_metadata].copy()
        self.expression_matrix = self.expression_matrix.drop(self.expression_matrix_metadata, axis=1)

    def filter_expression_and_priors(self):
        self.expression_matrix = self.expression_matrix.loc[~(self.expression_matrix.sum(axis=1) == 0)]
        self.priors_data = self.priors_data.reindex(index = self.expression_matrix.index).fillna(value=0)

    def read_expression(self):
        """
        Read expression file in from a gzipped file
        """
        with gzip.open(self.input_path(self.expression_matrix_file), mode='r') as matfh:
            self.expression_matrix = pd.read_table(matfh, index_col=0, header=0, sep="\t")

    def normalize_expression(self):
        umi = self.expression_matrix.sum(axis=0)
        self.expression_matrix = self.expression_matrix.divide(umi, axis=1)

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.expression_matrix, self.expression_matrix)
        self.design = TFA_calculator.compute_transcription_factor_activity()
        self.response = self.expression_matrix
        self.expression_matrix = None
