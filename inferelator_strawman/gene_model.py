
import numpy as np

class GeneModel:

    """
    A Gene model encapsulates
        - a list of gene namess
        - a list of transcription factor names
        - a heuristic for computing gene responses in time series.
    A Gene model knows how to construct design and response matrices
    for condition lists and time series.
    """
    # XXXX These calculations are not optimized to keep them
    # readable, because the effort in this initialization is negligible compared
    # to the regression calculation.

    def __init__(self, gene_names, tf_names, transition_response):
        self.gene_names = gene_names
        self.tf_names = tf_names
        self.transition_response = transition_response

    def response_matrix(self, conditions):
        """
        For a list of conditions return the response array A for the genes
        where A[rowi,colj] gives the response in condition i for gene j.
        It is assumed the conditions are not part of a time series.
        """
        genes = self.gene_names
        nrows = len(conditions)
        ncols = len(genes)
        A = np.zeros((nrows, ncols))
        for (col_j, gene_j) in enumerate(genes):
            for (row_i, condition_i) in enumerate(conditions):
                A[row_i, col_j] = condition_i.response_scalar(gene_j)
        return A

    def response_matrix_ts(self, ts):
        """
        Compute the response array A for the genes for a timeeseries ts
        where A[rowi, colj] gives the response for gene j in condition i
        of the timeseries ordered conditions.
        """
        genes = self.gene_names
        condition_names = ts.get_condition_name_order()
        ncols = len(genes)
        nrows = len(condition_names)
        A = np.zeros((nrows, ncols))
        gene_response = self.transition_response.gene_response
        for (col_j, gene_j) in enumerate(genes):
            for (row_i, cond_name_i) in enumerate(condition_names):
                params = ts.get_response_parameters(cond_name_i, gene_j)
                A[row_i, col_j] = gene_response(params)
        return A

    def design_matrix(self, conditions):
        """
        For a list of conditions return the design matrix X for the
        transcription factors where X[rowi, colj] gives the level for
        tf j in condition i.
        """
        tfs = self.tf_names
        ncols = len(tfs)
        nrows = len(conditions)
        X = np.zeros((nrows, ncols))
        for (row_i, condition_i) in enumerate(conditions):
            X[row_i, :] = condition_i.design_vector(tfs)
        return X

    def design_matrix_ts(self, ts):
        """
        Compute the design matrix X for tfs where X[rowi, colj]
        gives the level for tf j in condition i of the timeseries
        ordered conditions.
        """
        conditions = ts.get_condition_order()
        # XXXX is this right?
        return self.design_matrix(conditions)
