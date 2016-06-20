
import numpy as np
import pandas as pd
from . import condition

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

    def meta_data_tsv(self, conds, tss):
        """
        return string TSV representation for condition meta data for timeseries's and conditions'
        """
        L = [condition.Condition.META_DATA_HEADER]
        for cond in conds:
            L.append(cond.meta_data_tsv_line())
        for ts in tss:
            L.append(ts.meta_data_tsv_lines())
        return "".join(L)

    def expression_data_frame(self, conds, tss):
        """
        Return a pandas data frame for all conditions and conditions in timeseries.
        """
        # order tfs before non-tfs
        tfs = self.tf_names
        tfset = set(tfs)
        non_tfs = [g for g in self.gene_names if g not in tfset]
        new_index = list(tfs) + list(non_tfs)
        # non-time-series before time-series conditions
        all_conditions = list(conds)
        for ts in tss:
            all_conditions.extend(ts.get_condition_order())
        gene_mappings = [c.gene_mapping for c in all_conditions]
        result = pd.concat(gene_mappings, axis=1).reindex(new_index)
        return result

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

    def design_and_response(self, steady_state_conditions, time_series=None):
        """
        Generate a design and response matrix information for conditions
        and time series.
        
        Parameters:
        -----------
        gene_model: GeneModel
            The gene model to derive the matrices.
        steady_state_conditions: list of Condition
            Conditions not in time series.
        time_series: list of TimeSeries
            Sequence of condition time series.

        Returns
        -------
        DesignAndResponseMatrices
        """
        return DesignAndResponseMatrices(self, steady_state_conditions, time_series)

class DesignAndResponseMatrices:

    """
    Container for design and response matrix and information used to
    derive the design and response matrices,

    - self.design gives the design matrix.
    - self.response gives the response matrix
    - self.all_conditions gives the condition order associated with the matrix rows,
    - self.gene_model gives the gene model used to derive the matrices.
    - self.steady_state_conditions gives the conditions not in timeseries.
    - self.timeseries gives the timeseries.

    Parameters:
    -----------
    gene_model: GeneModel
        The gene model to derive the matrices.
    steady_state_conditions: list of Conditions
        Conditions not in time series.
    time_series: list of TimeSeries
        Sequence of condition time series.
    """

    def __init__(self, gene_model, steady_state_conditions, time_series=None):
        if time_series is None:
            time_series = []
        self.gene_model = gene_model
        self.steady_state_conditions = steady_state_conditions
        self.time_series = time_series
        all_conditions = steady_state_conditions[:]
        design_stack = []
        response_stack = []
        design_stack.append(gene_model.design_matrix(steady_state_conditions))
        response_stack.append(gene_model.response_matrix(steady_state_conditions))
        for ts in time_series:
            design_stack.append(gene_model.design_matrix_ts(ts))
            response_stack.append(gene_model.response_matrix_ts(ts))
            all_conditions.extend(ts.get_condition_order())
        self.design = np.concatenate(design_stack)
        self.response = np.concatenate(response_stack)
        self.all_conditions = all_conditions