"""
Run Single Cell Network Inference

This is the same network inference with some extra preprocessing functionality
"""
import numpy as np

from inferelator.utils import Validator as check
from inferelator import tfa_workflow
from inferelator.preprocessing import single_cell
from inferelator import default


class SingleCellWorkflow(tfa_workflow.TFAWorkFlow):
    # Single-cell expression data manipulations
    count_minimum = default.DEFAULT_COUNT_MINIMUM  # float

    # Preprocessing workflow holder
    preprocessing_workflow = None

    # Do not use a design-response driver
    drd_driver = None

    def startup_finish(self):
        # Preprocess the single-cell data based on the preprocessing steps added to the workflow
        self.single_cell_normalize()

        super(SingleCellWorkflow, self).startup_finish()

    def single_cell_normalize(self):
        """
        Single cell normalization. Requires expression_matrix to be all numeric, and to be [N x G].
        Executes all preprocessing workflow steps from the preprocessing_workflow list that's set by the
        add_preprocess_step() class function
        """

        # Transpose the expression matrix from [G x N] to [N x G] for preprocessing
        self.expression_matrix = self.expression_matrix.transpose()

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

        # Transpose the expression matrix from [N x G] to [G x N] for the rest of the workflow
        self.expression_matrix = self.expression_matrix.transpose()

    def add_preprocess_step(self, fun, **kwargs):
        if self.preprocessing_workflow is None:
            self.preprocessing_workflow = []
        self.preprocessing_workflow.append((fun, kwargs))
