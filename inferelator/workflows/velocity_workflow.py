from inferelator.workflows.workflow_base import _H5AD, _HDF5, _TSV
from inferelator.workflows.tfa_workflow import TFAWorkFlow
from inferelator.workflows.single_cell_workflow import SingleCellWorkflow
from inferelator.utils import InferelatorData, Debug, Validator as check
from inferelator.tfa.pinv_tfa import ActivityOnlyPinvTFA
from inferelator.preprocessing.velocity import extract_transcriptional_output

import numpy as np
import pandas as pd
import warnings

_VELOCITY_FILE_TYPES = [_TSV, _HDF5, _H5AD]


class VelocityWorkflow(SingleCellWorkflow):
    _velocity_data = None
    _velocity_file_name = None
    _velocity_file_type = None
    _velocity_h5_layer = None

    _decay_constants = None
    _decay_constant_var_col = None
    _decay_constant_file_name = None
    _decay_constant_file_type = None
    _decay_constant_h5_layer = None

    _global_decay_constant = None
    _gene_specific_decay_constant = False
    _gene_sample_decay_constant = False

    tfa_driver = ActivityOnlyPinvTFA

    def get_data(self):
        super(VelocityWorkflow, self).get_data()
        self.load_velocity()
        self.load_decay()

    def startup_finish(self):
        self.single_cell_normalize()
        self._check_decays()
        self._align_velocity()
        TFAWorkFlow.startup_finish(self)

    def set_velocity_parameters(
        self,
        velocity_file_name=None,
        velocity_file_type=None,
        velocity_file_layer=None
    ):
        """
        Set velocity file arguments

        :param velocity_file_name: File name that has velocity data.
            Orientation must match expression data
        :type velocity_file_name: str
        :param velocity_file_type: Type of file to load.
            Accepts 'tsv', 'h5ad' and 'hdf5'.
        :type velocity_file_type: str
        :param velocity_file_layer: If the loaded file is an h5 file,
            which layer should be used for velocity
        :type velocity_file_layer: str
        """

        self._set_with_warning(
            "_velocity_file_name",
            velocity_file_name
        )

        self._set_with_warning(
            "_velocity_h5_layer",
            velocity_file_layer
        )

        self._set_file_type(
            "_velocity_file_type",
            velocity_file_type
        )

    def set_decay_parameters(
        self,
        global_decay_constant=None,
        gene_metadata_decay_constant_column=None,
        decay_constant_file=None,
        decay_constant_file_type=None,
        decay_constant_file_layer=None
    ):
        """
        Set decay arguments

        :param global_decay_constant: Set decay constant for all genes
        :type global_decay_constant: numeric
        :param decay_constant_file: File containing decay constants for
            each gene
        :type decay_constant_file: str
        :param gene_metadata_decay_constant_column: Column in gene_metadata
            which has decay constants
        :type gene_metadata_decay_constant_column: str
        :param decay_constant_type: Type of file to load.
            Accepts 'tsv', 'h5ad' and 'hdf5'.
        :type decay_constant_type: str
        :param decay_constant_layer: If the loaded file is an h5 file,
            which layer should be used for decay constants
        :type decay_constant_layer: str
        """

        self._set_with_warning(
            "_global_decay_constant",
            global_decay_constant
        )

        self._set_with_warning(
            "_decay_constant_file_name",
            decay_constant_file
        )

        self._set_with_warning(
            "_decay_constant_var_col",
            gene_metadata_decay_constant_column
        )

        self._set_with_warning(
            "_decay_constant_h5_layer",
            decay_constant_file_layer
        )

        self._set_file_type(
            "_decay_constant_file_type",
            decay_constant_file_type
        )

    def _set_file_type(
        self,
        selfattr,
        filetype
    ):

        if filetype is None:
            return None

        elif filetype.lower() in _VELOCITY_FILE_TYPES:
            return self._set_with_warning(
                selfattr,
                filetype.lower()
            )

        else:
            raise ValueError(
                f"file_type must be in {_VELOCITY_FILE_TYPES}; "
                f"{filetype} provided"
            )

    def load_velocity(self):

        self._velocity_data = self.read_gene_data_file(
            self._velocity_file_name,
            self._velocity_file_type,
            file_layer=self._velocity_h5_layer,
            gene_data_file=self.gene_metadata_file,
            meta_data_file=self.meta_data_file
        )
        self._velocity_data.name = "Velocity"

    def load_decay(self, file=None):

        file = file if file is not None else self._decay_constant_file_name

        # If a decay constant file has been provided
        # Load it
        if self._decay_constant_file_name is not None:

            # Check and see if it's a 2-column TSV
            # Which is per-gene decay constants
            # Or if it's larger
            # Which is full samples x genes decay constants
            if self._decay_constant_file_type == _TSV:
                _check_df_size = self.read_data_frame(
                    file,
                    nrows=2
                )

                if _check_df_size.shape[1] == 1:
                    self._gene_specific_decay_constant = True

                else:
                    self._gene_sample_decay_constant = True

            else:
                self._gene_sample_decay_constant = True

            # Load a two-column TSV and reindex to match
            # The gene names
            if self._gene_specific_decay_constant:
                self._decay_constants = self.read_data_frame(
                    file
                ).reindex(
                    self.gene_names
                )

            # Load a samples x genes data object
            # With the standard loader
            else:

                self._decay_constants = self.read_gene_data_file(
                    self._decay_constant_file_name,
                    self._decay_constant_file_type,
                    file_layer=self._decay_constant_h5_layer,
                    gene_data_file=self.gene_metadata_file,
                    meta_data_file=self.meta_data_file
                )

                self._decay_constants.name = "Decay Constants"

        elif self._decay_constant_var_col is not None:

            if self._decay_constant_var_col not in self.data.gene_data.columns:
                raise ValueError(
                    f"Column {self._decay_constant_var_col} not in "
                    f"gene metadata columns {self.data.gene_data.columns.tolist()}"
                )

            Debug.vprint(
                f"Using decay constants from gene metadata "
                f"column {self._decay_constant_var_col} ",
                level=0
            )

            self._decay_constants = self.data.gene_data[
                self._decay_constant_var_col
            ].copy()

            self._gene_specific_decay_constant = True

        elif self._global_decay_constant is not None:

            Debug.vprint(
                f"Setting decay constant {self._global_decay_constant} "
                "for all genes",
                level=0
            )

        else:
            self._global_decay_constant = 0

            Debug.vprint(
                "Setting decay constant to 0 for all genes; "
                "expression and decay will not be included in model",
                level=0
            )

    def _align_velocity(self):

        # Find intersection of velocity and expression
        keep_genes, _lose_genes = self._aligned_names(
            self._velocity_data.gene_names,
            self.data.gene_names
        )

        # Also find intersection of decay if it's a obs x genes
        # data object
        if isinstance(self._decay_constants, InferelatorData):
            keep_genes, _lose_genes_2 = self._aligned_names(
                keep_genes,
                self._decay_constants.gene_names
            )

            _lose_genes = _lose_genes.union(_lose_genes_2)

            self._decay_constants.trim_genes(
                remove_constant_genes=False,
                trim_gene_list=keep_genes
            )

        # If it's a genes dataframe, reindex it to the other
        # data objects and fill NA with zeros
        elif isinstance(self._decay_constants, pd.DataFrame):

            self._decay_constants = self._decay_constants.reindex(
                keep_genes
            )

            _no_decays = pd.isna(self._decay_constants)

            if _no_decays.sum() > 0:
                Debug.vprint(
                    f"{_no_decays.sum()} genes not in decay constant "
                    "data; decay constant for these genes set to 0. "
                    f"[{', '.join(keep_genes[_no_decays])}]"
                )

                self._decay_constants = self._decay_constants.fillna(0)

        Debug.vprint(
            "Aligning expression and dynamic data on "
            f"{len(keep_genes)} genes; {len(_lose_genes)} removed"
        )

        # Trim genes
        self._velocity_data.trim_genes(
            remove_constant_genes=False,
            trim_gene_list=keep_genes
        )

        self.data.trim_genes(
            remove_constant_genes=False,
            trim_gene_list=keep_genes
        )

        assert check.indexes_align(
            (self._velocity_data.gene_names, self.data.gene_names)
        )

        assert check.indexes_align(
            (self._velocity_data.sample_names, self.data.sample_names)
        )

    def compute_common_data(self):
        pass

    def _recalculate_design(self):
        """
        Calculate dX/dt + lambda * X as response and A_hat as design
        :return:
        """

        self.response = self._combine_expression_velocity(
            self.data,
            self._velocity_data
        )

        self.data = None
        self._velocity_data = None
        self._decay_constants = None

        self.design = self.tfa_driver().compute_transcription_factor_activity(
            self.priors_data,
            self.response
        )

    def _check_decays(self):
        """
        Check for negative decay parameters
        """

        if (self._global_decay_constant is not None
            and self._global_decay_constant < 0):

            warnings.warn(
                f"Decay constant is negative ({self._global_decay_constant})"
                "this is highly inadvisable"
            )

        if self._decay_constants is not None:

            _n_neg = np.sum(self._decay_constants.values < 0)

            if _n_neg > 0:
                warnings.warn(
                    f"Negative decay parameters (n = {_n_neg}); "
                    "this is highly inadvisable"
                )

    def _combine_expression_velocity(self, expression, velocity):
        """
        Calculate dX/dt + lambda * X
        :param expression:
        :param velocity:
        :return:
        """

        if self._global_decay_constant:
            return extract_transcriptional_output(
                expression,
                velocity,
                global_decay=self._global_decay_constant
            )

        elif self._decay_constants is None:
            return extract_transcriptional_output(
                expression,
                velocity
            )

        # If a full samples x genes decay constant
        # Array has been provided, use it directly
        # As dx/dt + \lambda * X = f(A)
        elif self._gene_sample_decay_constant:
            return extract_transcriptional_output(
                expression,
                velocity,
                gene_and_sample_decay=self._decay_constants
            )

        # If gene-specific decay constants are provided
        # broadcast them to the full array
        # As dx/dt + \lambda * X = f(A)
        else:
            return extract_transcriptional_output(
                expression,
                velocity,
                gene_specific_decay=self._decay_constants
            )

    @staticmethod
    def _aligned_names(idx1, idx2):

        keep_genes = idx1.intersection(idx2)
        lose_genes = idx1.symmetric_difference(idx2)

        return keep_genes, lose_genes
