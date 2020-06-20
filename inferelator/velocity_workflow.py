from inferelator.workflow import _H5AD, _HDF5, _TSV
from inferelator.tfa_workflow import TFAWorkFlow
from inferelator.single_cell_workflow import SingleCellWorkflow
from inferelator.utils import InferelatorDataLoader, InferelatorData, Debug, Validator as check
from inferelator.preprocessing.velocity_tfa import VelocityTFA
import numpy as np


_VELOCITY_FILE_TYPES = [_TSV, _HDF5, _H5AD]


class VelocityWorkflow(SingleCellWorkflow):
    _velocity_data = None
    _velocity_file_name = None
    _velocity_file_type = None
    _velocity_h5_layer = None

    _decay_constants = None
    _use_precalculated_decay_constants = True

    tau = None
    tfa_driver = VelocityTFA

    def get_data(self):
        super(VelocityWorkflow, self).get_data()
        self.load_velocity()

    def startup_finish(self):
        self.single_cell_normalize()
        self._align_velocity()
        TFAWorkFlow.startup_finish(self)

    def set_velocity_parameters(self, velocity_file_name=None, velocity_file_type=None, velocity_file_layer=None):
        """
        Set velocity file arguments

        :param velocity_file_name: File name that has velocity data. Orientation must match expression data
        :type velocity_file_name: str
        :param velocity_file_type: Type of file to load. Accepts 'tsv', 'h5ad' and 'hdf5'.
        :type velocity_file_type: str
        :param velocity_file_layer: If the loaded file is an h5 file, which layer should be used
        :type velocity_file_layer: str
        """

        self._set_with_warning("_velocity_file_name", velocity_file_name)
        self._set_with_warning("_velocity_h5_layer", velocity_file_layer)

        if velocity_file_type is not None and velocity_file_type.lower() in _VELOCITY_FILE_TYPES:
            self._set_with_warning("_velocity_file_type", velocity_file_type)
        elif velocity_file_type is not None:
            msg = "velocity_file_type must be in {ft}".format(ft=_VELOCITY_FILE_TYPES)
            raise ValueError(msg)

    def load_velocity(self, velocity_file=None, loader_type=None):

        velocity_file = self._velocity_file_name if velocity_file is None else velocity_file
        loader_type = self._velocity_file_type if loader_type is None else loader_type
        transpose = not self.expression_matrix_columns_are_genes

        loader = InferelatorDataLoader(input_dir=self.input_dir, file_format_settings=self._file_format_settings)
        Debug.vprint("Loading velocity data from {f}".format(f=velocity_file), level=1)

        if loader_type == _TSV or loader_type is None:
            self._velocity_data = loader.load_data_tsv(velocity_file, transpose_expression_data=transpose)

        elif loader_type == _H5AD:
            self._velocity_data = loader.load_data_h5ad(velocity_file, use_layer=self._velocity_h5_layer)

        elif loader_type == _HDF5:
            self._velocity_data = loader.load_data_hdf5(velocity_file, transpose_expression_data=transpose,
                                                        use_layer=self._velocity_h5_layer)
        else:
            raise ValueError("Invalid velocity_file_type: {a}".format(a=loader_type))

        self._velocity_data.name = "Velocity"

    def _align_velocity(self):

        keep_genes = self._velocity_data.gene_names.intersection(self.data.gene_names)
        Debug.vprint("Aligning velocity and expression data on {n} genes".format(n=len(keep_genes)))

        self._velocity_data.trim_genes(remove_constant_genes=False, trim_gene_list=keep_genes)
        self.data.trim_genes(remove_constant_genes=False, trim_gene_list=keep_genes)

        assert check.indexes_align((self._velocity_data.gene_names, self.data.gene_names))
        assert check.indexes_align((self._velocity_data.sample_names, self.data.sample_names))

    def compute_common_data(self):
        pass

    def _recalculate_design(self):
        """
        Calculate dX/dt + lambda * X as response and A_hat as design
        :return:
        """

        self.response = self._combine_expression_velocity(self.data, self._velocity_data)
        self.data = None
        self._velocity_data = None

        self.design = self.tfa_driver().compute_transcription_factor_activity(self.priors_data, self.response)

    def _combine_expression_velocity(self, expression, velocity):
        """
        Calculate dX/dt + lambda * X
        :param expression:
        :param velocity:
        :return:
        """

        assert check.indexes_align((expression.gene_names, velocity.gene_names))
        assert check.indexes_align((expression.sample_names, velocity.sample_names))

        if self._decay_constants is not None:
            Debug.vprint("Using preloaded decay constants in _decay_constants")
            decay_constants = self._decay_constants
        elif self.tau is not None:
            Debug.vprint("Calculating decay constants for tau {t}".format(t=self.tau))
            decay_constants = np.repeat(1 / self.tau, expression.num_genes)
        elif "decay_constants" in velocity.gene_data.columns and self._use_precalculated_decay_constants:
            Debug.vprint("Extracting decay constants from {n}".format(n=velocity.name))
            decay_constants = velocity.gene_data["decay_constants"].values
        elif "decay_constants" in expression.gene_data.columns and self._use_precalculated_decay_constants:
            Debug.vprint("Extracting decay constants from {n}".format(n=expression.name))
            decay_constants = expression.gene_data["decay_constants"].values
        else:
            Debug.vprint("No decay information found. Solving dX/dt = AB for Betas")
            return velocity

        x = np.multiply(expression.values, decay_constants[None, :])
        return InferelatorData(np.add(velocity.values, x), gene_names=expression.gene_names,
                               sample_names=expression.sample_names, meta_data=expression.meta_data)
