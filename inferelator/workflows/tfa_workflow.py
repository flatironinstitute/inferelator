"""
Implementation for the Transcription Factor Activity (TFA)
based Inferelator workflow.

This workflow also has a design driver which will incorporate
timecourse data.

This is the standard workflow for most applications.
"""

import numpy as np
from inferelator import workflow
from inferelator.preprocessing import design_response_translation
from inferelator.preprocessing.tfa import TFA, NoTFA
from inferelator.utils import (
    InferelatorDataLoader,
    Debug,
    InferelatorData,
    Validator as check
)


class TFAWorkFlow(workflow.WorkflowBase):
    """
    TFAWorkFlow runs the timecourse driver and the TFA driver
    prior to regression.
    """
    # Design/response parameters
    delTmin = 0
    delTmax = 120
    tau = 45

    # Regression data
    # InferelatorData [N x G]
    design = None

    # InferelatorData [N x K]
    response = None
    half_tau_response = None

    # TFA implementation
    tfa_driver = TFA
    _tfa_output_file = None

    # Precalculated TFA
    _tfa_input_file = None
    _tfa_input_file_type = None

    # Design-Response Driver implementation
    drd_driver = design_response_translation.PythonDRDriver

    def set_design_settings(
        self,
        timecourse_response_driver=None,
        delTmin=None,
        delTmax=None,
        tau=None
    ):
        """
        Set the parameters used in the timecourse design-response driver.

        :param timecourse_response_driver: A flag to indicate that the
            timecourse calculations should be performed.
            If set False, no other timecourse settings will have any effect.
            Defaults to True.
        :type timecourse_response_driver: bool
        :param delTmin: The minimum allowed time difference between timepoints
            to model as a time series. Provide in the
            same units as the metadata time column (usually minutes).
            Defaults to 0.
        :type delTmin: int, float
        :param delTmax: The maximum allowed time difference between timepoints
            to model as a time series. Provide in the
            same units as the metadata time column (usually minutes).
            Defaults to 120.
        :type delTmax: int, float
        :param tau: The tau parameter. Provide in the same units as the
            metadata time column (usually minutes).
            Defaults to 45.
        :type tau: int, float
        """

        if timecourse_response_driver is None:
            pass
        elif timecourse_response_driver:
            self.drd_driver = design_response_translation.PythonDRDriver
        else:
            self.drd_driver = None

        self._set_without_warning("delTmin", delTmin)
        self._set_without_warning("delTmax", delTmax)
        self._set_without_warning("tau", tau)

    def set_tfa(
        self,
        tfa_driver=None,
        tfa_output_file=None,
        tfa_input_file=None,
        tfa_input_file_type=None
    ):
        """
        Perform or skip the TFA calculations; by default the design matrix will
        be transcription factor activity. If this is called with
        `tfa_driver = False`, the design matrix will be transcription factor
        expression. It is not necessary to call this function unless setting
        `tfa_driver = False`.

        :param tfa_driver: A flag to indicate that the TFA calculations should
            be performed.
            Defaults to True
        :type tfa_driver: bool
        :param tfa_output_file: A path to a TSV file which will be created
            with the calculated TFAs. Note that this file may contain TF
            expression if the TFA cannot be calculated for that TF.
            If None, no output file will be produced.
            Defaults to None
        :type tfa_output_file: str, optional
        :param tfa_input_file: A path to a TFA file which will be loaded
            and used in place of activity calculations. If set, all TFA-related
            settings will be irrelevant. TSV file MUST be Samples X TFA.
            If None, the inferelator will calculate TFA
            Defaults to None
        :type tfa_output_file: str, optional
        :param tfa_input_file_type: A string which identifies file type.
            Accepts "tsv" and "h5ad".
            If None, assume the file is a TSV
            Defaults to None
        :type tfa_output_file: str, optional
        """

        if tfa_driver is None:
            pass
        elif tfa_driver is True:
            self.tfa_driver = TFA
        elif tfa_driver is False:
            self.tfa_driver = NoTFA
        else:
            self.tfa_driver = tfa_driver

        self._set_with_warning(
            "_tfa_output_file",
            tfa_output_file
        )

        self._set_file_name(
            "_tfa_input_file",
            tfa_input_file
        )

        self._set_without_warning(
            "_tfa_input_file_type",
            tfa_input_file_type
        )

    def run(self):
        """
        Execute workflow, after all configuration.
        """

        # Set the random seed (for bootstrap selection)
        np.random.seed(self.random_seed)

        # Call the startup workflow
        self.startup()

        # Run regression after startup
        betas, rescaled_betas, full_betas, full_rescale = self.run_regression()

        # Write the results out to a file
        return self.emit_results(
            betas,
            rescaled_betas,
            self.gold_standard,
            self.priors_data,
            full_model=full_betas,
            full_exp_var=full_rescale
        )

    def startup_run(self):
        self.get_data()
        self.process_priors_and_gold_standard()

    def startup_finish(self):
        self.align_priors_and_expression()

        self.compute_common_data()
        self.compute_activity()

        # Most operations will be column-wise
        # change sparse type if needed here
        self.response.to_csc()

    def run_regression(self):
        raise NotImplementedError

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """

        # If there is a tfa driver, run it to calculate TFA
        # from the prior & expression data
        if self._tfa_input_file is not None:
            self.load_activity()
        else:
            self._recalculate_design()

        self.half_tau_response = None

        # Save output TFA file
        if self._tfa_output_file is not None:
            self.create_output_dir()
            self.design.to_csv(
                self.output_path(self._tfa_output_file),
                sep="\t"
            )

            Debug.vprint(
                f"Writing TFA {self.design.shape} to {self._tfa_output_file}",
                level=1
            )

    def _recalculate_design(self):
        """
        Use the TFA driver to recalculate the design matrix
        """
        self.design.convert_to_float()
        self.half_tau_response.convert_to_float()
        self.design = self.tfa_driver().compute_transcription_factor_activity(
            self.priors_data,
            self.design,
            self.half_tau_response
        )

        Debug.vprint(
            f"Rebuilt design matrix {self.design.shape} with TF activity",
            level=1
        )

    def load_activity(self, file=None, file_type=None):

        file = self._tfa_input_file if file is None else file
        file_type = self._tfa_input_file_type if file_type is None else file_type

        loader = InferelatorDataLoader(
            input_dir=self.input_dir,
            file_format_settings=self._file_format_settings
        )

        if file_type is None:
            self.design = loader.load_data_tsv(file)
        if file_type.lower() == "h5ad":
            self.design = loader.load_data_h5ad(file)
        else:
            self.design = loader.load_data_tsv(file)

        Debug.vprint(
            f"Loaded {file} as design matrix {self.design.shape}",
            level=1
        )

        self.design.trim_genes(
            remove_constant_genes=False,
            trim_gene_list=self.design.gene_names.intersection(self.tf_names)
        )

        Debug.vprint(
            f"Trimmed to {self.design.shape} for TF activity",
            level=1
        )

        assert check.indexes_align(
            [self.design.sample_names, self.response.sample_names]
        )

    def emit_results(
        self,
        betas,
        rescaled_betas,
        gold_standard,
        priors,
        full_model=None,
        full_exp_var=None
    ):
        """
        Output result report(s) for workflow run.
        """

        self.create_output_dir()

        rp = self._result_processor_driver(
            betas,
            rescaled_betas,
            filter_method=self.gold_standard_filter_method,
            metric=self.metric
        )

        self.results = rp.summarize_network(
            self.output_dir,
            gold_standard,
            priors,
            full_model_betas=full_model,
            full_model_var_exp=full_exp_var
        )

        return self.results

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """

        if self.drd_driver is not None:
            drd = self.drd_driver(
                metadata_handler=self.metadata_handler,
                return_half_tau=True
            )
        else:
            drd = None

        # If there is no design-response driver set,
        # use the expression data for design and response
        # Also do this if there is no usable metadata
        if drd is None or not drd.validate_run(self.data.meta_data):

            self.design = self.data
            self.response = self.data.copy()
            self.half_tau_response = self.data

        # Otherwise calculate the design-response ODE
        # TODO: Rewrite DRD for InferelatorData
        # TODO: This is *horrifying* as is from a memory perspective
        # TODO: Really fix this soon
        else:
            Debug.vprint(
                'Creating design and response matrix from time metadata'
            )

            drd.delTmin, drd.delTmax = self.delTmin, self.delTmax
            drd.tau = self.tau

            design, response, half_tau_response = drd.run(
                self.data.to_df().T,
                self.data.meta_data
            )

            self.design = InferelatorData(design.T)
            self.response = InferelatorData(response.T)
            self.half_tau_response = InferelatorData(half_tau_response.T)

        Debug.vprint(
            f"Constructed design {self.design.shape} and response "
            f"{self.response.shape} matrices", level=1
        )

        self.data = None
