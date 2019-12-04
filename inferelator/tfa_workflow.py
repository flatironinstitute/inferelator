"""
Implementation for the Transcription Factor Activity (TFA) based Inferelator workflow.
This workflow also has a design driver which will incorporate timecourse data.
This is the standard workflow for most applications.
"""

import numpy as np
from inferelator import workflow
from inferelator.preprocessing import design_response_translation  # added python design_response
from inferelator.preprocessing.tfa import TFA, NoTFA
from inferelator import utils


class TFAWorkFlow(workflow.WorkflowBase):
    """
    TFAWorkFlow runs the timecourse driver and the TFA driver prior to regression.
    """
    # Design/response parameters
    delTmin = 0
    delTmax = 120
    tau = 45

    # Regression data
    design = None
    response = None
    half_tau_response = None

    # TFA implementation
    tfa_driver = TFA
    _tfa_output_file = None

    # Design-Response Driver implementation
    drd_driver = design_response_translation.PythonDRDriver

    def set_design_settings(self, timecourse_response_driver=True, delTmin=None, delTmax=None, tau=None):
        """
        Set the parameters used in the timecourse design-response driver.

        :param timecourse_response_driver: A flag to indicate that the timecourse calculations should be performed.
            If set False, no other timecourse settings will have any effect.
            Defaults to True.
        :type timecourse_response_driver: bool
        :param delTmin: The minimum allowed time difference between timepoints to model as a time series. Provide in the
            same units as the metadata time column (usually minutes).
            Defaults to 0.
        :type delTmin: int, float
        :param delTmax: The maximum allowed time difference between timepoints to model as a time series. Provide in the
            same units as the metadata time column (usually minutes).
            Defaults to 120.
        :type delTmax: int, float
        :param tau: The tau parameter. Provide in the same units as the metadata time column (usually minutes).
            Defaults to 45.
        :type tau: int, float
        """

        if timecourse_response_driver:
            self.drd_driver = design_response_translation.PythonDRDriver
        else:
            self.drd_driver = None

        self._set_without_warning("delTmin", delTmin)
        self._set_without_warning("delTmax", delTmax)
        self._set_without_warning("tau", tau)

    def set_tfa(self, tfa_driver=True, tfa_output_file=None):
        """
        Perform or skip the TFA calculations; by default the design matrix will be transcription factor activity.
        If this is called with `tfa_driver = False`, the design matrix will be transcription factor expression.
        It is not necessary to call this function unless setting `tfa_driver = False`.

        :param tfa_driver: A flag to indicate that the TFA calculations should be performed.
            Defaults to True
        :type tfa_driver: bool
        :param tfa_output_file: A path to a TSV file which will be created with the calculated TFAs. Note that this file
            may contain TF expression if the TFA cannot be calculated for that TF.
            If None, no output file will be produced.
            Defaults to None
        :type tfa_output_file: str, optional
        """

        if tfa_driver:
            self.tfa_driver = TFA
        else:
            self.tfa_driver = NoTFA

        self._set_with_warning("_tfa_output_file", tfa_output_file)

    def run(self):
        """
        Execute workflow, after all configuration.
        """

        # Set the random seed (for bootstrap selection)
        np.random.seed(self.random_seed)

        # Call the startup workflow
        self.startup()

        # Run regression after startup
        betas, rescaled_betas = self.run_regression()

        # Write the results out to a file
        return self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)

    def startup_run(self):
        self.get_data()
        self.process_priors_and_gold_standard()

    def startup_finish(self):
        self.align_priors_and_expression()
        self.compute_common_data()
        self.compute_activity()

    def run_regression(self):
        raise NotImplementedError

    def run_bootstrap(self, bootstrap):
        raise NotImplementedError

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        # If there is a tfa driver, run it to calculate TFA from the prior & expression data
        utils.Debug.vprint('Computing Transcription Factor Activity ... ')
        tfa_calculator = self.tfa_driver(self.priors_data, self.design, self.half_tau_response)
        self.design = tfa_calculator.compute_transcription_factor_activity()
        self.half_tau_response = None

        if self._tfa_output_file is not None and self.is_master():
            self.create_output_dir()
            self.design.to_csv(self.output_path(self._tfa_output_file), sep="\t")


        utils.Debug.vprint("Rebuilt design matrix {d} with TF activity".format(d=self.design.shape), level=1)

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        if self.is_master():
            self.create_output_dir()
            rp = self._result_processor_driver(betas, rescaled_betas, filter_method=self.gold_standard_filter_method,
                                               metric=self.metric)
            self.results = rp.summarize_network(self.output_dir, gold_standard, priors)
            return self.results
        else:
            return None

    def compute_common_data(self):
        """
        Compute common data structures like design and response matrices.
        """

        if self.drd_driver is not None:
            # If there is a design-response driver, run it to create design and response
            drd = self.drd_driver(metadata_handler=self.metadata_handler, return_half_tau=True)
            utils.Debug.vprint('Creating design and response matrix ... ')
            drd.delTmin, drd.delTmax, drd.tau = self.delTmin, self.delTmax, self.tau
            self.design, self.response, self.half_tau_response = drd.run(self.expression_matrix, self.meta_data)
        else:
            # If there is no design-response driver set, use the expression data for design and response
            self.design = self.expression_matrix
            self.response, self.half_tau_response = self.expression_matrix, self.expression_matrix

        utils.Debug.vprint("Constructed design {d} and response {r} matrices".format(d=self.design.shape,
                                                                                     r=self.response.shape),
                           level=1)

        self.expression_matrix = None
