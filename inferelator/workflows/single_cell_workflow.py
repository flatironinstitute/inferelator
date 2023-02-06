"""
Run Single Cell Network Inference. This is the same TFA network inference
with some extra preprocessing functionality.
"""

from inferelator.workflows import tfa_workflow
from inferelator.preprocessing import single_cell
from inferelator import utils

PREPROCESSING_FUNCS = {
    "log2": single_cell.log2_data,
    "ln": single_cell.ln_data,
    "log10": single_cell.log10_data,
    "ftt": single_cell.tf_sqrt_data
}


class SingleCellWorkflow(tfa_workflow.TFAWorkFlow):
    """
    SingleCellWorkflow has some additional preprocessing
    prior to calculating TFA and running regression
    """
    # Single-cell expression data manipulations
    count_minimum = None  # float

    # Preprocessing workflow holder
    preprocessing_workflow = None

    # Do not use a design-response driver
    drd_driver = None

    def set_count_minimum(
        self,
        count_minimum=None
    ):
        """
        Set the minimum count value for each gene
        (averaged over all samples)

        :param count_minimum: The mean expression value which is required
            to retain a gene for modeling. Data that has already been
            normalized should probably be filtered during normalization,
            not now. Defaults to None (disabled).
        :type count_minimum: float
        """

        self.count_minimum = count_minimum

    def add_preprocess_step(self, fun, **kwargs):
        """
        Add a preprocessing step after count filtering but before
        calculating TFA or regression.

        :param fun: Preprocessing function. Can be provided as a string
        or as a function in `preprocessing.single_cell`.

            "log10" will take the log10 of pseudocounts

            "ln" will take the natural log of pseudocounts

            "log2" will take the log2 of pseudocounts

            "fft" will do the Freeman-Tukey transform

        :type fun: str, `preprocessing.single_cell` function
        :param kwargs: Additional arguments to the preprocessing function
        """
        if self.preprocessing_workflow is None:
            self.preprocessing_workflow = []

        if utils.is_string(fun) and fun.lower() in PREPROCESSING_FUNCS:
            self.preprocessing_workflow.append(
                (PREPROCESSING_FUNCS[fun], kwargs)
            )

        elif utils.is_string(fun) and fun.lower() not in PREPROCESSING_FUNCS:
            raise ValueError(
                f"Unable to translate {fun} into a function to call"
            )
        else:
            self.preprocessing_workflow.append(
                (fun, kwargs)
            )

    def startup_finish(self):
        # Preprocess the single-cell data based on the preprocessing
        # steps added to the workflow
        self.data_white_noise()
        self.single_cell_normalize()

        super(SingleCellWorkflow, self).startup_finish()

    def single_cell_normalize(self):
        """
        Single cell normalization. Requires expression_matrix to be
        all numeric, and to be [N x G].

        Executes all preprocessing workflow steps from the
        preprocessing_workflow list that's set by the
        add_preprocess_step() class function
        """

        single_cell.filter_genes_for_count(
            self.data,
            count_minimum=self.count_minimum
        )

        if self.preprocessing_workflow is not None:
            for sc_func, sc_kwargs in self.preprocessing_workflow:
                sc_kwargs['random_seed'] = self.random_seed
                sc_func(self.data, **sc_kwargs)

        num_nonfinite, name_nonfinite = self.data.non_finite

        if num_nonfinite > 0:
            utils.Debug.vprint(
                "These genes have non-finite values: "
                " ".join(name_nonfinite),
                level=0
            )
            raise ValueError(
                "NaN values have been introduced into the "
                "expression matrix by normalization"
            )
