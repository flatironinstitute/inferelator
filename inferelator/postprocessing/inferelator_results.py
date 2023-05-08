import pandas as pd
import os
import matplotlib.pyplot as plt
import anndata as ad

from inferelator.utils import (
    Validator as check,
    join_pandas_index,
    align_dataframe_fill
)

from inferelator.preprocessing import PreprocessData

_SERIALIZE_ATTRS = [
    "network",
    "betas",
    "betas_stack",
    "betas_sign",
    "combined_confidences",
    "tasks"
]


class InferelatorResults:
    """
    For network analysis, the results produced in the output_dir
    are sufficient. Model development and comparisons may require
    access to values that are not written to files.

    An InferelatorResults object is returned by the
    ``workflow.run()`` methods

    A list of InferelatorResults objects is returned by the
    ``CrossValidationManager.run()`` methods

    This object allows access to most of the internal values
    created by the inferelator.
    """

    #: Results name, usually set to task name.
    #: Defaults to None.
    name = None

    #: Network dataframe, usually written to network.tsv.gz
    network = None

    #: Fit model coefficients for each model bootstrap.
    #: This is a list of dataframes which are Genes x TFs.
    betas = None

    #: Count of non-zero betas.
    #: This is a dataframe which is Genes x TFs
    betas_stack = None

    #: The aggregate sign of non-zero betas.
    #: This is a dataframe which is Genes x TFs
    betas_sign = None

    #: Confidence scores for tf-gene network edges.
    #: This is a dataframe which is Genes x TFs
    combined_confidences = None

    #: Task result objects if there were multiple tasks.
    #: None if there were not.
    #: This is a dict, keyed by task ID
    tasks = None

    # File names
    network_file_name = "network.tsv.gz"
    confidence_file_name = "combined_confidences.tsv.gz"
    threshold_file_name = "model_coefficients.tsv.gz"
    curve_file_name = "combined_metrics.pdf"
    model_file_name = "inferelator_model.h5ad"
    curve_data_file_name = None

    # Performance metrics
    metric = None
    curve = None
    score = None

    # Performance metrics - all scores
    all_scores = None
    all_names = None

    def __init__(
        self,
        network_data,
        betas_stack,
        combined_confidences,
        metric_object,
        betas_sign=None,
        betas=None,
        priors=None,
        gold_standard=None
    ):
        self.network = network_data
        self.betas_stack = betas_stack
        self.combined_confidences = combined_confidences
        self.metric = metric_object
        self.curve = metric_object.curve_dataframe()
        _, self.score = metric_object.score()
        self.all_names = metric_object.all_names()
        self.all_scores = metric_object.all_scores()
        self.betas_sign = betas_sign
        self.betas = betas
        self.priors = priors
        self.gold_standard = gold_standard

    def new_metric(
        self,
        metric_object,
        curve_file_name=None,
        curve_data_file_name=None
    ):
        """
        Generate a new result object with a new metric
        :param metric_object:
        :param curve_file_name:
        :param curve_data_file_name:
        :return:
        """
        new_result = InferelatorResults(
            self.network,
            self.betas_stack,
            self.combined_confidences,
            metric_object,
            betas_sign=self.betas_sign,
            betas=self.betas
        )

        new_result.curve_data_file_name = curve_data_file_name
        new_result.curve_file_name = curve_file_name

        return new_result

    def plot_other_metric(
        self,
        metric_object,
        output_dir,
        curve_file_name=None,
        curve_data_file_name=None
    ):
        """
        Write just the curve files for another provided metric.

        :param metric_object:
        :param output_dir:
        :param curve_file_name:
        :param curve_data_file_name:
        :return:
        """
        nm = self.new_metric(
            metric_object,
            curve_file_name=curve_file_name,
            curve_data_file_name=curve_data_file_name
        )

        if curve_file_name is not None:
            nm.metric.output_curve_pdf(output_dir, curve_file_name)

        if curve_data_file_name is not None:
            self.write_to_tsv(nm.curve, output_dir, curve_data_file_name)

    def write_result_files(
        self,
        output_dir
    ):
        """
        Write all of the output files. Any individual file output can be
        overridden by setting the file name to None.
        All files can be suppressed by calling output_dir as None

        :param output_dir: Path to a directory where output files should
            be created. If None, no files will be made
        :type output_dir: str, None
        """

        # Validate that the output path exists (create it if necessary)
        check.argument_path(
            output_dir,
            allow_none=True,
            create_if_needed=True
        )

        # Write TSV files
        self.write_to_tsv(
            self.network,
            output_dir,
            self.network_file_name,
            index=False
        )

        self.write_to_tsv(
            self.combined_confidences,
            output_dir,
            self.confidence_file_name,
            index=True
        )

        self.write_to_tsv(
            self.betas_stack,
            output_dir,
            self.threshold_file_name,
            index=True
        )

        self.write_to_tsv(
            self.curve,
            output_dir,
            self.curve_data_file_name,
            index=False
        )

        self.save(
            output_dir,
            self.model_file_name
        )

        # Write Metric Curve PDF
        if self.curve_file_name is not None:
            fig, ax = self.metric.output_curve_pdf(
                output_dir,
                self.curve_file_name
            )

            plt.close(fig)

    def clear_output_file_names(self):
        """
        Reset the output file names.
        Nothing will be output if this is called,
        unless new file names are set
        """

        for a in (
            'network_file_name',
            'confidence_file_name',
            'threshold_file_name',
            'curve_file_name',
            'curve_data_file_name',
            'model_file_name'
        ):
            setattr(self, a, None)

    def save(
        self,
        output_dir,
        output_file_name
    ):
        """
        Save the InferelatorResults to an AnnData format
        H5 file.

        Skip if either output_dir or output_file_name is None

        :param output_dir:
        :param output_file_name:
        """

        if output_dir is None or output_file_name is None:
            return None

        model_adata = self._pack_adata(
            self.betas_stack,
            self.priors,
            self.gold_standard,
            self.network,
            self.all_scores
        )

        if self.tasks is not None:
            model_adata.uns['tasks'] = pd.Series(self.tasks.keys())

            for t in model_adata.uns['tasks']:
                self._pack_adata(
                    self.tasks[t].betas_stack,
                    self.tasks[t].priors,
                    self.tasks[t].gold_standard,
                    self.tasks[t].network,
                    self.tasks[t].all_scores,
                    model_adata=model_adata,
                    prefix=str(t)
                )

        model_adata.write(
            os.path.join(output_dir, output_file_name)
        )

    @staticmethod
    def _pack_adata(
        coefficients,
        priors,
        gold_standard,
        network,
        scores,
        model_adata=None,
        prefix=''
    ):
        """
        Create a new AnnData object or put network information into
        an existing AnnData object

        :param coefficients: Model coefficients
        :type coefficients: pd.DataFrame
        :param priors: Model priors
        :type priors: pd.DataFrame
        :param gold_standard: Model gold standard
        :type gold_standard: pd.DataFrame
        :param network: Long network dataframe
        :type network: pd.DataFrame
        :param scores: Dict of scores, keyed by metric name
        :type scores: dict
        :param model_adata: Existing model object,
            create a new object if None,
            defaults to None
        :type model_adata: ad.AnnData, optional
        :param prefix: String prefix to identify tasks,
            defaults to ''
        :type prefix: str, optional
        :return: AnnData object with inferelator model results
        :rtype: ad.AnnData
        """

        _targets = join_pandas_index(
            *[
                x.index if x is not None else x
                for x in (coefficients, priors, gold_standard)
            ],
            method='union'
        )

        _regulators = join_pandas_index(
            *[
                x.columns if x is not None else x
                for x in (coefficients, priors, gold_standard)
            ],
            method='union'
        )

        # Make sure the prefix has an underscore
        if prefix != '' and not prefix.endswith("_"):
            prefix += '_'

        # Create a model output object
        # this is a genes x TFs AnnData
        if model_adata is None:
            model_adata = ad.AnnData(
                align_dataframe_fill(
                    coefficients,
                    index=_targets,
                    columns=_regulators,
                    fillna=0.0
                )
            )

            lref = model_adata.layers
        else:
            model_adata.uns[prefix + 'model'] = align_dataframe_fill(
                coefficients,
                index=_targets,
                columns=_regulators,
                fillna=0.0
            )

            lref = model_adata.uns

        if priors is not None:
            lref[prefix + 'prior'] = align_dataframe_fill(
                priors,
                index=_targets,
                columns=_regulators,
                fillna=0.0
            ).astype(priors.dtypes.iloc[0])

        if gold_standard is not None:
            lref[prefix + 'gold_standard'] = align_dataframe_fill(
                gold_standard,
                index=_targets,
                columns=_regulators,
                fillna=0.0
            ).astype(gold_standard.dtypes.iloc[0])

        if network is not None:
            model_adata.uns[prefix + 'network'] = network

        if scores is not None:
            model_adata.uns[prefix + 'scoring'] = scores

        model_adata.uns['preprocessing'] = PreprocessData.to_dict()

        return model_adata

    @staticmethod
    def write_to_tsv(
        data_frame,
        output_dir,
        output_file_name,
        index=False,
        float_format='%.6f'
    ):
        """
        Save a DataFrame to a TSV file

        :param data_frame: Data to write
        :type data_frame: pd.DataFrame
        :param output_dir: The path to the output file.
            If None, don't save anything
        :type output_dir: str, None
        :param output_file_name: The output file name.
            If None, don't save anything
        :type output_file_name: str, None
        :param index: Include the index in the output file
        :type index: bool
        :param float_format: Sprintf float format to reformat floats.
            Set to None to disable.
        :type float_format: str, None
        """

        assert check.argument_type(data_frame, pd.DataFrame, allow_none=True)
        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(output_file_name, str, allow_none=True)

        # Write output
        _write_output = all(map(
            lambda x: x is not None,
            (output_dir, output_file_name, data_frame)
        ))

        if _write_output:
            data_frame.to_csv(
                os.path.join(output_dir, output_file_name),
                sep="\t",
                index=index,
                header=True,
                float_format=float_format
            )
