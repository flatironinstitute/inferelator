import pandas as pd
import os

from inferelator.utils import Validator as check


class InferelatorResults(object):
    """
    For network analysis, the results produced in the output_dir are sufficient.
    Model development and comparisons may require to values that are not written to files.
    An InferelatorResults object is returned by the workflow.run() methods
    (A list of InferelatorResults objects is returned by the CrossValidationManager.run() method).

    This object allows access to most of the internal values created by the inferelator.
    """

    #: Results name, usually set to task name.
    #: Defaults to None.
    name = None

    #: Network dataframe, usually written to network.tsv
    network = None

    #: Fit model coefficients for each model bootstrap.
    #: This is a list of dataframes which are Genes x TFs.
    betas = None

    #: Count of non-zero betas, usually written to betas_stack.tsv
    #: This is a dataframe which is Genes x TFs
    betas_stack = None

    #: The aggregate sign of non-zero betas.
    #: This is a dataframe which is Genes x TFs
    betas_sign = None

    #: Confidence scores for tf-gene network edges.
    #: This is a dataframe which is Genes x TFs
    combined_confidences = None

    #: Task result objects if there were multiple tasks. None if there were not.
    #: This is a dict, keyed by task ID
    tasks = None

    # File names
    network_file_name = "network.tsv"
    confidence_file_name = "combined_confidences.tsv"
    threshold_file_name = None
    curve_file_name = "combined_metrics.pdf"
    curve_data_file_name = None

    # Performance metrics
    metric = None
    curve = None
    score = None

    # Performance metrics - all scores
    all_scores = None
    all_names = None

    def __init__(self, network_data, betas_stack, combined_confidences, metric_object, betas_sign=None, betas=None):
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

    def new_metric(self, metric_object, curve_file_name=None, curve_data_file_name=None):
        """
        Generate a new result object with a new metric
        :param metric_object:
        :param curve_file_name:
        :param curve_data_file_name:
        :return:
        """
        new_result = InferelatorResults(self.network, self.betas_stack, self.combined_confidences, metric_object,
                                        betas_sign=self.betas_sign, betas=self.betas)

        new_result.curve_data_file_name = curve_data_file_name
        new_result.curve_file_name = curve_file_name

        return new_result

    def plot_other_metric(self, metric_object, output_dir, curve_file_name=None, curve_data_file_name=None):
        """
        Write just the curve files for another provided metric.

        :param metric_object:
        :param output_dir:
        :param curve_file_name:
        :param curve_data_file_name:
        :return:
        """
        nm = self.new_metric(metric_object, curve_file_name=curve_file_name, curve_data_file_name=curve_data_file_name)
        nm.metric.output_curve_pdf(output_dir, curve_file_name) if curve_file_name is not None else None
        self.write_to_tsv(nm.curve, output_dir, curve_data_file_name) if curve_data_file_name is not None else None

    def write_result_files(self, output_dir):
        """
        Write all of the output files. Any individual file output can be overridden by setting the file name to None.
        All files can be suppressed by calling output_dir as None

        :param output_dir: Path to a directory where output files should be created. If None, no files will be made
        :type output_dir: str, None
        """

        # Validate that the output path exists (create it if necessary)
        check.argument_path(output_dir, allow_none=True, create_if_needed=True)

        # Write TSV files
        self.write_to_tsv(self.network, output_dir, self.network_file_name, index=False)
        self.write_to_tsv(self.combined_confidences, output_dir, self.confidence_file_name)
        self.write_to_tsv(self.betas_stack, output_dir, self.threshold_file_name, index=False)
        self.write_to_tsv(self.curve, output_dir, self.curve_data_file_name, index=False)

        # Write Metric Curve PDF
        self.metric.output_curve_pdf(output_dir, self.curve_file_name) if self.curve_file_name is None else None

    def clear_output_file_names(self):
        """
        Reset the output file names (nothing will be output if this is called, unless new file names are set)
        """

        self.network_file_name, self.confidence_file_name, self.threshold_file_name = None, None, None
        self.curve_file_name, self.curve_data_file_name = None, None

    @staticmethod
    def write_to_tsv(data_frame, output_dir, output_file_name, index=False, float_format='%.6f'):
        """
        Save a DataFrame to a TSV file
        :param data_frame: pd.DataFrame
            Data to write
        :param output_dir: str
            The path to the output file. If None, don't save anything
        :param output_file_name: str
            The output file name. If None, don't save anything
        :param index: bool
            Include the index in the output file
        :param float_format: str
            Reformat floats. Set to None to disable.
        """

        assert check.argument_type(data_frame, pd.DataFrame, allow_none=True)
        assert check.argument_path(output_dir, allow_none=True)
        assert check.argument_type(output_file_name, str, allow_none=True)

        # Write output
        if output_dir is not None and output_file_name is not None and data_frame is not None:
            data_frame.to_csv(os.path.join(output_dir, output_file_name), sep="\t", index=index, header=True,
                              float_format=float_format)
