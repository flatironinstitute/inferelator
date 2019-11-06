from __future__ import print_function

import os
import csv
import types
import copy
import itertools

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import Validator as check
from inferelator import utils, default
from inferelator.regression import base_regression
from inferelator.postprocessing import results_processor
from inferelator import workflow


class CrossvalidationManager(object):
    """
    This does cross-validation for workflows
    """

    output_dir = None
    baseline_workflow = None

    # Output settings
    write_network = True  # bool
    csv_writer = None  # csv.csvwriter
    csv_header = ()  # list[]
    output_file_name = "aupr.tsv"  # str

    # Grid search parameters
    grid_params = None
    grid_param_values = None

    # Dropin/dropout categories
    dropout_column = None
    grouping_column = None

    # Size subsampling categories
    size_sample_vector = None
    size_sample_stratified_column = None

    def __init__(self, workflow_object):

        assert check.argument_is_subclass(workflow_object, workflow.WorkflowBase)

        self.baseline_workflow = workflow_object

    def add_gridsearch_parameter(self, param_name, param_vector):
        """
        Set a parameter to search through by exhaustive grid search

        :param param_name: str
            The workflow parameter to change for each run
        :param param_vector: list, tuple
            An iterable with values to use for the parameter
        """

        if self.grid_param_values is None:
            self.grid_param_values = {}

        if self.grid_params is None:
            self.grid_params = []

        self.grid_params.append(param_name)
        self.grid_param_values[param_name] = param_vector

    def add_grouping_dropout(self, metadata_column_name):
        """
        Drop each group (defined by a metadata column) and run modeling on all of the other groups

        :param metadata_column_name: str
        :return:
        """

    def add_grouping_separately(self, metadata_column_name):
        """

        :param metadata_column_name: str
        :return:
        """

    def add_size_subsampling(self, size_vector, stratified_column_name=None):
        """

        :param size_vector: list, tuple
        :param stratified_column_name:
        """

    def run(self):
        if self.output_dir is None:
            self.output_dir = self.baseline_workflow.output_dir
        self._create_writer()
        self._initial_data_load()


    def _create_writer(self):
        """
        Create a CSVWriter and stash it in self.writer
        """

        if MPControl.is_master:

            # Create a CSV header from grid search param names
            self.csv_header = copy.copy(self.grid_params) if self.grid_params is not None else []

            # Add Test & Value columns for dropouts/etc
            self.csv_header.extend(["Test", "Value"])

            # Also add the metric
            self.csv_header.append(self.baseline_workflow.metric)

            # Create a CSV writer
            self._create_output_path()
            self.csv_writer = csv.writer(open(os.path.expanduser(os.path.join(self.output_dir, self.output_file_name)),
                                              mode="w", buffering=1), delimiter="\t", lineterminator="\n",
                                         quoting=csv.QUOTE_NONE)

            # Write the header line
            self.csv_writer.writerow(self.csv_header)

    def _create_output_path(self):
        """
        Create the output path
        """
        if self.output_dir is None:
            raise ValueError("No output path has been provided")
        self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))

        try:
            os.makedirs(self.output_dir)
        except FileExistsError:
            pass

    def _initial_data_load(self):
        """
        Load data into the workflow
        """

        # Load data with the baseline get_data() function
        self.baseline_workflow.get_data()

        # Blow up the get_data() function so that it doesn't get re-run
        def mock_get_data(slf):
            pass

        self.baseline_workflow.get_data = types.MethodType(mock_get_data, self.baseline_workflow)

    def _get_workflow_copy(self):
        """
        Copies and returns the workflow which has loaded data
        """

        return copy.deepcopy(self.baseline_workflow)

    def _grid_search(self, test=None, value=None):

        # This is unpacked in the same order that is used in the header
        ordered_unpack = [self.grid_param_values[param] for param in self.grid_params]

        for param_values in itertools.product(*ordered_unpack):
            params = zip(self.grid_params, param_values)
            csv_line = []

            # Get the workflow and set the CV parameters
            cv_workflow = self._get_workflow_copy()
            for name, param_value in params:
                csv_line.append(param_value)
                setattr(cv_workflow, name, param_value)

            # Set the parameters into the output path
            cv_workflow.append_to_path("_".join(map(str, csv_line)))

            # Run the workflow
            result = cv_workflow.run()
            csv_line.append(test)
            csv_line.append(value)
            csv_line.append(result.score)

            del cv_workflow




