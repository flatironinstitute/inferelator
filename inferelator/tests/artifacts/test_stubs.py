from inferelator import amusr_workflow
from inferelator.regression.base_regression import RegressionWorkflow
from inferelator.tests.artifacts.test_data import TestDataSingleCellLike

import pandas as pd
import numpy as np


class TaskDataStub(amusr_workflow.create_task_data_class(workflow_class="single-cell")):
    expression_matrix = TestDataSingleCellLike.expression_matrix
    meta_data = TestDataSingleCellLike.meta_data
    priors_data = TestDataSingleCellLike.priors_data
    gene_metadata = TestDataSingleCellLike.gene_metadata
    gene_list_index = TestDataSingleCellLike.gene_list_index
    tf_names = TestDataSingleCellLike.tf_names

    meta_data_task_column = "Condition"
    tasks_from_metadata = True

    task_name = "TestStub"
    task_workflow_type = "single-cell"

    def get_data(self):
        if self.tasks_from_metadata:
            return self.separate_tasks_by_metadata()
        else:
            return [self]

class FakeDRD:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, expr, meta):
        return expr, expr, expr

class FakeWriter(object):
    def writerow(self, *args, **kwargs):
        pass

class FakeRegression(RegressionWorkflow):

    def run_regression(self):
        beta = [pd.DataFrame(np.array([[0, 1], [0.5, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        beta_resc = [pd.DataFrame(np.array([[0, 1], [1, 0.05]]), index=['gene1', 'gene2'], columns=['tf1', 'tf2'])]
        return beta, beta_resc

    def run_bootstrap(self, bootstrap):
        return True

class FakeResultProcessor:
    network_data = None

    def __init__(self, *args, **kwargs):
        pass

    def summarize_network(self, *args, **kwargs):
        return 1, 0, 0