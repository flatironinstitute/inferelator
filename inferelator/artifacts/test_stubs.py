from inferelator import amusr_workflow
from inferelator.regression.base_regression import RegressionWorkflow
from inferelator.artifacts.test_data import TestDataSingleCellLike


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


class FakeRegression(RegressionWorkflow):

    def run_bootstrap(self, bootstrap):
        return True


class FakeResultProcessor:

    network_data = None

    def __init__(self, *args, **kwargs):
        pass

    def summarize_network(self, *args, **kwargs):
        pass