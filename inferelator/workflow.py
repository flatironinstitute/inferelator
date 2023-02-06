"""
Construct inferelator workflows from preprocessing, postprocessing, and regression modules
"""

import inspect

from inferelator.utils import is_string
from inferelator.workflows.workflow_base import (
    WorkflowBase,
    WorkflowBaseLoader
)
from inferelator.regression.base_regression import _RegressionWorkflowMixin


def _factory_build_inferelator(
    regression=_RegressionWorkflowMixin,
    workflow=WorkflowBase
):
    """
    This is the factory method to create workflow classes that combine
    preprocessing and postprocessing (from workflow)
    with a regression method (from regression)

    :param regression: A class object which implements the run_regression and
        run_bootstrap methods for a specific regression strategy
    :type regression: RegressionWorkflow
    :param workflow:  A class object which implements the necessary data
        loading and preprocessing to create design & response data for the
        regression strategy, and then the postprocessing to turn regression
        betas into a network
    :type workflow: WorkflowBase
    :return RegressWorkflow: This returns an uninstantiated class which is
        the multi-inheritance result of both the regression workflow and
        the preprocessing/postprocessing workflow
    :rtype: Workflow
    """

    use_mtl_regression = False

    # Decide which preprocessing/postprocessing workflow to use
    # String arguments are parsed for convenience in the run script
    if is_string(workflow):

        workflow = workflow.lower()

        if workflow == "base":

            workflow_class = WorkflowBase

        elif workflow == "tfa":

            from inferelator.workflows.tfa_workflow import (
                TFAWorkFlow
            )

            workflow_class = TFAWorkFlow

        elif workflow == "amusr" or workflow == "multitask":

            from inferelator.workflows.amusr_workflow import (
                MultitaskLearningWorkflow
            )

            workflow_class = MultitaskLearningWorkflow
            use_mtl_regression = True

        elif workflow == "single-cell":

            from inferelator.workflows.single_cell_workflow import (
                SingleCellWorkflow
            )

            workflow_class = SingleCellWorkflow

        elif workflow == "velocity":

            from inferelator.workflows.velocity_workflow import (
                VelocityWorkflow
            )

            workflow_class = VelocityWorkflow

        else:
            raise ValueError(
                f"{workflow} is not a string that can be mapped "
                "to a workflow class"
            )

    # Or just use a workflow class directly
    elif inspect.isclass(workflow) and issubclass(workflow, WorkflowBase):
        workflow_class = workflow
    else:
        raise ValueError(
            "workflow must be a string that maps to a Workflow class "
            "or an actual Workflow class"
        )

    # Decide which regression workflow to use
    # Return just the workflow if regression is set to None
    if regression is None:
        class InferelatorRegressWorkflow(workflow_class):
            regression_type = None

        return InferelatorRegressWorkflow

    # String arguments are parsed for convenience in the run script
    elif is_string(regression):

        regression = regression.lower()

        if regression.endswith("-by-task"):
            use_mtl_regression = True
            regression = regression[:-8]

        if regression == "base":

            regression_class = _RegressionWorkflowMixin

        elif regression == "bbsr":

            from inferelator.regression.bbsr_python import (
                BBSRRegressionWorkflowMixin
            )

            from inferelator.regression.bbsr_multitask import (
                BBSRByTaskRegressionWorkflowMixin
            )

            if use_mtl_regression:
                regression_class = BBSRByTaskRegressionWorkflowMixin
            else:
                regression_class = BBSRRegressionWorkflowMixin

        elif regression == "elasticnet":

            from inferelator.regression.elasticnet_python import (
                ElasticNetWorkflowMixin,
                ElasticNetByTaskRegressionWorkflowMixin
            )

            if use_mtl_regression:
                regression_class = ElasticNetByTaskRegressionWorkflowMixin
            else:
                regression_class = ElasticNetWorkflowMixin

        elif regression == "amusr":

            from inferelator.regression.amusr_regression import (
                AMUSRRegressionWorkflowMixin
            )

            regression_class = AMUSRRegressionWorkflowMixin


        elif regression == "stars":

            from inferelator.regression.stability_selection import (
                StARSWorkflowMixin,
                StARSWorkflowByTaskMixin
            )

            if use_mtl_regression:
                regression_class = StARSWorkflowByTaskMixin
            else:
                regression_class = StARSWorkflowMixin

        elif regression == "sklearn":

            from inferelator.regression.sklearn_regression import (
                SKLearnWorkflowMixin,
                SKLearnByTaskMixin
            )

            if use_mtl_regression:
                regression_class = SKLearnByTaskMixin
            else:
                regression_class = SKLearnWorkflowMixin

        else:
            raise ValueError(
                f"{regression} is not a string that can be mapped to "
                "a regression class"
            )

    # Or just use a regression class directly
    elif (inspect.isclass(regression) and
          issubclass(regression, _RegressionWorkflowMixin)):

        regression_class = regression
    else:
        raise ValueError(
            "Regression must be a string that maps to a regression"
            "class or an actual regression class"
        )

    class InferelatorRegressWorkflow(regression_class, workflow_class):
        regression_type = regression_class

    return InferelatorRegressWorkflow


def inferelator_workflow(
    regression=_RegressionWorkflowMixin,
    workflow=WorkflowBase
):
    """
    Create and instantiate an Inferelator workflow.

    :param regression: A class object which implements the
        run_regression and run_bootstrap methods for a specific
        regression strategy. This can be provided as a string.

        "base" loads a non-functional regression stub.

        "bbsr" loads Bayesian Best Subset Regression.

        "elasticnet" loads Elastic Net Regression.

        "sklearn" loads scikit-learn Regression.

        "stars" loads the StARS stability Regression.

        "amusr" loads AMuSR Regression. This requires multitask workflow.

        Defaults to "base".
    :type regression: str, RegressionWorkflow subclass

    :param workflow: A class object which implements the necessary
        data loading and preprocessing to create design &
        response data for the regression strategy, and then the
        postprocessing to turn regression betas into a network.
        This can be provided as a string.

        "base" loads a non-functional workflow stub.

        "tfa" loads the TFA-based workflow.

        "single-cell" loads the Single Cell TFA-based workflow.

        "multitask" loads the multitask workflow.

        Defaults to "base".
    :type workflow: str, WorkflowBase subclass

    :return: This returns an initialized object which has both the
        regression workflow and the preprocessing/postprocessing workflow.
        This object can then have settings assigned to it, and can be run
        with `.run()`
    :rtype: Workflow instance
    """

    return _factory_build_inferelator(
        regression=regression,
        workflow=workflow
    )()
