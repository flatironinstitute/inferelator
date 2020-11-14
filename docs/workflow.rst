Workflows
=====================

Workflow Constructor
--------------------
.. automodule:: inferelator.workflow
   :members: inferelator_workflow
   :no-undoc-members:


Common Workflow
---------------------------
.. autoclass:: inferelator.workflow.WorkflowBaseLoader
   :members: set_file_paths, set_expression_file, set_file_properties, set_network_data_flags, set_file_loading_arguments, print_file_loading_arguments, append_to_path
   :no-undoc-members:

.. autoclass:: inferelator.workflow.WorkflowBase
   :members: set_crossvalidation_parameters, set_shuffle_parameters, set_postprocessing_parameters, set_run_parameters, set_output_file_names, run
   :no-undoc-members:

Transcription Factor Activity (TFA) Workflow
--------------------------------------------

.. automodule:: inferelator.tfa_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.tfa_workflow.TFAWorkFlow
   :members: set_design_settings, set_tfa, run
   :no-undoc-members:
   :show-inheritance:

Single-Cell Workflow
--------------------

.. automodule:: inferelator.single_cell_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.single_cell_workflow.SingleCellWorkflow
   :members: set_count_minimum, add_preprocess_step, run
   :no-undoc-members:
   :show-inheritance:

Multi-Task AMuSR Workflow
-------------------------

.. automodule:: inferelator.amusr_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.amusr_workflow.MultitaskLearningWorkflow
   :members: create_task, set_task_filters
   :no-undoc-members:
   :show-inheritance:


Cross-Validation Workflow Wrapper
---------------------------------

.. automodule:: inferelator.crossvalidation_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.crossvalidation_workflow.CrossValidationManager
   :members: __init__, add_gridsearch_parameter, add_grouping_dropout, add_grouping_dropin, add_size_subsampling
   :no-undoc-members:
   :show-inheritance:

