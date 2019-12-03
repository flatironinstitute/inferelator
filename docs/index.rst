inferelator: Regularized regression for Gene Regulatory Network inference
=========================================================================

inferelator.workflow module
---------------------------
.. automodule:: inferelator.workflow
   :members: inferelator_workflow
   :no-undoc-members:

.. autoclass:: inferelator.workflow.WorkflowBaseLoader
   :members: set_file_paths, set_file_properties, set_network_data_flags, set_file_loading_arguments, print_file_loading_arguments, append_to_path
   :no-undoc-members:
   :show-inheritance:

.. autoclass:: inferelator.workflow.WorkflowBase
   :members: set_crossvalidation_parameters, set_shuffle_parameters, set_postprocessing_parameters, set_run_parameters, run
   :no-undoc-members:
   :show-inheritance:

inferelator.tfa\_workflow module
--------------------------------

.. automodule:: inferelator.tfa_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.tfa_workflow.TFAWorkFlow
   :members: set_design_settings, set_tfa, run
   :no-undoc-members:
   :show-inheritance:

inferelator.single\_cell\_workflow module
-----------------------------------------

.. automodule:: inferelator.single_cell_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.single_cell_workflow.SingleCellWorkflow
   :members: set_count_minimum, add_preprocess_step, run
   :no-undoc-members:
   :show-inheritance:

inferelator.amusr\_workflow module
----------------------------------

.. automodule:: inferelator.amusr_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.amusr_workflow.MultitaskLearningWorkflow
   :members: create_task, set_task_filters
   :no-undoc-members:
   :show-inheritance:


inferelator.crossvalidation\_workflow module
--------------------------------------------

.. automodule:: inferelator.crossvalidation_workflow
   :no-members:
   :no-undoc-members:

.. autoclass:: inferelator.crossvalidation_workflow.CrossValidationManager
   :members: __init__, add_gridsearch_parameter, add_grouping_dropout, add_grouping_dropin, add_size_subsampling
   :no-undoc-members:
   :show-inheritance:

