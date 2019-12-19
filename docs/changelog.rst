Change Log
==========

Inferelator v0.3.2
------------------

New Functionality:

- Improved error messages associated with misaligned data structures

Bug Fixes:

- Corrected several bugs when using the CrossValidationManager on multitask workflows

Code Refactoring:

- This is the final release which will be fully py2.7 compatible
- Additional unit testing

Inferelator v0.3.1 `December 10, 2019`
--------------------------------------

New Functionality:

- Created a CrossValidationManager which handles parameter searches on workflows.
  Replaces the single_cell_cv_workflow which was did not generalize well.
- Workflow parameters are now set through functional setters like set_file_paths(),
  instead of through setting instance variables
- Calculated transcription factor activities can be saved to a file prior to inference.
  This is set with `workflow.set_tfa(tfa_output_file = "Filename.tsv")`

Bug Fixes:

- Many

Code Refactoring:

- Rebuilt the multitask workflow with TaskData objects instead managing data in many lists of things.

Inferelator v0.3.0 `July 30, 2019`
----------------------------------

New Functionality:

- Created a MultiprocessingManger for abstract control of multiprocessing.
- Implemented a scheduler-worker model through the dask package for cluster computing.
- Implemented a map model through the pathos implementation of multiprocessing for local computing.
- Example scripts and datasets are now provided

Bug Fixes:

- Many

Code Refactoring:

- Rebuilt the core workflow
- Workflow assembly by inheritance is managed with a factory function
- Refactored regression to act as a mapped function for easier multiprocessing