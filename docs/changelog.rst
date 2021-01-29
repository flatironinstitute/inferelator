Change Log
==========

Inferelator v0.5.2 `January 29, 2021`
--------------------------------------

New Functionality:

- Added flag ``.set_shuffle_parameters(make_data_noise=True)`` to model on randomly generated noise
- Output TSV files are gzipped by default
- Added ``.set_output_file_names()`` as interface to change output file names
- Added ``.set_regression_parameters(lambda_Bs=None, lambda_Ss=None, heuristic_Cs=None)`` for AMuSR regression

Bug Fixes:

- Fixed bug(s) with dask cluster scaling

Code Refactoring:

- Added additional tests
- Refactored AMuSR code

Inferelator v0.5.1 `November 22, 2020`
--------------------------------------

Bug Fixes:

- Fixed bug that prevented PDF summary figure generation

Inferelator v0.5.0 `November 14, 2020`
--------------------------------------

New Functionality:

- Changed output to include additional performance metrics (Matthews Correlation Coefficient and F1)

Bug Fixes:

- Fixed several bugs around data loading
- Fixed several float tolerance bugs

Code Refactoring:

- Added additional tests
- Improved dask cluster configurations
- Improved documentation

Inferelator v0.4.1 `August 4, 2020`
--------------------------------------

New Functionality:

- Added a regression module based on stability selection
- Added a regression module that can apply any scikit-learn regression model

Bug Fixes:

- Fixed row labels in matrix outputs

Code Refactoring:

- Added additional tests

Inferelator v0.4.0 `April 7, 2020`
--------------------------------------

New Functionality:

- Support for sparse data structures
- Support for h5 and mtx input files
- Added several flags that can change behavior of BBSR (clr_only, ols_only)

Bug Fixes:

- Changed behavior of precision-recall to average the precision of ties instead of randomly ordering

Code Refactoring:

- Refactored the core data structures from pandas to AnnData backed by numpy or scipy arrays
- Data matrices are loaded and maintained as OBS x VAR throughout the workflow.
  Data files which are in GENE x SAMPLE orientation can be loaded if
  ``.set_file_properties(expression_matrix_columns_are_genes=False)`` is set.
- Use sparse_dot_mkl with the intel Math Kernel Library to handle sparse (dot) dense multiplication
- Improved memory usage
- Added unit tests for dask-related functionality
- Changed a number of error messages to improve clarity

Inferelator v0.3.2 `December 19, 2019`
--------------------------------------

New Functionality:

- Improved error messages associated with misaligned data structures
- Added example script and data for the multitask workflows

Bug Fixes:

- Corrected several bugs when using the CrossValidationManager on multitask workflows

Code Refactoring:

- This is the final release which will be fully py2.7 compatible
- Additional unit testing

Inferelator v0.3.1 `December 10, 2019`
--------------------------------------

New Functionality:

- Created a CrossValidationManager which handles parameter searches on workflows.
  Replaces the single_cell_cv_workflow which did not generalize well.
- Workflow parameters are now set through functional setters like set_file_paths(),
  instead of through setting (cryptic) instance variables
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