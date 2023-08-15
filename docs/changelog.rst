Change Log
==========

Inferelator v0.6.3 `August 15, 2023`
----------------------------------------

New Functionality:

- Accepts h5ad objects directly into the constructor

Bug Fixes:

- Fixed several deprecated arguments in dependencies

Inferelator v0.6.2 `May 8, 2023`
----------------------------------------

New Functionality:

- Generates & reports non-bootstrap model weights as part of results
- Saves full model information into an h5ad file
- Added new experimental prediction modules
- Added new preprocessing & normalization options

Code Refactoring:

- Logging messages now use logging module

Bug Fixes:

- Fixed several errors when sparse data was passed unexpectedly
- Corrected several deprecated numpy calls
- Updated calls and version requirement to anndata


Inferelator v0.6.1 `January 3, 2023`
----------------------------------------

New Functionality:

- Extended support for mRNA velocity & decay calculations
- Added new experimental TFA modules

Code Refactoring:

- Workflow, ResultProcessor, and InferelatorData restructured for readability and clearer commenting

Bug Fixes:

- Slicing individual gene data returns numpy vector instead of anndata view 
- Corrected several deprecated pandas calls to eliminate FutureWarnings

Inferelator v0.6.0 `September  14, 2022`
----------------------------------------

New Functionality:

- Support for grouping arbitrary genes from multiple tasks into learning groups
- Workflow to learn homology groups together
- Workflow to explicitly incorporate velocity and decay into learning
- Added support for batching parallelization calls to reduce overhead when data is relatively small

Code Refactoring:

- Refactored multi-task learning to parameterize tfs and genes for each task
- Refactored parallelization around joblib & dask
- Removed pathos and replaced with joblib
- Optimized StARS-LASSO by replacing standalone LASSO with lasso_path

Bug Fixes:

- Fixed several messages to be more informative
- use_no_prior is appropriately applied in multitask learning

Inferelator v0.5.8 `February  23, 2022`
---------------------------------------

Bug Fixes:

- Corrected combining multi-task gene and tf labels

Inferelator v0.5.7 `September 29, 2021`
---------------------------------------

New Functionality:

- Added support for numba acceleration of AMuSR with ``.set_run_parameters(use_numba=True)`` (PR #46)

Code Refactoring:

- Updated example scripts
- Removed deprecated KVS multiprocessing and associated code

Bug Fixes:

- Gene labels are included as the first column of the produced confidences TSV file by default
- Matplotlib backend selection checks for non-interactive mode

Inferelator v0.5.6 `August 16, 2021`
------------------------------------

New Functionality:

- Added code to randomly generate noise in prior with ``.set_shuffle_parameters(add_prior_noise=None)``
- Added in-workflow benchmarks for CellOracle and pySCENIC
  

Code Refactoring:

- Minor changes to matplotlib interface
- Improved testing for multitask workflows
- Improved error messaging around prior and gold standard
- Switch from Travis.ci to GitHub Actions for continuous integration
  

Inferelator v0.5.5 `April 29, 2021`
-----------------------------------

New Functionality:

- Added ``.set_regression_parameters(tol=None)`` to parameterize tolerances in AMuSR regression

Code Refactoring:

- Profiled and optimized AMuSR code

Inferelator v0.5.4 `April 23, 2021`
-----------------------------------

Bug Fixes:

- Fixed bug in multitask prior processing
- Fixed bug in dask cluster setup
- Suppressed stdout warning when output network MCC is not finite

Inferelator v0.5.3 `March 22, 2021`
--------------------------------------

New Functionality:

- Added the ability to control threads-per-process when using dask

Bug Fixes:

- Fixed bug in result dataframe that failed to create columns in older versions of pandas

Inferelator v0.5.2 `January 29, 2021`
-------------------------------------

New Functionality:

- Added flag ``.set_shuffle_parameters(make_data_noise=True)`` to model on randomly generated noise
- Output TSV files are gzipped by default
- Added ``.set_output_file_names()`` as interface to change output file names
- Added ``.set_regression_parameters(lambda_Bs=None, lambda_Ss=None, heuristic_Cs=None)`` for AMuSR regression

Bug Fixes:

- Fixed bug(s) with dask cluster scaling
- Fixed float precision bug in mutual information

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