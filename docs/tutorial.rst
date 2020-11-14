********************
Inferelator Tutorial
********************

Input Data
==========

All data provided to the inferelator should be in TSV format.

The inferelator package **requires** two data structures to function:

- A gene expression matrix which contains some expression data for *G* genes and *N* samples.
  Any unit is generally acceptable provided all samples are the same unit and are reasonably normalized together.
- A text list of *K* genes which should be modeled as regulators (like Transcription Factors)

The performance with no additional data is extremely poor, however.
In addition to the two required data elements, there is other data which can be provided.

The most important of these additional elements is some known knowledge about regulatory connections.

- A prior knowledge connectivity matrix *[G x K]* which links the genes *G* to the regulators *K*.
  This matrix should have a zero where a gene is not regulated by a regulator.
  It should have a non-zero value where a gene is known to be regulated by a regulator.
  This can be as simple as a boolean matrix, but sign and magnitude will affect calculation of regulator activity.
- A gold standard connectivity matrix *[G x K]* which links the genes *G* to the regulators *K*.
  This matrix should have a zero where a gene is not regulated by a regulator.
  It should have a non-zero value where a gene is known to be regulated by a regulator.
  It will be interpreted as a boolean matrix, so sign and magnitude of non-zeros is not considered.

Also important is sample metadata. This is necessary if there is a time element to the samples,
or if there is some grouping criteria. If time series data is included, there are two supported formats for this data.
If time series data is not included, any metadata structure is valid.

The first format is **branching** and has 5 columns::

   isTs | is1stLast | prevCol | del.t | condName
   =============================================
   TRUE |     f     |   NA    |   NA  |   A-1
   TRUE |     m     |   A-1   |   15  |   A-2
   TRUE |     l     |   A-2   |   15  |   A-3


- **isTs** is TRUE or FALSE and indicates if this sample is in a time series.
- **is1stLast** is **f** if this sample is the first sample in a time series.
  It is **m** if this sample is a middle sample in a time series.
  It is **l** if this sample is the last sample in a time series.
  It is **NA** if this sample is not in a time series
- **prevCol** is the name of the sample which comes before this sample
- **del.t** is the time elapsed since the sample which comes before this sample
- **condName** is the name of this sample. It must match the sample name in the expression data.

The second format is **nonbranching** and has 3 columns::

   condName | strain | time
   ========================
      A-1   |    A   |  0
      A-2   |    A   |  15
      A-3   |    A   |  30

- **condName** is the name of this sample. It must match the sample name in the expression data.
- **strain** is the name of the sample group.
- **time** is the absolute time elapsed during this sample group's experiment.

Finally, gene metadata can also be provided. This is currently used to restrict modeling to just some genes.

Workflow setup
==============

The inferelator is implemented on a workflow model. The first step is to create a workflow object.
At this stage, the type of regression model and workflow must be chosen::

   from inferelator import inferelator_workflow

   worker = inferelator_workflow(regression="bbsr", workflow="tfa")

- Valid options for regression include "bbsr", "elastic-net", and "amusr".
- Valid options for workflow include "tfa", "single-cell", and "multitask".

The next step is to set the location of input data files::

   worker.set_file_paths(input_dir=".",
                         output_dir="./output_inferelator",
                         expression_matrix_file="expression.tsv",
                         tf_names_file="regulators.tsv",
                         meta_data_file="meta_data.tsv",
                         priors_file="priors.tsv",
                         gold_standard_file="gold_standard.tsv")

The input directory will be added to all file locations which are not absolute paths.
The output directory will be created if it does not exist.

Finally, run parameters should be set::

   worker.set_run_parameters(num_bootstraps=5, random_seed=42)

This worker can now be run with::

   network_result = worker.run()

Multitask Workflows
===================

The inferelator supports inferring networks from multiple separate "tasks" at the same time.
Several modeling options exist, but all must use the multitask workflow::

  worker = inferelator_workflow(regression="amusr", workflow="multitask")

- **amusr** regression is a multitask learning model that shares information during regression.
- **bbsr-by-task** regression learns separate networks using the BBSR model,
  and then aggregates them into a joint network.
- **elasticnet-by-task** regression learns separate networks using the Elastic Net model,
  and then aggregates them into a joint network.

After creating a workflow, only the input, output and gold standard file location should be provided directly::

  worker.set_file_paths(input_dir=".", output_dir="./output_network", gold_standard_file="gold_standard.tsv.gz")

Other information should be provided to each separate task.
These can be created by calling the ``.create_task()`` function.
This function returns a task reference which can be used to set additional task properties::

  task_1 = worker.create_task(task_name="Bsubtilis_1",
                              input_dir=".",
                              tf_names_file='tf_names.tsv',
                              meta_data_file='GSE67023_meta_data.tsv',
                              priors_file='gold_standard.tsv.gz',
                              workflow_type="tfa")
  task_1.set_expression_file(tsv='GSE67023_expression.tsv.gz')

  task_2 = worker.create_task(task_name="Bsubtilis_2",
                              input_dir=".",
                              tf_names_file='tf_names.tsv',
                              meta_data_file='meta_data.tsv',
                              priors_file='gold_standard.tsv.gz',
                              workflow_type="tfa")
  task_2.set_expression_file(tsv='expression.tsv.gz')


Additional parameters can be set on the main workflow.
Task references made with ``.create_task()`` are automatically included when the workflow is started.
The workflow can then be started with ``.run()``::

  worker.set_run_parameters(num_bootstraps=5, random_seed=42)
  worker.run()

Parallelization
===============

The inferelator supports three major parallelization options. These can be set using a controller class.
Calling the multiprocessing environment should be protected with the ``if __name__ == '__main__'`` pragma.
This is necessary to prevent a specific error in creating new processes that occurs when ``os.fork()`` is unavailable.
Multiprocessing options should be set prior to creating and running workflows.
It is not necessary to set multiprocessing more then once per session::

    from inferelator import MPControl

    if __name__ == '__main__':
        MPControl.set_multiprocess_engine("multiprocessing")
        MPControl.client.processes = 12
        MPControl.connect()

- **multiprocessing** engine uses the pathos implementation of python's multiprocessing.
  It creates multiple processes on one computer.
- **local** engine uses no multiprocessing and runs from a single process.
  In some cases, python libraries (like numpy) may use multiple threads within this process.
- **dask-cluster** engine uses the dask scheduler-worker library in combination with the dask_jobqueue
  cluster-management library to manage processes through a job scheduler. Currently, only SLURM is supported.
  Correctly configuring this for your cluster may be a challenge.