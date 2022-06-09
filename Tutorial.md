This tutorial is designed to walk through a basic example of network inference in Yeast and the basic mechanism for
constructing an inference workflow for an arbitrary data set

## Set Up the Inferelator from the GitHub repository

Clone the codebase:
```
git clone https://github.com/flatironinstitute/inferelator.git
```
Enter its top-level directory:
```
cd inferelator
```
Install required python libraries:
```
python -m pip install -r requirements.txt
```
Install required libraries for parallelization (running on a single machine requires only `python -m pip install joblib`):
```
python -m pip install -r requirements-multiprocessing.txt
```
Install the inferelator package
```
python setup.py develop --user
```

## Set Up the Inferelator using pip

Install the inferelator
```
python -m pip install inferelator --user
```

## Download the example data files and run scripts from Zenodo

```
wget https://zenodo.org/record/3355524/files/inferelator_example_data.tar.gz
```
Unpack the example data files and run scripts
```
tar -xzf inferelator_example_data.tar.gz
```
Change the value of `DATA_DIR =` in the example scripts to point to the just-created example data paths

## Run network inference on Yeast Microarray data

```
python examples/yeast_network_inference_run_script.py
```

## Acquire necessary data for network inference in a different organism

Obtain expression data and save it as a TSV file of [Samples x Genes] 
(Other file formats are also possible; AnnData h5 `.h5ad` files are recommended for large or sparse data)

Obtain prior interaction data between TFs and target genes and save it as a TSV file of [Genes x TFs]

Create a list of TFs to model for inference and save it as a text file with each TF on a separate line [TFs]

Note that the TF and Gene names must match between files, and these files can be compressed with `gzip` if needed

## Construct a new run script (`new_organism.py`) for a different organism

Select parallelization options:
```
from inferelator.distributed.inferelator_mp import MPControl

# The if __name__ == '__main__' pragma prevents runaway spawning
# when os.fork is not available. 

if __name__ == '__main__':
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = 4
    MPControl.connect()
```
Create an inferelator workflow
```
from inferelator import workflow
worker = workflow.inferelator_workflow(regression="bbsr", workflow="tfa")
```
Set file names and paths:
```
worker.set_file_paths(input_dir='data/new_organism',
                      output_dir='~/new_organism/',
                      tf_names_file='tf_list.txt',
                      priors_file='new_prior.tsv.gz',
                      gold_standard_file='new_prior.tsv.gz')
worker.set_expression_file(tsv='new_expression.tsv.gz')
```
Add a line to execute inference:
```
worker.run()
```
This script can now be run from the command line as `python new_organism.py`
