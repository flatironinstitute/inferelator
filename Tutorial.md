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
Install required libraries for parallelization:
```
python -m pip install -r requirements-multiprocessing.txt
```
(running on a single machine requires only pathos which can be installed with python -m pip install pathos)


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

## Construct a new run script for a different organism

Obtain expression data and save it as a TSV file of [Genes x Samples]
Obtain prior interaction data between TFs and target genes and save it as a TSV file of [Genes x TFs]
