# Inferelator 3.0

[![PyPI version](https://badge.fury.io/py/inferelator.svg)](https://badge.fury.io/py/inferelator)
[![CI](https://github.com/flatironinstitute/inferelator/actions/workflows/python-package.yml/badge.svg)](https://github.com/flatironinstitute/inferelator/actions/workflows/python-package.yml/)
[![codecov](https://codecov.io/gh/flatironinstitute/inferelator/branch/release/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/inferelator)
[![Documentation Status](https://readthedocs.org/projects/inferelator/badge/?version=latest)](https://inferelator.readthedocs.io/en/latest/?badge=latest)

The [Inferelator 3.0](https://doi.org/10.1093/bioinformatics/btac117) is a package for gene regulatory network inference that is based on regularized regression. 
It is an update of the [Inferelator 2.0](https://ieeexplore.ieee.org/document/5334018), which is an update of the original [Inferelator](https://doi.org/10.1186/gb-2006-7-5-r36)
It is maintained by the Bonneau lab in the [Systems Biology group of the Flatiron Institute](https://www.simonsfoundation.org/flatiron/center-for-computational-biology/systems-biology/).

This repository is the actively developed inferelator package for python. It works for both single-cell and bulk transcriptome experiments.
Includes [AMuSR](https://github.com/simonsfoundation/multitask_inferelator/tree/AMuSR/inferelator_ng) 
[(Castro et al 2019)](https://doi.org/10.1371/journal.pcbi.1006591), 
elements of [InfereCLaDR](https://github.com/simonsfoundation/inferelator_ng/tree/InfereCLaDR) 
[(Tchourine et al 2018)](https://doi.org/10.1016/j.celrep.2018.03.048), 
and single-cell workflows [(Jackson et al 2020)](https://elifesciences.org/articles/51254).

We recommend installing this package from PyPi using `python -m pip install inferelator`. 
If running locally, also install `joblib` by `python -m pip install joblib` for parallelization.
If running on a cluster, also install `dask` by `python -m pip install dask[complete] dask_jobqueue` for dask-based parallelization.

This package can also be installed from the github repository. 
Clone the [inferelator GitHub](https://github.com/flatironinstitute/inferelator) repository and run `python setup.py install`.

Documentation is available at [https://inferelator.readthedocs.io](https://inferelator.readthedocs.io/en/latest/), and
basic workflows for ***Bacillus subtilis*** and ***Saccharomyces cerevisiae*** are included with a tutorial. 

All current example data and scripts are available from Zenodo 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3355524.svg)](https://doi.org/10.5281/zenodo.3355524).