# inferelator 

[![Travis](https://travis-ci.org/flatironinstitute/inferelator.svg?branch=master)](https://travis-ci.org/flatironinstitute/inferelator)
[![codecov](https://codecov.io/gh/flatironinstitute/inferelator/branch/master/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/inferelator)
[![Documentation Status](https://readthedocs.org/projects/inferelator/badge/?version=latest)](https://inferelator.readthedocs.io/en/latest/?badge=latest)

The [inferelator](https://doi.org/10.1186/gb-2006-7-5-r36) is a package for gene regulatory network inference that is based on regularized regression. 
It is maintained by the Bonneau lab in the [Systems Biology group of the Flatiron Institute](https://www.simonsfoundation.org/flatiron/center-for-computational-biology/systems-biology/).

This repository is the actively developed inferelator package for python; it works for both single-cell and bulk transcriptome experiments.
Includes [AMuSR](https://github.com/simonsfoundation/multitask_inferelator/tree/AMuSR/inferelator_ng)  [(Castro et al 2019)](https://doi.org/10.1371/journal.pcbi.1006591)
and elements of [InfereCLaDR](https://github.com/simonsfoundation/inferelator_ng/tree/InfereCLaDR) [(Tchourine et al 2018)](https://doi.org/10.1016/j.celrep.2018.03.048).

To install the python packages needed for the inferelator, run `pip install -r requirements.txt`.
To install the python packages needed for the inferelator multiprocessing functionality, run `pip install -r requirements-multiprocessing.txt`.
To install this package, clone the [inferelator GitHub](https://github.com/flatironinstitute/inferelator) repository and run `python setup.py install`, or run `pip install inferelator`.

Basic workflows for ***Bacillus subtilis*** and ***Saccharomyces cerevisiae*** are included with a tutorial. 

All current example data and scripts are available from Zenodo 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3355524.svg)](https://doi.org/10.5281/zenodo.3355524)
