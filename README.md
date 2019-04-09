# inferelator_sc

[![Travis](https://travis-ci.org/flatironinstitute/inferelator.svg?branch=master)](https://travis-ci.org/flatironinstitute/inferelator)
[![codecov](https://codecov.io/gh/flatironinstitute/inferelator/branch/master/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/inferelator)

The inferelator codebase for analyzing data one cell at a time. This is a fork of the currently-inactive [inferelator_ng](https://github.com/simonsfoundation/inferelator_ng) repository. Currently includes [AMuSR](https://github.com/simonsfoundation/multitask_inferelator/tree/AMuSR/inferelator_ng) [(Castro et al 2019)](https://doi.org/10.1371/journal.pcbi.1006591) and elements of [InfereCLaDR](https://github.com/simonsfoundation/inferelator_ng/tree/InfereCLaDR) [(Tchourine et al 2018)](https://doi.org/10.1016/j.celrep.2018.03.048).

To install the python packages needed for the inferelator, run `pip install -r requirements.txt`. To install the python packages needed for the inferelator multiprocessing functionality, run `pip install -r requirements-multiprocessing.txt`. To install this package, run `python setup.py install`

Basic workflows for Bacillus subtilis and Saccharomyces cerevisiae are included with a tutorial. 
