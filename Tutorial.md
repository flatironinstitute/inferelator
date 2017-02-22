
This tutorial walks through a gene regulatory network example in yeast. 

## 1) Prepare data
The largest file needed for network inference is the expression dataset, which for yeast is larger than the github file size cut off. As a result, we host the file on dropbox, where it is publicsly accessible. 
`wget https://www.dropbox.com/s/dhj4amz0wcfdn58/expression.tsv data/yeast`

## 2) Install required python libraries
If you have Python 2 >=2.7.9 or Python 3 >=3.4 you should already have pip, the tool for installing python libaries, installed, but if you don't you can download it here: https://packaging.python.org/installing/#install-pip-setuptools-and-wheel. 

To install the packages needed for the inferelator, run:
```
pip install pandas
pip install scipy
pip install matplotlib
```

## 3) Run workflow
`python yeast_bbsr_workflow_runner.py`

## 4) Analyze output
Look in the data/yeast folder for a time-stamped folder that will contain the predicted network as network.tsv and will contain a precision / recall curve as pr_curve.png. The AUPR should be ~0.2.

## Additional config:

The number of bootstraps is currently set to 2, but can be modified via setting variables in yeast_bbsr_workflow_runner.py
