
This tutorial walks through a gene regulatory network example in yeast. 

## 0) Get Code

Clone the codebase:
```
git clone git@github.com:simonsfoundation/inferelator_ng.git
```

Enter its top-level directory:
```
cd inferelator_ng
```

## 1) Prepare data
The largest file needed for network inference is the expression dataset, which for yeast is larger than the github file size cut off. As a result, we host the file on dropbox, where it is publicly accessible. 
```
wget https://www.dropbox.com/s/dhj4amz0wcfdn58/expression.tsv -P data/yeast
```

## 2) Install required python libraries
If you have Python 2 >=2.7.9 or Python 3 >=3.4 you should already have pip, the tool for installing python libaries, installed, but if you don't you can download it here: https://packaging.python.org/installing/#install-pip-setuptools-and-wheel. 

To install the python packages needed for the inferelator, run:
```
pip install -r requirements.txt
```

If you do not have R install, you can download it from https://www.r-project.org/

To install the required R packages, run:
```
R -f inferelator_ng/R_code/packages.R
```

## 3) Run workflow
`bash inferelator_runner_slurmless.sh yeast_bbsr_workflow_runner.py`

## 4) Analyze output
Look in the data/yeast folder for a time-stamped folder that will contain the predicted network as network.tsv and will contain a precision-recall curve as pr_curve.png. The AUPR should be ~0.2.

## Additional Config:
The number of bootstraps is currently set to 2, but can be modified via setting variables in yeast_bbsr_workflow_runner.py. 

## Additional Information:
The whole genome expression.tsv was downloaded from the Gene Expression Omnibus (GEO), filtering for samples from Saccharomyces cerevisiae using the Yeast Affymetrix 2.0 platform (accessioning number GPL2529). The prior data file, yeast-motif-prior.tsv, is derived from ATAC-Seq peaks with motif analysis, using data from GEO (GSE66386), and signed (+/- 1) using knock-out results (GSE42527). The gold standard, gold_standard.tsv, is primarily from the YEASTRACT repository. 
