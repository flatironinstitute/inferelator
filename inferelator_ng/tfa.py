import numpy as np
import pandas as pd

class TFA:

	"""
	TFA calculates transcription factor activity using matrix pseudoinverse

		Parameters
	--------
	prior: pd.dataframe
		binary g by t matrix stating existence of gene-TF interactions. 
		g--gene, t--TF.

	exp.mat: pd.dataframe
		normalized expression matrix.

	exp.mat.halftau: pd.dataframe
		normalized expression matrix for time series.

	noself=True: boolean
		By default, self-regulatory interactions are removed.

	dup_self=True: boolean
		By default, self interactions for TFs that other TFs with the 
		exact same set of interactions in the prior are kept
	"""
	"""
	def __init__(self, prior, exp_mat):
		self.prior = prior
		self.exp_mat = exp_mat
	"""
	def tfa(prior, exp_mat, exp_mat_halftau, noself = True, dup_self = True):
	    tfwt = prior.abs().sum(axis = 0) > 0
	    tfwt = tfwt[tfwt].index.tolist()

	    if dup_self:
	        # subset of prior matrix whose columns are not all zeros (snz: sum non-zero)
	        prior_snz = prior[tfwt]			
	        duplicates = prior_snz.transpose().duplicated(keep=False) # mark duplicates as true
	        dTFs = duplicates[duplicates].index.tolist()

	    # find non-duplicated TFs that are also present in target gene list 
	    ndTFs = list(set(prior.columns.values.tolist()).difference(dTFs))

	    if noself:
	        selfTFs = list(set(ndTFs).intersection(prior.index.values.tolist()))
	        prior.loc[selfTFs, selfTFs] = 0

	    activity = pd.DataFrame(0, index = prior.columns, columns = exp_mat_halftau.columns)

	    if any(tfwt):
	        activity.loc[tfwt,:] = np.matrix(linalg.pinv2(prior_snz))* np.matrix(expr)

	    ntfwt = list(set(prior.columns.values.tolist()).difference(tfwt))
	    activity.loc[ntfwt,:] = expr.loc[ntfwt,:]

	    return activity

			




