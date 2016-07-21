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

	dup.self=True: boolean
		By default, self interactions for TFs that other TFs with the 
		exact same set of interactions in the prior are kept
	"""

	def TFA(self,)



