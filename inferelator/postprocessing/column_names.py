"""
This is a central location for dataframe column names used in the
postprocessing modules

They're weirdly named for historical reasons and can be changed with
no consequence unless you have other
code that uses these names
"""

BETA_SIGN_COLUMN = "beta.sign.sum"
MEDIAN_EXPLAIN_VAR_COLUMN = "var.exp.median"
CONFIDENCE_COLUMN = "combined_confidences"
BETA_THRESHOLD_COLUMN = "beta_threshold"
GOLD_STANDARD_COLUMN = "gold_standard"
PRIOR_COLUMN = "prior"
TARGET_COLUMN = "target"
REGULATOR_COLUMN = "regulator"

MODEL_COEF_COLUMN = "model_coefficient"
MODEL_EXP_VAR_COLUMN = "model_exp_var"

# Precision/Recall

PRECISION_COLUMN = "precision"
RECALL_COLUMN = "recall"

# Matthew Correlation Coef

MCC_COLUMN = "MCC"

# F1 Score

F1_COLUMN = "F1"

# Confusion Matrix

TP, FP, TN, FN = 'TP', 'FP', 'TN', 'FN'
