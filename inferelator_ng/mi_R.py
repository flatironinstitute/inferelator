"""
Compute mutual information by calling R subprocess.
"""

import os
import subprocess
import pandas as pd
from . import utils

MI_MODULE = utils.local_path("R_code", "mi_and_clr.R")

R_template = r"""
source('{mi_module}')

X <- read.table('{x_file}', sep = ',', header = 1, row.names = 1)
Y <- read.table('{y_file}', sep = ',', header = 1, row.names = 1)

PARS <- list()
PARS$mi.bins <- {bins}
PARS$cores <- {cores}

# fill mutual information matrices
Ms <- mi(t(Y), t(X), nbins=PARS$mi.bins, cpu.n=PARS$cores)
# write out the initial mi value for diagnostic purposes.
write.table(Ms, '{mi_file}', sep = '\t')
diag(Ms) <- 0
Ms_bg <- mi(t(X), t(X), nbins=PARS$mi.bins, cpu.n=PARS$cores)
diag(Ms_bg) <- 0

# get CLR matrix
clr.mat = mixedCLR(Ms_bg,Ms)
dimnames(clr.mat) <- list(rownames(Y), rownames(X))

write.table(clr.mat, '{matrix_file}', sep = '\t')
cat("done. \n")
"""

def save_mi_driver(to_filename, x_file="X.csv", y_file="Y.csv",
                   bins=10, cores=10, matrix_file="clr_matrix.tsv",
                   mi_file="mi_matrix.tsv"):
    "Write R driver script."
    assert os.path.exists(MI_MODULE), "doesn't exist " + repr(MI_MODULE)
    text = R_template.format(mi_module=MI_MODULE,
                             x_file=x_file, y_file=y_file, bins=bins,
                             cores=cores, matrix_file=matrix_file, mi_file=mi_file)
    with open(to_filename, "w") as outfile:
        outfile.write(text)
    return (to_filename, matrix_file, mi_file)


class MIDriver(utils.RDriver):

    """
    Configurable container for calling R subprocess to
    compute mutual information.
    """

    target_directory = "/tmp"
    x_file = "x.csv"
    y_file = "y.csv"
    script_file = "run_mi.R"
    matrix_file = "clr_matrix.tsv"
    mi_file = "mi_matrix.tsv"
    bins = 10
    cores = 10

    def run(self, x_data_frame, y_dataframe):
        x = utils.convert_to_R_df(x_data_frame)
        y = utils.convert_to_R_df(y_dataframe)
        x_path = self.path(self.x_file)
        y_path = self.path(self.y_file)
        x.to_csv(x_path)
        y.to_csv(y_path)
        (driver_path, matrix_path, mi_path) = save_mi_driver(
            to_filename=self.path(self.script_file),
            x_file=x_path,
            y_file=y_path,
            bins=self.bins, cores=self.cores,
            matrix_file=self.path(self.matrix_file),
            mi_file=self.path(self.mi_file)
        )
        utils.call_R(driver_path)
        matrix_data_frame = pd.read_csv(matrix_path, sep='\t')
        mi_data_frame = pd.read_csv(mi_path, sep='\t')
        return matrix_data_frame, mi_data_frame

