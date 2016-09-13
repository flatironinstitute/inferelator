"""
Compute design and response by calling R subprocess.
"""

import os
import pandas as pd
from . import utils


DR_module = utils.local_path("R_code", "design_and_response.R")

R_template = r"""
source('{module}')

meta.data <- read.table('{meta_file}', sep = ',', header = 1, row.names = 1, check.names = FALSE)
exp.mat <- read.table('{exp_file}', sep = ',', header = 1, row.names = 1, check.names = FALSE)
delT.min <- {delTmin} 
delT.max <- {delTmax}
tau <- {tau}
dr <- design.and.response(meta.data, exp.mat, delT.min, delT.max, tau)
#dr$final_response_matrix
#dr$final_design_matrix

write.table(as.matrix(dr$final_response_matrix), '{response_file}', sep = '\t')
write.table(as.matrix(dr$final_design_matrix), '{design_file}', sep = '\t')
cat("done. \n")
"""


def save_R_driver(to_filename, delTmin=0, delTmax=110, tau=45,
        meta_file="meta_data.csv", exp_file="exp_mat.csv",
        module=DR_module, response_file='response.tsv', design_file='design.tsv'):
    assert os.path.exists(module), "doesn't exist " + repr(module)
    text = R_template.format(delTmin=delTmin, delTmax=delTmax, tau=tau,
                meta_file=meta_file, exp_file=exp_file, module=module,
                response_file=response_file, design_file=design_file)
    with open(to_filename, "w") as outfile:
        outfile.write(text)
    return (to_filename, design_file, response_file)


class DRDriver(utils.RDriver):

    """
    Configurable container for calling R subprocess to
    compute design and response.
    """

    meta_file = "meta_data.csv"
    exp_file = "exp_mat.csv"
    script_file = "run_design_response.R"
    response_file = "response.tsv"
    design_file = "design.tsv"
    delTmin = 0
    delTmax = 110
    tau = 45

    def run(self, expression_data_frame, metadata_dataframe):
        exp = utils.convert_to_R_df(expression_data_frame)
        md = utils.convert_to_R_df(metadata_dataframe)
        exp.to_csv(self.path(self.exp_file), na_rep='NA')
        md.to_csv(self.path(self.meta_file), na_rep='NA')
        (driver_path, design_path, response_path) = save_R_driver(
            to_filename=self.path(self.script_file),
            delTmin=self.delTmin,
            delTmax=self.delTmax,
            tau=self.tau,
            meta_file=self.path(self.meta_file),
            exp_file=self.path(self.exp_file),
            response_file=self.path(self.response_file),
            design_file=self.path(self.design_file)
        )
        utils.call_R(driver_path)
        final_design = pd.read_csv(design_path, sep='\t')
        final_response = pd.read_csv(response_path, sep='\t')
        return (final_design, final_response)



