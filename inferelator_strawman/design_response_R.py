"""
Compute design and response by calling R subprocess.
"""

import os

my_dir = os.path.dirname(__file__)

R_dir = os.path.join(my_dir, "R_code")

DR_module = os.path.join(R_dir, "design_and_response.R")

R_template = """
source('{module}')

meta.data <- read.table('{meta_file}', sep = ',', header = 1, row.names = 1)
exp.mat <- read.table('{exp_file}', sep = ',', header = 1, row.names = 1)
delT.min <- {delTmin} 
delT.max <- {delTmax}
tau <- {tau}
dr <- design.and.response(meta.data, exp.mat, delT.min, delT.max, tau)
dr$final_response_matrix
dr$final_design_matrix

write.table(as.matrix(dr$final_response_matrix), 'response.tsv', sep = '\t')
write.table(as.matrix(dr$final_design_matrix), 'design.tsv', sep = '\t')
"""

def save_R_driver(to_filename, delTmin=0, delTmax=110, tau=45,
        meta_file="meta_data.csv", exp_file="exp_mat.csv",
        module=DR_module):
    assert os.path.exists(DR_module), "doesn't exist " + repr(DR_module)
    text = R_template.format(delTmin=delTmin, delTmax=delTmax, tau=tau,
                meta_file=meta_file, exp_file=exp_file, module=module)
    with open(to_filename, "w") as outfile:
        outfile.write(text)
