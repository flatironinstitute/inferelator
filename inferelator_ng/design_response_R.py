"""
Compute design and response by calling R subprocess.
"""

import os
import subprocess
import pandas as pd

my_dir = os.path.dirname(__file__).replace('\\', '/')

def my_path(location):
    return os.path.join(my_dir, location).replace('\\', '/')

DR_module = my_path("R_code/design_and_response.R")

R_template = r"""
source('{module}')

meta.data <- read.table('{meta_file}', sep = ',', header = 1, row.names = 1)
exp.mat <- read.table('{exp_file}', sep = ',', header = 1, row.names = 1)
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
    assert os.path.exists(DR_module), "doesn't exist " + repr(DR_module)
    text = R_template.format(delTmin=delTmin, delTmax=delTmax, tau=tau,
                meta_file=meta_file, exp_file=exp_file, module=module,
                response_file=response_file, design_file=design_file)
    with open(to_filename, "w") as outfile:
        outfile.write(text)
    return (to_filename, design_file, response_file)

def convert_to_R_df(df):
    """
    Convert booleans to "TRUE" and "FALSE" so they will be read correctly from CSV
    format by R.
    """
    new_df = pd.DataFrame(df)
    for col in new_df:
        if new_df[col].dtype == 'bool':
            new_df[col] = [str(x).upper() for x in new_df[col]]
    return new_df

class DR_driver:

    """
    Configurable container for calling R subprocess to
    compute design and response.
    """

    target_directory = "/tmp"
    meta_file = "meta_data.csv"
    exp_file = "exp_mat.csv"
    script_file = "run_design_response.R"
    response_file = "response.tsv"
    design_file = "design.tsv"
    delTmin = 0
    delTmax = 110
    tau = 45

    def path(self, filename):
        return os.path.join(self.target_directory, filename).replace('\\', '/')

    def run(self, expression_data_frame, metadata_dataframe):
        exp = convert_to_R_df(expression_data_frame)
        md = convert_to_R_df(metadata_dataframe)
        exp.to_csv(self.path(self.exp_file))
        md.to_csv(self.path(self.meta_file))
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
        #subprocess.call(['R', '-f', driver_path])
        # command = "R -f " + driver_path
        # execu = 'C:\Program Files\R\R-3.2.2\bin\Rscript.exe'
        call_R(driver_path)
        # stdout = subprocess.check_output(command, shell=True)
        # assert stdout.strip().split()[-2:] == [b"done.", b">"], (
        #     "bad stdout tail: " + repr(stdout.strip().split()[-2:])
        # )
        final_design = pd.read_csv(design_path, sep='\t')
        final_response = pd.read_csv(response_path, sep='\t')
        return (final_design, final_response)

def call_R(driver_path):
    if os.name == "posix":
        command = "R -f " + driver_path
        return subprocess.check_output(command, shell=True)
    else:
        theproc = subprocess.Popen(['R', '-f', driver_path])
        return theproc.communicate()
