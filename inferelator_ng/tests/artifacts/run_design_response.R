
source('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/R_code/design_and_response.R')

meta.data <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/meta_data.csv', sep = ',', header = 1, row.names = 1)
exp.mat <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/exp_mat.csv', sep = ',', header = 1, row.names = 1)
delT.min <- 2 
delT.max <- 4
tau <- 2
dr <- design.and.response(meta.data, exp.mat, delT.min, delT.max, tau)
#dr$final_response_matrix
#dr$final_design_matrix

write.table(as.matrix(dr$final_response_matrix), '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/response.tsv', sep = '\t')
write.table(as.matrix(dr$final_design_matrix), '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/design.tsv', sep = '\t')
cat("done. \n")
