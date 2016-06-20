source('design_and_response.R')

meta.data <- read.table('meta_data.csv', sep = ',', header = 1, row.names = 1)
exp.mat <- read.table('exp_mat.csv', sep = ',', header = 1, row.names = 1)
delT.min <- 0 
delT.max <- 110
tau <- 45
dr <- design.and.response(meta.data, exp.mat, delT.min, delT.max, tau)
dr$final_response_matrix
dr$final_design_matrix

write.table(as.matrix(dr$final_response_matrix), 'response.tsv', sep = '\t')
write.table(as.matrix(dr$final_design_matrix), 'design.tsv', sep = '\t')

# creates a thing called off. 
# sets all delT larger to max
# logically we don't expect this to filter
