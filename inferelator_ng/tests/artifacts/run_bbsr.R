
library('Matrix')
if ('parallel' %in% installed.packages()[, 'Package']) {
  library('parallel')
} else {
  library('multicore')
}

source('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/R_code/bayesianRegression.R')

X <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/X.csv', sep = ',', header = 1, row.names = 1)
Y <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/Y.csv', sep = ',', header = 1, row.names = 1)
clr.mat <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/clr_matrix.csv', sep = ',', header = 1, row.names = 1)
prior.mat <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/priors_mat.csv', sep = ',', header = 1, row.names = 1)

nS <- 10
cores <- 10
no.pr.weight <- 1
prior.weight <- 1 # prior weights has to be larger than 1 to have an effect

weights.mat <- prior.mat * 0 + no.pr.weight
weights.mat[prior.mat != 0] <- prior.weight

x <- BBSR(X, Y, clr.mat, nS, no.pr.weight, weights.mat, 
                prior.mat, cores)

# Made a change here: replaced gp.out$tf.names with colnames(prior.mat)
bs.betas <- Matrix(0, nrow(Y), length(colnames(prior.mat)), 
                       dimnames=list(rownames(Y), colnames(prior.mat)))
bs.betas.resc <- bs.betas
for (res in x) {
      bs.betas[res$ind, rownames(X)[res$pp]] <- res$betas
      bs.betas.resc[res$ind, rownames(X)[res$pp]] <- res$betas.resc
}

write.table(as.matrix(bs.betas), '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/betas.tsv', sep = '\t')
write.table(as.matrix(bs.betas.resc), '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/betas_rescaled.tsv', sep = '\t')
cat("done. \n")
