
source('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/R_code/mi_and_clr.R')

X <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/x.csv', sep = ',', header = 1, row.names = 1)
Y <- read.table('/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/y.csv', sep = ',', header = 1, row.names = 1)

PARS <- list()
PARS$mi.bins <- 10
PARS$cores <- 10

# fill mutual information matrices
Ms <- mi(t(Y), t(X), nbins=PARS$mi.bins, cpu.n=PARS$cores)
# write out the initial mi value for diagnostic purposes.
write.table(Ms, '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/mi_matrix.tsv', sep = '\t')
diag(Ms) <- 0
Ms_bg <- mi(t(X), t(X), nbins=PARS$mi.bins, cpu.n=PARS$cores)
diag(Ms_bg) <- 0

# get CLR matrix
clr.mat = mixedCLR(Ms_bg,Ms)
dimnames(clr.mat) <- list(rownames(Y), rownames(X))

write.table(clr.mat, '/Users/ndeveaux/Dev/inferelator_ng/inferelator_ng/tests/artifacts/clr_matrix.tsv', sep = '\t')
cat("done. \n")
