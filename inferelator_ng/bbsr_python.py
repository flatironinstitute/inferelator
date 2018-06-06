import pandas as pd
import numpy as np
import itertools
from itertools import compress
import math
from scipy import special
import multiprocessing
from functools import partial
import os, sys
from . import utils

# Wrapper function for BBSRforOneGene that's called in BBSR
gx, gy, gpp, gwm, gns = None, None, None, None, None
def BBSRforOneGeneWrapper(ind): return BBSRforOneGene(ind, gx, gy, gpp, gwm, gns)

def BBSR(X, Y, clr_mat, nS, no_pr_val, weights_mat, prior_mat, kvs, rank, ownCheck):
    G = Y.shape[0] # number of genes
    genes = Y.index.values.tolist()
    K = X.shape[0]  # max number of possible predictors (number of TFs)
    tfs = X.index.values.tolist()

    # Scale and permute design and response matrix
    X = ((X.transpose() - X.transpose().mean()) / X.transpose().std(ddof=1)).transpose()
    Y = ((Y.transpose() - Y.transpose().mean()) / Y.transpose().std(ddof=1)).transpose()

    weights_mat = weights_mat.loc[genes,tfs]
    clr_mat = clr_mat.loc[genes, tfs]
    prior_mat = prior_mat.loc[genes, tfs]
    # keep all predictors that we have priors for
    pp = pd.DataFrame(((prior_mat.ix[:,:] != 0)|(weights_mat.ix[:,:]!=no_pr_val)) & ~pd.isnull(clr_mat))
    mask = clr_mat == 0
    # for each gene, add the top nS predictors of the list to possible predictors
    clr_mat[mask] = np.nan

    for ind in range(0,G):

        clr_na = len(np.argwhere(np.isnan(clr_mat.ix[ind,])).flatten().tolist())
        clr_w_na = np.argsort(clr_mat.ix[ind,].tolist())

        if clr_na>0:
            clr_order = clr_w_na[:-clr_na][::-1]
        else:
            clr_order = clr_w_na[:][::-1]
        pp.ix[ind, clr_order[0:min(K, nS, len(clr_order))]] = True

    preds = np.intersect1d(genes, tfs)
    subset = pp.ix[preds,preds].values
    np.fill_diagonal(subset,False)
    pp=pp.set_value(preds, preds, subset)

    out_list=[]

    global gx, gy, gpp, gwm, gns
    gx, gy, gpp, gwm, gns = X, Y, pp, weights_mat, nS
    # Here we illustrate splitting a simple loop, but the same approach
    # would work with any iterative control structure, as long as it is
    # deterministic.
    s = []
    limit = G
    for j in range(limit):
        if next(ownCheck):
            s.append(BBSRforOneGeneWrapper(j))
    # Report partial result.
    kvs.put('plist',(rank,s))
    # One participant gathers the partial results and generates the final
    # output.
    if 0 == rank:
        s=[]
        workers=int(os.environ['SLURM_NTASKS'])
        for p in range(workers):
            wrank,ps = kvs.get('plist')
            s.extend(ps)
        print ('final s', len(s))
        utils.kvsTearDown(kvs, rank)
        return s
    else:
        return None

def BBSRforOneGene(ind, X, Y, pp, weights_mat, nS):
    if ind % 100 == 0:
        print('Progress: computing BBSR for gene {}'.format(ind))

    pp_i = pp.ix[ind,].values # converted to numpy array
    pp_i_index = [l for l, j in enumerate(pp_i) if j]

    if sum(pp_i) == 0:
        return dict(ind=ind,pp=np.repeat(True, len(pp_i)).tolist(),betas=0, betas_resc=0)

    # create BestSubsetRegression input
    y = Y.ix[ind,:][:, np.newaxis]
    x = X.ix[pp_i_index,:].transpose().values # converted to numpy array
    g = np.matrix(weights_mat.ix[ind,pp_i_index],dtype=np.float)

    # experimental stuff
    spp = ReduceNumberOfPredictors(y, x, g, nS)

    #check again
    pp_i[pp_i==True] = spp # this could cause issues if they aren't the same length
    pp_i_index = [l for l, j in enumerate(pp_i) if j]
    x = X.ix[pp_i_index,:].transpose().values # converted to numpy array
    g = np.matrix(weights_mat.ix[ind,pp_i_index],dtype=np.float)

    betas = BestSubsetRegression(y, x, g)
    betas_resc = PredictErrorReduction(y, x, betas)

    return (dict(ind=ind, pp=pp_i, betas=betas, betas_resc=betas_resc))



def ReduceNumberOfPredictors(y, x, g, n):
    K = x.shape[1] #what is the maximum size of K, print K
    spp = None
    if K <= n:
        spp = np.repeat(True, K).tolist()
        return spp

    combos = np.hstack((np.diag(np.repeat(True,K)),CombCols(K)))
    bics = ExpBICforAllCombos(y, x, g, combos)
    bics_sum = np.sum(np.multiply(combos.transpose(),bics[:, np.newaxis]).transpose(),1)
    bics_sum = list(bics_sum)
    ret = np.repeat(False, K)
    ret[np.argsort(bics_sum)[0:n]] = True
    return ret


def BestSubsetRegression(y, x, g):
    # Do best subset regression by using all possible combinations of columns of
    #x as predictors of y. Model selection criterion is BIC using results of
    # Bayesian regression with Zellner's g-prior.
    # Args:
    #   y: dependent variable
    #   x: independent variable
    #   g: value for Zellner's g-prior; can be single value or vector
    # Returns:
    #   Beta vector of best mode
    K = x.shape[1]
    N = x.shape[0]
    ret = []

    combos = AllCombinations(K)
    bics = ExpBICforAllCombos(y, x, g, combos)
    not_done = True
    while not_done:

        best = np.argmin(bics)
        betas = np.repeat(0.0,K)
        if best > 0:

            lst_combos_bool=combos[:, best]
            lst_true_index = [i for i, j in enumerate(lst_combos_bool) if j]
            x_tmp = x[:,lst_true_index]

            bhat = np.linalg.solve(np.dot(x_tmp.transpose(),x_tmp),np.dot(x_tmp.transpose(),y))
            for m in range(len(lst_true_index)):
                ind_t=lst_true_index[m]
                betas[ind_t]=bhat[m]
            not_done = False
        else:
            not_done = False

    return betas


def AllCombinations(k):
    # Create a boolean matrix with all possible combinations of 1:k. Output has k rows and 2^k columns where each column is one combination.
    # Note that the first column is all FALSE and corresponds to the null model.
    if k < 1:
        raise ValueError("No combinations for k < 1")
    lst = map(list, itertools.product([False, True], repeat=k))
    out=np.array([i for i in lst]).transpose()
    return out

# Get all possible pairs of K predictors
def CombCols(K):
    num_pair = K*(K-1)/2
    a = np.full((num_pair,K), False, dtype=bool)
    b = list(list(tup) for tup in itertools.combinations(range(K), 2))
    for i in range(len(b)):
        a[i,b[i]]=True
    c = a.transpose()
    return c

def ExpBICforAllCombos(y, x, g, combos):
    # For a list of combinations of predictors do Bayesian linear regression, more specifically calculate the parametrization of the inverse gamma
    # distribution that underlies sigma squared using Zellner's g-prior method.
    # Parameter g can be a vector. The expected value of the log of sigma squared is used to compute expected values of BIC.
    # Returns list of expected BIC values, one for each model.
    K = x.shape[1]
    N = x.shape[0]
    C = combos.shape[1]
    bics = np.array(np.repeat(0,C),dtype=np.float)

    # is the first combination the null model?
    first_combo = 0
    if sum(combos[:,0]) == 0:
        bics[0] = N * math.log(np.var(y,ddof=1))
        first_combo = 1

    # shape parameter for the inverse gamma sigma squared would be drawn from
    shape = N / 2
    # compute digamma of shape here, so we can re-use it later
    dig_shape = special.digamma(shape)

    #### pre-compute the dot products that we will need to solve for beta
    xtx = np.dot(x.transpose(),x)
    xty = np.dot(x.transpose(),y)

    # In Zellner's formulation there is a factor in the calculation of the rate parameter: 1 / (g + 1)
    # Here we replace the factor with the approriate matrix since g is a vector now.
    var_mult = np.array(np.repeat(np.sqrt(1 / (g + 1)), K,axis=0)).transpose()
    var_mult = np.multiply(var_mult,var_mult.transpose())

    for i in range(first_combo, C):
        comb = combos[:, i]
        comb=np.where(comb)[0]

        x_tmp = x[:,comb]
        k = len(comb)

        xtx_tmp=xtx[:,comb][comb,:]
        # if the xtx_tmp matrix is singular, set bic to infinity
        if np.linalg.matrix_rank(xtx_tmp, tol=1e-10) == xtx_tmp.shape[1]:
            var_mult_tmp=var_mult[:,comb][comb,:]
            #faster than calling lm
            bhat = np.linalg.solve(xtx_tmp,xty[comb])
            ssr = np.sum(np.power(np.subtract(y,np.dot(x_tmp, bhat)),2)) # sum of squares of residuals
            # rate parameter for the inverse gamma sigma squared would be drawn from our guess on the regression vector beta is all 0 for sparse models
            rate = (ssr + np.dot((0 - bhat.transpose()) , np.dot(np.multiply(xtx_tmp, var_mult_tmp) ,(0 - bhat.transpose()).transpose()))) / 2
            # the expected value of the log of sigma squared based on the parametrization of the inverse gamma by rate and shape
            exp_log_sigma2 = math.log(rate) - dig_shape
            # expected value of BIC
            bics[i] = N * exp_log_sigma2 + k * math.log(N)

        # set bic to infinity if lin alg error
        else:
            bics[i] = np.inf

    return(bics)



def PredictErrorReduction(y, x, beta):
    # Calculates the error reduction (measured by variance of residuals) of each
    # predictor - compare full model to model without that predictor
    N = x.shape[0]
    K = x.shape[1]
    pred = [True if item!=0 else False for item in beta]
    pred_index = [l for l, j in enumerate(pred) if j]
    P = sum(pred)

    # compute sigma^2 for full model
    residuals = np.subtract(y,np.dot(x,beta)[:, np.newaxis])
    sigma_sq_full = np.var(residuals,ddof=1)
    # this will be the output
    err_red = np.repeat(0.0,K)

    # special case if there is only one predictor
    if P == 1:
        err_red[pred_index] = 1 - (sigma_sq_full/np.var(y,ddof=1))

    # one by one leave out each predictor and re-compute the model with the remaining ones
    for i in pred_index[0:K]:
        pred_tmp = pred[:]
        pred_tmp[i] = False
        pred_tmp_index= [l for l, j in enumerate(pred_tmp) if j]

        x_tmp = x[:,pred_tmp_index]

        bhat = np.linalg.solve(np.dot(x_tmp.transpose(),x_tmp),np.dot(x_tmp.transpose(),y))

        residuals = np.subtract(y,np.dot(x_tmp,bhat))
        sigma_sq = np.var(residuals,ddof=1)
        err_red[i] = 1 - (sigma_sq_full / sigma_sq)

    return err_red


class BBSR_runner:
    def run(self, X, Y, clr, prior_mat, kvs=None, rank=0, ownCheck=None):
        n = 10
        no_prior_weight = 1
        prior_weight = 1 # prior weight has to be larger than 1 to have an effect
        weights_mat = prior_mat * 0 + no_prior_weight
        weights_mat = weights_mat.mask(prior_mat != 0, other=prior_weight)

        run_result = BBSR(X, Y, clr, n, no_prior_weight, weights_mat, prior_mat, kvs, rank, ownCheck)
        if rank:
            return (None,None)
        bs_betas = pd.DataFrame(np.zeros((Y.shape[0],prior_mat.shape[1])),index=Y.index,columns=prior_mat.columns)
        bs_betas_resc = bs_betas.copy(deep=True)
        for res in run_result:
            bs_betas.ix[res['ind'],X.index.values[res['pp']]] = res['betas']
            bs_betas_resc.ix[res['ind'],X.index.values[res['pp']]] = res['betas_resc']
        return (bs_betas, bs_betas_resc)
