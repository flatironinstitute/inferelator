import pandas as pd
import numpy as np
import itertools
from itertools import compress
import math
from scipy import special
import multiprocessing
from functools import partial
import os, sys

gx, gy, gpp, gwm, gns = None, None, None, None, None
def BBSRforOneGeneWrapper(ind): return BBSRforOneGene(ind, gx, gy, gpp, gwm, gns)

def BBSR(X, Y, clr_mat, nS, no_pr_val, weights_mat, prior_mat, cores):
    G = Y.shape[0] # number of genes
    genes = Y.index.values.tolist()
    K = X.shape[0]  # max number of possible predictors (number of TFs)
    tfs = X.index.values.tolist()

    X = ((X.transpose() - X.transpose().mean()) / X.transpose().std(ddof=1)).transpose()
    Y = ((Y.transpose() - Y.transpose().mean()) / Y.transpose().std(ddof=1)).transpose()

    weights_mat = weights_mat.loc[genes,tfs]
    clr_mat = clr_mat.loc[genes, tfs]
    prior_mat = prior_mat.loc[genes, tfs]

    pp = pd.DataFrame(((prior_mat.ix[:,:] != 0)|(weights_mat.ix[:,:]!=no_pr_val)) & ~pd.isnull(clr_mat))
    mask = clr_mat == 0
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
    for i in range(0,G):
        #OneGene=BBSRforOneGene(i, X, Y, pp, weights_mat, nS)
        OneGene=BBSRforOneGeneWrapper(i)
        out_list.append(OneGene)


    '''
    multiprocessing implementation
    global gx, gy, gpp, gwm, gns
    gx, gy, gpp, gwm, gns = X, Y, pp, weights_mat, nS
    pool = multiprocessing.Pool(processes=4)
    #G=1
    gene_list = range(0,G)
    #BBSR_inp=partial(BBSRforOneGene,X=X, Y=Y, pp=pp, weights_mat=weights_mat, nS=nS)
    out_list = pool.map(BBSRforOneGeneWrapper, gene_list) #chunksize, # shared memory concept , #xy in shared memory, wrapper for BBSR that invokes computation and references copies in shared memoryy
    '''

    return out_list

def BBSRforOneGene(ind, X, Y, pp, weights_mat, nS):
    if ind % 100 == 0:
        print('Progress: BBSR for gene', ind , '\n')

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
    betas_resc = PredErrRed(y, x, betas)

    return (dict(ind=ind, pp=pp_i, betas=betas, betas_resc=betas_resc))#, x=x,g=g,spp=spp,y=y))



def ReduceNumberOfPredictors(y, x, g, n):
    K = x.shape[1] #what is the maximum size of K, print K
    spp = None
    if K <= n:
        spp = np.repeat(True, K).tolist()
        return spp

    combos = np.hstack((np.diag(np.repeat(True,K)),CombCols(K)))
    bics = ExpBICforAllCombos(y, x, g, combos)
    bics_avg = np.sum(np.multiply(combos.transpose(),bics[:, np.newaxis]).transpose(),1)
    bics_avg = list(bics_avg)
    ret = np.repeat(False, K)
    ret[np.argsort(bics_avg)[0:n]] = True
    return ret


def BestSubsetRegression(y, x, g):
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

            try:
                bhat = np.linalg.solve(np.dot(x_tmp.transpose(),x_tmp),np.dot(x_tmp.transpose(),y))
                for m in range(len(lst_true_index)):
                    ind_t=lst_true_index[m]
                    betas[ind_t]=bhat[m]
                not_done = False

            except:
                bics[best] = np.inf

                '''
                if e[:,call].str.contains('solve.default') and e[:,message].str.contains('singular'):
                    # error in solve - system is computationally singular
                    print bics[best], 'at', best, 'replaced\n'
                    bics[best] = np.nan #shaky bics[best] <<- Inf
                else:
                    raise ValueError('')
                '''
        else:
            not_done = False

    return betas


def AllCombinations(k):
    if k < 1:
        raise ValueError("No combinations for k < 1")
    lst = map(list, itertools.product([False, True], repeat=k))
    out=np.array(lst).transpose()
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
    K = x.shape[1]
    N = x.shape[0]

    C = combos.shape[1]
    bics = np.array(np.repeat(0,C),dtype=np.float)

    first_combo = 0
    if sum(combos[:,0]) == 0:
        bics[0] = N * math.log(np.var(y,ddof=1))
        first_combo = 1

    shape = N / 2
    dig_shape = special.digamma(shape)

    #### pre-compute the dot product we will need to solve for beta
    xtx = np.dot(x.transpose(),x)
    xty = np.dot(x.transpose(),y)

    var_mult = np.array(np.repeat(np.sqrt(1 / (g + 1)), K,axis=0)).transpose()
    var_mult = np.multiply(var_mult,var_mult.transpose())

    for i in range(first_combo, C):
        comb = combos[:, i]
        comb=np.where(comb)[0]

        x_tmp = x[:,comb]
        k = len(comb)

        try:
            xtx_tmp=xtx[:,comb][comb,:]
            var_mult_tmp=var_mult[:,comb][comb,:]

            bhat = np.linalg.solve(xtx_tmp,xty[comb])
            ssr = np.sum(np.power(np.subtract(y,np.dot(x_tmp, bhat)),2)) # sum of squares of residuals
            rate = (ssr + np.dot((0 - bhat.transpose()) , np.dot(np.multiply(xtx_tmp, var_mult_tmp) ,(0 - bhat.transpose()).transpose()))) / 2
            exp_log_sigma2 = math.log(rate) - dig_shape
            bics[i] = N * exp_log_sigma2 + k * math.log(N)

        except:
            bics[i] = np.inf
            raise ValueError('')
            '''
            if e[:,call].str.contains('solve.default') | e[:,message].str.contains('singular'):

            #https://stackoverflow.com/questions/38745710/simplest-python-equivalent-to-rs-grepl
            #https://stat.ethz.ch/R-manual/R-devel/library/base/html/grep.html
                bics[i] = np.nan  # in R this was bics[i] <<- Inf
            else:
                raise ValueError('') # in R this was just stop
            '''
    return(bics)



def PredErrRed(y, x, beta):
    N = x.shape[0]
    K = x.shape[1]
    pred = [True if item!=0 else False for item in beta]
    pred_index = [l for l, j in enumerate(pred) if j]
    P = sum(pred)

    residuals = np.subtract(y,np.dot(x,beta)[:, np.newaxis])
    sigma_sq_full = np.var(residuals,ddof=1)
    err_red = np.repeat(0.0,K)


    if P == 1:
        err_red[pred_index] = 1 - (sigma_sq_full/np.var(y,ddof=1))

    for i in pred_index[0:K]:
        pred_tmp = pred[:]
        pred_tmp[i] = False
        pred_tmp_index= [l for l, j in enumerate(pred_tmp) if j]

        x_tmp = x[:,pred_tmp_index]

        try:
            bhat = np.linalg.solve(np.dot(x_tmp.transpose(),x_tmp),np.dot(x_tmp.transpose(),y))
        except:
            raise ValueError('')

        residuals = np.subtract(y,np.dot(x_tmp,bhat))
        sigma_sq = np.var(residuals,ddof=1)
        err_red[i] = 1 - (sigma_sq_full / sigma_sq)


    return err_red


class BBSR_runner:
    def run(self, X, Y, clr, priors):
        n = 10
        cores = 10
        no_prior_weight = 1
        prior_weight = 1 # prior weights has to be larger than 1 to have an effect
        no_pr_val = no_prior_weight
        nS = n
        X = X
        Y = Y
        clr_mat = clr
        prior_mat = priors
        weights_mat = prior_mat * 0 + no_prior_weight
        weights_mat=weights_mat.mask(prior_mat != 0, other=prior_weight)

        x = BBSR(X, Y, clr_mat, nS, no_pr_val, weights_mat, prior_mat, cores)
        bs_betas = pd.DataFrame(np.zeros((Y.shape[0],prior_mat.shape[1])),index=Y.index,columns=prior_mat.columns)
        bs_betas_resc = bs_betas.copy(deep=True)
        for res in x:
            bs_betas.ix[res['ind'],X.index.values[res['pp']]] = res['betas']
            bs_betas_resc.ix[res['ind'],X.index.values[res['pp']]] = res['betas_resc']
        return (bs_betas, bs_betas_resc)
