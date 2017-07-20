import pandas as pd
import numpy as np
import itertools
from itertools import compress
from sklearn.preprocessing import scale
import math
from scipy import special
#from dask import delayed
#from dask.distributed import Client
import multiprocessing
from functools import partial

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

        clr_na = len(np.argwhere(np.isnan(clr_mat.ix[ind,])).flatten().tolist()) #tolist()
        clr_w_na = np.argsort(clr_mat.ix[ind,].tolist()) #produces -1 for nan, they will be at end of sort
        #clr_order= clr_w_na[:-clr_na][::-1]
        if clr_na>0:
            clr_order = clr_w_na[:-clr_na][::-1]
        else:
            clr_order = clr_w_na[:][::-1]
        pp.ix[ind, clr_order[0:min(K, nS, len(clr_order))]] = True

    preds = np.intersect1d(genes, tfs)
    subset = pp.ix[preds,preds].values
    np.fill_diagonal(subset,False)
    pp=pp.set_value(preds, preds, subset)
    '''
    out_list=[]

    for i in range(0,G):
        OneGene=BBSRforOneGene(i, X, Y, pp, weights_mat, nS)
        #OneGene=delayed(BBSRforOneGene)(i, X, Y, pp, weights_mat, nS)
        out_list.append(OneGene)
    #return out_list
    total = delayed(out_list)
    return total.compute()

    #client = Client()
    #out_list = [client.submit(BBSRforOneGene, i, X, Y, pp, weights_mat, nS) for i in range(0,G)]
    #out_list = client.gather(out_list)
    return out_list
    '''
    pool = multiprocessing.Pool(processes=20)
    gene_list = range(0,G)
    BBSR_inp=partial(BBSRforOneGene,X=X, Y=Y, pp=pp, weights_mat=weights_mat, nS=nS)
    out_list = pool.map(BBSR_inp, gene_list)
    return out_list

def BBSRforOneGene(ind, X, Y, pp, weights_mat, nS):
    if ind % 100 == 0:
        print('Progress: BBSR for gene', ind , '\n')

    pp_i = pp.ix[ind,]
    pp_i_index = [l for l, j in enumerate(pp_i) if j]

    if sum(pp_i) == 0:
        return dict(ind=ind,pp=np.repeat(True, len(pp_i)).tolist(),betas=0, betas_resc=0)

    # create BestSubsetRegression input
    y = Y.ix[ind,:][:, np.newaxis]
    x = X.ix[pp_i_index,:].transpose()
    g = np.matrix(weights_mat.ix[ind,pp_i_index],dtype=np.float)

    # experimental stuff
    spp = ReduceNumberOfPredictors(y, x, g, nS)

    #check again
    pp_i[pp_i==True] = spp # this could cause issues if they aren't the same length
    pp_i_index = [l for l, j in enumerate(pp_i) if j]
    x = X.ix[pp_i_index,:].transpose()
    g = np.matrix(weights_mat.ix[ind,pp_i_index],dtype=np.float)

    betas = BestSubsetRegression(y, x, g)
    betas_resc = PredErrRed(y, x, betas)

    return (dict(ind=ind, pp=pp_i, betas=betas, betas_resc=betas_resc, x=x,g=g,spp=spp,y=y))


#will have to fix index numbers

def ReduceNumberOfPredictors(y, x, g, n):
    K = x.shape[1]
    spp = None
    if K <= n:
        spp = np.repeat(True, K).tolist()
        return spp

    combos = np.hstack((np.diag(np.repeat(True,K)),CombCols(np.diag(np.repeat(1,K)))))
    bics = ExpBICforAllCombos(y, x, g, combos)
    bics_avg = np.sum(np.multiply(combos.transpose(),bics[:, np.newaxis]).transpose(),1)
    bics_avg = list(itertools.chain.from_iterable(np.array(bics_avg)))
    ret = np.repeat(False, K)
    ret[np.argsort(bics_avg)[0:n]] = True

    return ret


def BestSubsetRegression(y, x, g):
    K = x.shape[1]
    N = x.shape[0]
    ret = []

    combos = AllCombinations(K)
    #print combos
    bics = ExpBICforAllCombos(y, x, g, combos)
    #print bics
    not_done = True
    while not_done:

        best = np.argmin(bics)
        betas = np.repeat(0.0,K)
        if best > 0:

            lst_combos_bool=combos[:, best]
            lst_true_index = [i for i, j in enumerate(lst_combos_bool) if j]
            x_tmp = x.ix[:,lst_true_index]

            try:
                bhat = np.linalg.solve(np.dot(x_tmp.transpose(),x_tmp),np.dot(x_tmp.transpose(),y))
                for m in range(len(lst_true_index)):
                    ind_t=lst_true_index[m]
                    betas[ind_t]=bhat[m]
                not_done = False

            except:
                bics[best] = np.inf #will remove this line later
                #raise ValueError('')
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
    out=np.matrix(lst).transpose()
    return out

# Get all possible pairs of K predictors
def CombCols(m):
    K = m.shape[1]
    lst = map(list, itertools.product([False, True], repeat=K))
    lst_pairs=[item for item in lst if sum(item)==2]
    ret=np.matrix(lst_pairs).transpose()
    return ret



def ExpBICforAllCombos(y, x, g, combos):
    import math
    from scipy import special

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

    #### pre-compute the crossprod that we will need to solve for beta (this is actually a dot product...misnomer)
    xtx = np.dot(x.transpose(),x)
    xty = np.dot(x.transpose(),y)

    var_mult = np.matrix(np.repeat(np.sqrt(1 / (g + 1)), K,axis=0)).transpose()
    var_mult = np.multiply(var_mult,var_mult.transpose())

    for i in range(first_combo, C): #will have to fix index
        comb = combos[:, i]
        comb = [l for l, j in enumerate(comb) if j]

        x_tmp = x.ix[:,comb]
        k = len(comb)

        try:
            xtx_tmp=xtx[:,comb]
            xtx_tmp=xtx_tmp[comb,:]

            var_mult_tmp=var_mult[:,comb]
            var_mult_tmp=var_mult_tmp[comb,:]

            #print xtx_tmp
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
    pred = [True if item!=0 else False for item in beta]   #changing up code a bit, in R we do pred <- beta != 0
    pred_index = [l for l, j in enumerate(pred) if j] #get index of items that are True
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

        x_tmp = x.ix[:,pred_tmp_index]    #x_tmp = np.matrix(x[:,pred_tmp], N, P-1)

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
        prior_weight = 1
        no_pr_val = no_prior_weight
        nS = n
        X = X
        Y = Y
        clr_mat = clr
        prior_mat = priors
        weights_mat = prior_mat * 0 + no_prior_weight
        weights_mat=weights_mat.mask(prior_mat != 0, other=prior_mat)

        x = BBSR(X, Y, clr_mat, nS, no_pr_val, weights_mat, prior_mat, cores)
        bs_betas = pd.DataFrame(np.zeros((Y.shape[0],prior_mat.shape[1])),index=Y.index,columns=prior_mat.columns)
        bs_betas_resc = bs_betas.copy(deep=True)
        for res in x:
            bs_betas.ix[res['ind'],X.index.values[res['pp']]] = res['betas']
            bs_betas_resc.ix[res['ind'],X.index.values[res['pp']]] = res['betas_resc']
        return (bs_betas, bs_betas_resc)
