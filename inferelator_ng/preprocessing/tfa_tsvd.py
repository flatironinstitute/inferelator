from __future__ import division

import itertools
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.utils.extmath import randomized_svd

from inferelator_ng import utils
from inferelator_ng import default
from inferelator_ng.distributed.inferelator_mp import MPControl
from inferelator_ng.preprocessing import tfa

DEFAULT_TSVD_POWER_NORMALIZER = "QR"


class TruncatedSVDTFA(tfa.TFA):

    def compute_transcription_factor_activity(self, allow_self_interactions_for_duplicate_prior_columns=True):

        activity, self.prior, non_zero_tfs = tfa.process_expression_into_activity(self.expression_matrix, self.prior)
        self.fix_self_interacting(non_zero_tfs, allow_duplicates=allow_self_interactions_for_duplicate_prior_columns)

        # Set the activity of non-zero tfs to the pseudoinverse of the prior matrix times the expression
        if len(non_zero_tfs) > 0:
            P = np.mat(self.prior[non_zero_tfs])
            X = np.matrix(self.expression_matrix_halftau)
            utils.Debug.vprint('Running TSVD...', level=1)
            self.k_val = gcv(P, X, 0)['val']
            utils.Debug.vprint('Selected {k} dimensions for TSVD'.format(k=self.k_val), level=1)
            A_k = tsvd_simple(P, X, self.k_val)
            activity.loc[non_zero_tfs, :] = np.matrix(A_k)
        else:
            utils.Debug.vprint("No prior information for TFs exists. Using expression for TFA exclusively.", level=0)

        return activity


def tsvd_simple(P, X, k, seed=default.DEFAULT_RANDOM_SEED, power_iteration_normalizer=DEFAULT_TSVD_POWER_NORMALIZER):
    U, Sigma, VT = randomized_svd(P, n_components=k, random_state=seed,
                                  power_iteration_normalizer=power_iteration_normalizer)
    Sigma_inv = [0 if s == 0 else 1. / s for s in Sigma]
    S_inv = np.diagflat(Sigma_inv)
    A_k = np.transpose(np.mat(VT)) * np.mat(S_inv) * np.transpose(np.mat(U)) * np.mat(X)
    return A_k


def gcv(P,X,biggest):
    #Make into a 'map' call so this is vectorized
    m = len(P)
    if biggest == 0:
        biggest = np.linalg.matrix_rank(P)

    if MPControl.name() == "dask":
        GCVect = gcv_dask(P, X, biggest, m)
    else:
        num_iter = biggest - 1
        GCVect = MPControl.map(calculate_gcval, itertools.repeat(P, num_iter), itertools.repeat(X, num_iter),
                               range(1, biggest), itertools.repeat(m, num_iter))

    GCVal = GCVect.index(min(GCVect)) + 1
    return {'val':GCVal,'vect':GCVect}


def calculate_gcval(P, X, k, m):
    utils.Debug.vprint("TSVD: {k} / {i}".format(k=k, i=min(P.shape)), level=2)
    A_k = tsvd_simple(P, X, k)
    Res = linalg.norm(P * A_k - X, 2)
    return (Res / (m - k)) ** 2


def gcv_dask(P, X, biggest, m):
    from dask import distributed
    DaskController = MPControl.client

    def gcv_maker(P, X, k, m):
        return k, calculate_gcval(P, X, k, m)

    [scatter_p] = DaskController.client.scatter([P])
    [scatter_x] = DaskController.client.scatter([X])
    future_list = [DaskController.client.submit(gcv_maker, scatter_p, scatter_x, i, m)
                   for i in range(1, biggest)]

    # Collect results as they finish instead of waiting for all workers to be done
    result_list = [None] * len(future_list)
    for finished_future, (j, result_data) in distributed.as_completed(future_list, with_results=True):
        result_list[j - 1] = result_data
        finished_future.cancel()

    DaskController.client.cancel(scatter_x)
    DaskController.client.cancel(scatter_p)
    DaskController.client.cancel(future_list)

    return result_list
