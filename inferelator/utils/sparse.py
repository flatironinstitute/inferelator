import scipy.sparse as sps


def todense(sarr):

    if (
        sps.issparse(sarr) or
        sps.isspmatrix(sarr)
    ):

        try:
            sarr = sarr.toarray()
        except AttributeError:
            pass

    try:
        sarr = sarr.A
    except AttributeError:
        pass

    return sarr
