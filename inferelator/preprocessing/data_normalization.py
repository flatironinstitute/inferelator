import numpy as np
from scipy import (
    sparse,
    stats
)
from sklearn.preprocessing import (
    RobustScaler
)

_PREPROCESS_METHODS = {
    'zscore': (
        lambda x, y: x.zscore(magnitude_limit=y),
        lambda x, y: scale_vector(x, magnitude_limit=y)
    ),
    'robustscaler': (
        lambda x, y: x.apply(robust_scale_array, magnitude_limit=y),
        lambda x, y: robust_scale_vector(magnitude_limit=y)
    ),
    'raw': (
        lambda x, y: x,
        lambda x, y: y
    )
}


class PreprocessData:

    method = 'zscore'
    scale_limit = None

    _design_func = _PREPROCESS_METHODS['zscore'][0]
    _response_func = _PREPROCESS_METHODS['zscore'][1]

    @classmethod
    def set_preprocessing_method(
        cls,
        method=None,
        scale_limit=''
    ):
        """
        Set preprocessing method.

        :param method: Method. Support 'zscore', 'robustscaler', and 'raw'.
            Defaults to 'zscore'.
        :type method: str
        :param scale_limit: Absolute value limit for scaled values.
            None disables. Defaults to None.
        :type scale_limit: numeric, None
        """

        if method is not None:
            if method not in _PREPROCESS_METHODS.keys():
                raise ValueError(
                    f"{method} is not supported; options are "
                    f"{list(_PREPROCESS_METHODS.keys())}"
                )

            cls._design_func, cls._response_func = _PREPROCESS_METHODS[method]

        if scale_limit != '':
            cls.scale_limit = scale_limit

    @classmethod
    def preprocess_design(cls, X):
        """
        Preprocess data for design matrix.
        Modifies InferelatorData in place and returns reference

        :param X: Calculated activity matrix (from full prior) [N x K]
        :type X: InferelatorData
        """

        return cls._design_func(
            X,
            cls.scale_limit
        )

    @classmethod
    def preprocess_response_vector(cls, y):
        """
        Preprocess data for design matrix.
        Should not modify underlying data.

        :param y: Design vector [N x K]
        :type X: InferelatorData
        """

        return cls._response_func(
            y,
            cls.scale_limit
        )


def robust_scale_vector(
    vec,
    magnitude_limit=None
):

    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)

    return robust_scale_array(
        vec,
        magnitude_limit=magnitude_limit
    ).ravel()


def robust_scale_array(
    arr,
    magnitude_limit=None
):

    z = RobustScaler(
        with_centering=False
    ).fit_transform(
        arr
    )

    if magnitude_limit is None:
        return z
    else:
        return _magnitude_limit(z, magnitude_limit)


def scale_vector(
    vec,
    ddof=1,
    magnitude_limit=None
):
    """
    Take a vector and normalize it to a mean 0 and
    standard deviation 1 (z-score)

    :param vec: A 1d vector to be normalized
    :type vec: np.ndarray, sp.sparse.spmatrix
    :param ddof: The delta degrees of freedom for variance calculation
    :type ddof: int
    :return: A centered and scaled vector
    :rtype: np.ndarray
    """

    # Convert a sparse vector to a dense vector
    if sparse.isspmatrix(vec):
        vec = vec.A

    # Return 0s if the variance is 0
    if np.var(vec) == 0:
        return np.zeros(vec.shape, dtype=float)

    # Otherwise scale with scipy.stats.zscore
    z = stats.zscore(vec, axis=None, ddof=ddof)

    if magnitude_limit is None:
        return z
    else:
        return _magnitude_limit(z, magnitude_limit)


def _magnitude_limit(x, lim):

    x[x > lim] = lim
    x[x < (-1 * lim)] = -1 * lim

    return x
