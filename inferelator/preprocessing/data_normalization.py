import numpy as np
from scipy import (
    sparse,
    stats
)
from sklearn.preprocessing import (
    RobustScaler
)

from inferelator.utils.debug import Debug
from inferelator.utils.data import convert_array_to_float

# Dict keyed by method
# Values are (design function, response function, pre-tfa function)
_PREPROCESS_METHODS = {
    'zscore': (
        lambda x, y: x.apply(scale_array, magnitude_limit=y),
        lambda x, y: scale_vector(x, magnitude_limit=y),
        lambda x, y: scale_array(x, magnitude_limit=y)
    ),
    'robustscaler': (
        lambda x, y: x.apply(robust_scale_array, magnitude_limit=y),
        lambda x, y: robust_scale_vector(x, magnitude_limit=y),
        lambda x, y: robust_scale_array(x, magnitude_limit=y)
    ),
    'raw': (
        lambda x, y: x,
        lambda x, y: x,
        lambda x, y: x
    )
}


class PreprocessData:

    method_predictors = 'zscore'
    method_response = 'zscore'
    method_tfa = 'raw'

    scale_limit_predictors = None
    scale_limit_response = None
    scale_limit_tfa = None

    _design_func = _PREPROCESS_METHODS['zscore'][0]
    _response_func = _PREPROCESS_METHODS['zscore'][1]
    _tfa_func = _PREPROCESS_METHODS['raw'][2]

    @classmethod
    def set_preprocessing_method(
        cls,
        method=None,
        method_predictors=None,
        method_response=None,
        method_tfa=None,
        scale_limit='',
        scale_limit_predictors='',
        scale_limit_response='',
        scale_limit_tfa=''
    ):
        """
        Set preprocessing method.

        :param method: Normalization method before regression.
            If passed, will set the normalization for both predictors and
            response variables. Overridden by method_predictors and
            method_response. Supports 'zscore', 'robustscaler', and 'raw'.
            Defaults to 'zscore'.
        :type method: str
        :param method_predictors: Normalization method before regression.
            If passed, will set the normalization for predictor variables.
            Supports 'zscore', 'robustscaler', and 'raw'.
            Defaults to 'zscore'.
        :type method_predictors: str
        :param method_response: Normalization method before regression.
            If passed, will set the normalization for response variables.
            Supports 'zscore', 'robustscaler', and 'raw'.
            Defaults to 'zscore'.
        :type method_response: str
        :type method: str
        :param scale_limit: Absolute value limit for scaled values.
            If passed, will set the magnitude limit for both predictors and
            response variables. Overridden by scale_limit_predictors and
            scale_limit_response.
            None disables. Defaults to None.
        :type scale_limit: numeric, None
        :param scale_limit_predictors: Absolute value limit for scaled values
            specific to predictors. None disables. Defaults to None.
        :type scale_limit_predictors: numeric, None
        :param scale_limit_response: Absolute value limit for scaled values
            specific to response. None disables. Defaults to None.
        :type scale_limit_response: numeric, None
        """

        if method is not None:
            cls._check_method_arg(method)
            cls._design_func = _PREPROCESS_METHODS[method][0]
            cls._response_func = _PREPROCESS_METHODS[method][1]
            cls.method_predictors = method
            cls.method_response = method

        if method_predictors is not None:
            cls._check_method_arg(method_predictors)
            cls._design_func = _PREPROCESS_METHODS[method_predictors][0]
            cls.method_predictors = method_predictors

        if method_response is not None:
            cls._check_method_arg(method_response)
            cls._response_func = _PREPROCESS_METHODS[method_response][1]
            cls.method_response = method_response

        if method_tfa is not None:
            cls._check_method_arg(method_tfa)
            cls._tfa_func = _PREPROCESS_METHODS[method_tfa][2]
            cls.method_tfa = method_tfa

        if scale_limit != '':
            cls.scale_limit_response = scale_limit
            cls.scale_limit_predictors = scale_limit

        if scale_limit_predictors != '':
            cls.scale_limit_predictors = scale_limit_predictors

        if scale_limit_response != '':
            cls.scale_limit_response = scale_limit_response

        if scale_limit_tfa != '':
            cls.scale_limit_tfa = scale_limit_tfa

        Debug.vprint(
            "Preprocessing methods selected: "
            f"Predictor method {cls.method_predictors} "
            f"[limit {cls.scale_limit_predictors}] "
            f"Response method {cls.method_response} "
            f"[limit {cls.scale_limit_response}] "
            f"Pre-TFA expression method {cls.method_tfa} "
            f"[limit {cls.scale_limit_tfa}] ",
            level=1
        )

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
            cls.scale_limit_predictors
        )

    @classmethod
    def preprocess_response_vector(cls, y):
        """
        Preprocess data for response vector.
        Should not modify underlying data.

        :param y: Design vector [N x K]
        :type X: InferelatorData
        """

        return cls._response_func(
            y,
            cls.scale_limit_response
        )

    @classmethod
    def preprocess_expression_array(cls, X):
        """
        Preprocess data for expression data matrix before TFA.
        Returns a reference to existing data or a copy

        :param X: Expression matrix [N x G]
        :type X: np.ndarray, sp.spmatrix
        """

        return cls._tfa_func(
            X,
            cls.scale_limit_tfa
        )

    @classmethod
    def to_dict(cls):
        """
        Return preprocessing settings as a dict
        """

        return {
            'method_predictors': cls.method_predictors,
            'method_response': cls.method_response,
            'method_tfa': cls.method_tfa,
            'scale_limit_predictors': cls.scale_limit_predictors,
            'scale_limit_response': cls.scale_limit_response,
            'scale_limit_tfa': cls.scale_limit_tfa,
        }

    @staticmethod
    def _check_method_arg(method):
        """
        Check the method argument against supported values.

        :param method: Method argument
        :type method: str
        :raises ValueError: Raise if method string isnt supported
        """
        if method not in _PREPROCESS_METHODS.keys():
            raise ValueError(
                f"{method} is not supported; options are "
                f"{list(_PREPROCESS_METHODS.keys())}"
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


def scale_array(
    array,
    ddof=1,
    magnitude_limit=None
):
    """
    Take a vector and normalize it to a mean 0 and
    standard deviation 1 (z-score)

    :param array: Array
    :type array: np.ndarray, sp.sparse.spmatrix
    :param ddof: The delta degrees of freedom for variance calculation
    :type ddof: int
    :param magnitude_limit: Absolute value limit,
        defaults to None
    :type magnitude_limit: numeric, optional
    """

    if sparse.isspmatrix(array):
        out = np.empty(
            shape=array.shape,
            dtype=float
        )
    else:
        array = convert_array_to_float(array)
        out = array

    for i in range(array.shape[1]):
        out[:, i] = scale_vector(
            array[:, i],
            ddof=ddof,
            magnitude_limit=magnitude_limit
        )

    return out


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
    :return: A centered and scaled 1d vector
    :rtype: np.ndarray
    """

    # Convert a sparse vector to a dense vector
    if sparse.isspmatrix(vec):
        vec = vec.A.ravel()

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

    ref = x.data if sparse.isspmatrix(x) else x

    np.minimum(ref, lim, out=ref)
    np.maximum(ref, -1 * lim, out=ref)

    return x
