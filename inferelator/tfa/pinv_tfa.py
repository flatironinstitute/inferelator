from inferelator import utils
from .tfa_base import (
    ActivityExpressionTFA,
    ActivityOnlyTFA
)
from scipy import linalg
from sklearn.preprocessing import RobustScaler
import numpy as np


class _Pinv_TFA_mixin:
    """
    TFA calculates transcription factor activity
    using matrix pseudoinverse """

    @staticmethod
    def _calculate_activity(
        prior,
        expression_data
    ):

        if expression_data.dtype == np.float32:
            _prior_dtype = np.float32
        else:
            _prior_dtype = np.float64

        return utils.DotProduct.dot(
            expression_data,
            linalg.pinv(prior).T.astype(_prior_dtype),
            dense=True,
            cast=True
        )


### This is named `TFA` for backwards compatibility ###
class TFA(_Pinv_TFA_mixin, ActivityExpressionTFA):
    pass


class ActivityOnlyPinvTFA(_Pinv_TFA_mixin, ActivityOnlyTFA):
    pass


class NormalizedExpressionPinvTFA(_Pinv_TFA_mixin, ActivityOnlyTFA):

    @staticmethod
    def _calculate_activity(prior, expression_data):

        return _Pinv_TFA_mixin._calculate_activity(
            prior,
            NormalizedExpressionPinvTFA._interval_normalize(
                expression_data
            )
        )

    @staticmethod
    def _interval_normalize(array):
        """
        Takes a 2d array and scale it with the RobustScaler
        Enforce positive values
            or (-1, 1)
        :param arr_vec: np.ndarray
            1d array of data
        :return array: np.ndarray
            1d array of scaled data
        """

        arr_scale = RobustScaler(
            with_centering=False
        ).fit_transform(array)

        return arr_scale
