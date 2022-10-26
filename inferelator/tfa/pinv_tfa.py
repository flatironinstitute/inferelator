from inferelator import utils
from .tfa_base import (
    ActivityExpressionTFA,
    ActivityOnlyTFA
)
from scipy import linalg
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
            utils.safe_apply_to_array(
                expression_data,
                NormalizedExpressionPinvTFA._interval_normalize
            )
        )

    @staticmethod
    def _interval_normalize(arr_vec):
        """
        Takes a 1d array or vector and scale it to (0, 1)
            or (-1, 1)
        :param arr_vec: np.ndarray
            1d array of data
        :return array: np.ndarray
            1d array of scaled data
        """

        # Get array min and max
        arr_min, arr_max = np.min(arr_vec), np.max(arr_vec)

        # Short circuit if the variance is 0
        if arr_min == arr_max:
            return np.zeros_like(arr_vec)

        # Symmetric around zero interval (-1 to 1)
        if arr_min < 0 and arr_max > 0:
            arr_max = max(abs(arr_min), abs(arr_max))
            arr_min = -1 * arr_max

            arr_ret = (arr_vec - arr_min)
            arr_ret /= (arr_max - arr_min) * 0.5
            arr_ret -= 1

            return arr_ret

        # Interval normalize
        else:
            return (arr_vec - arr_min) / (arr_max - arr_min)
