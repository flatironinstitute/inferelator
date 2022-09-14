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

        if expression_data.values.dtype == np.float32:
            _prior_dtype = np.float32
        else:
            _prior_dtype = np.float64

        return utils.DotProduct.dot(
            expression_data.values,
            linalg.pinv(prior).T.astype(_prior_dtype),
            dense=True,
            cast=True
        )

### This is named `TFA` for backwards compatibility ###
class TFA(_Pinv_TFA_mixin, ActivityExpressionTFA):
    pass

class ActivityOnlyPinvTFA(_Pinv_TFA_mixin, ActivityOnlyTFA):
    pass
