from inferelator import utils

from .tfa_base import (
    ActivityExpressionTFA,
    ActivityOnlyTFA
)

from scipy import (
    linalg,
    sparse
)

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

        _arr_piv = linalg.pinv(prior).T.astype(_prior_dtype)

        if sparse.isspmatrix(expression_data):
            _arr_piv = sparse.csr_matrix(_arr_piv)

        return utils.DotProduct.dot(
            expression_data,
            _arr_piv,
            dense=True,
            cast=True
        )


# This is named `TFA` for backwards compatibility #
class TFA(_Pinv_TFA_mixin, ActivityExpressionTFA):
    pass


class ActivityOnlyPinvTFA(_Pinv_TFA_mixin, ActivityOnlyTFA):
    pass
