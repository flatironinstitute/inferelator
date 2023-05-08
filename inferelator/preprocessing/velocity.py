import numpy as np
from scipy import sparse

from inferelator.utils import (
    InferelatorData,
    Debug
)
from inferelator.utils import Validator as check


def extract_transcriptional_output(
    expression,
    velocity,
    global_decay=None,
    gene_specific_decay=None,
    gene_and_sample_decay=None,
    decay_constant_maximum=None
):
    """
    Extract the transcriptional output
    by combining expression and velocity

    :param expression: Expression data X
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt
    :type velocity: InferelatorData, np.ndarray
    :param global_decay: Decay constant to use for
        every gene and observation, defaults to None
    :type global_decay: float, optional
    :param gene_specific_decay: Decay constants that differ
        for every gene but are the same for every observation, defaults to None
    :type gene_specific_decay: pd.DataFrame, optional
    :param gene_and_sample_decay: Decay constants that differ
        for every gene and differ for every observation, defaults to None
    :type gene_and_sample_decay: InferelatorData, np.ndarray, optional
    :param decay_constant_maximum: Maximum allowed value for decay constant,
        values larger will be set to this value if it is not None,
        defaults to None
    :type decay_constant_maximum: float, optional
    :returns: Transcriptional output estimates
    :rtype: InferelatorData
    """

    # If axis-labeled data is passed, check that
    # the data objects passed have the same
    # dimensions & labels
    # and then use a dense array
    if isinstance(velocity, InferelatorData):
        assert check.indexes_align(
            (expression.gene_names, velocity.gene_names)
        )

        assert check.indexes_align(
            (expression.sample_names, velocity.sample_names)
        )

        _velocity = velocity.values

    else:
        _velocity = velocity

    # Use the same decay constant for every gene and observation
    if global_decay is not None:

        Debug.vprint(
            "Modeling transcription with fixed decay constant "
            f"{global_decay:.4f} for every gene"
        )

        return _global_decay(
            expression,
            _velocity,
            global_decay
        )

    # Use the same decay constant for every observation
    # but a different decay constant for every gene
    elif gene_specific_decay is not None:

        # Convert a dataframe or series to an array
        # after checking labels
        try:
            assert check.indexes_align(
                (
                    expression.gene_names,
                    gene_specific_decay.index
                )
            )

            gene_specific_decay = gene_specific_decay.values.ravel()

        except AttributeError:
            pass

        Debug.vprint(
            "Modeling transcription with velocity and decay per gene"
        )

        return _gene_constant_decay(
            expression,
            _velocity,
            gene_specific_decay,
            decay_constant_maximum=decay_constant_maximum
        )

    # Use a different decay constant for every gene and observation
    # Decay constants must be a [M x N] data object
    elif gene_and_sample_decay is not None:

        if isinstance(gene_and_sample_decay, InferelatorData):
            assert check.indexes_align(
                (
                    expression.sample_names,
                    gene_and_sample_decay.sample_names
                )
            )

            gene_and_sample_decay = gene_and_sample_decay.values

        Debug.vprint(
            "Modeling transcription with velocity and decay "
            "per sample and per gene"
        )

        return _gene_variable_decay(
            expression,
            _velocity,
            gene_and_sample_decay,
            decay_constant_maximum=decay_constant_maximum
        )

    # Don't use any decay (all decay constants are zero)
    # Just return the velocity object
    else:
        Debug.vprint(
            "Modeling transcription with velocity only"
        )

        return velocity


def _global_decay(
    expression,
    velocity,
    constant
):
    """
    Calculate dX/dt + lambda * X
    for a single value of lambda

    :param expression: Expression data X
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt
    :type velocity: np.ndarray
    :param constant: Decay constant to use
    :type constant: float
    :return: dX/dt + lambda * X
    :rtype: InferelatorData
    """

    if constant < 0:
        raise ValueError(
            "Decay cannot be negative; "
            f"{constant} provided"
        )

    # dx/dt + constant * X = f(A)
    return InferelatorData(
        _sparse_safe_add(
            velocity,
            _sparse_safe_multiply(
                expression.values,
                constant
            )
        ),
        gene_names=expression.gene_names,
        sample_names=expression.sample_names,
        meta_data=expression.meta_data
    )


def _gene_constant_decay(
    expression,
    velocity,
    decay_constants,
    decay_constant_maximum=None
):
    """
    Calculate dX/dt + lambda * X
    where values of lambda can differ per gene

    :param expression: Expression data X ([M x N])
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt ([M x N])
    :type velocity: np.ndarray
    :param decay_constants: Decay constant to use ([N x 1])
    :type decay_constants: pd.DataFrame
    :return: dX/dt + lambda * X
    :rtype: InferelatorData
    """

    if np.sum(decay_constants < 0) > 0:
        raise ValueError(
            "Decay cannot be negative; "
            f"{np.sum(decay_constants < 0)} / {decay_constants.size} "
            " are negative values"
        )

    if decay_constant_maximum is not None:
        _decays = decay_constants.copy()
        _decays[_decays > decay_constant_maximum] = decay_constant_maximum
    else:
        _decays = decay_constants

    # dx/dt + \lambda * X = f(A)
    return InferelatorData(
        _sparse_safe_add(
            velocity,
            _sparse_safe_multiply(
                expression.values,
                _decays[None, :]
            )
        ),
        gene_names=expression.gene_names,
        sample_names=expression.sample_names,
        meta_data=expression.meta_data
    )


def _gene_variable_decay(
    expression,
    velocity,
    decay_constants,
    decay_constant_maximum=None
):

    """
    Calculate dX/dt + lambda * X
    where values of lambda can differ per gene

    Note that there is no check for negative decay

    :param expression: Expression data X ([M x N])
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt ([M x N])
    :type velocity: np.ndarray
    :param decay_constants: Decay constants ([M x N])
    :type decay_constants: np.ndarray
    :return: dX/dt + lambda * X
    :rtype: InferelatorData
    """

    if np.sum(decay_constants < 0) > 0:
        raise ValueError(
            "Decay cannot be negative; "
            f"{np.sum(decay_constants < 0)} / {decay_constants.size} "
            " are negative values"
        )

    if decay_constant_maximum is not None:
        _decay = decay_constants.copy()
        _decay[_decay > decay_constant_maximum] = decay_constant_maximum
    else:
        _decay = decay_constants

    # dx/dt + \lambda * X = f(A)
    return InferelatorData(
        _sparse_safe_add(
            velocity,
            _sparse_safe_multiply(
                expression.values,
                _decay
            )
        ),
        gene_names=expression.gene_names,
        sample_names=expression.sample_names,
        meta_data=expression.meta_data
    )


def _sparse_safe_multiply(x, y):
    """
    Sparse safe element-wise multiply

    :param x: Array
    :type x: np.ndarray, sp.spmatrix
    :param y: Array
    :type y: np.ndarray, sp.spmatrix
    :return: x * y
    :rtype: np.ndarray, sp.spmatrix
    """

    if sparse.isspmatrix(x):
        return x.multiply(y).tocsr()
    elif sparse.isspmatrix(y):
        return y.multiply(x).tocsr()
    else:
        return np.multiply(x, y)


def _sparse_safe_add(x, y):
    """
    Sparse safe element-wise add

    :param x: Array
    :type x: np.ndarray, sp.spmatrix
    :param y: Array
    :type y: np.ndarray, sp.spmatrix
    :return: x + y
    :rtype: np.ndarray
    """

    if sparse.isspmatrix(x) or sparse.isspmatrix(y):
        return (x + y).A
    else:
        return np.add(x, y)
