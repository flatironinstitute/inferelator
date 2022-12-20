import numpy as np

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
    gene_and_sample_decay=None
):
    """
    Extract the transcriptional output
    by combining expression and velocity

    :param expression: Expression data X
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt
    :type velocity: InferelatorData
    :param global_decay: Decay constant to use for
        every gene and observation, defaults to None
    :type global_decay: float, optional
    :param gene_specific_decay: Decay constants that differ
        for every gene but are the same for every observation, defaults to None
    :type gene_specific_decay: pd.DataFrame, optional
    :param gene_and_sample_decay: Decay constants that differ
        for every gene and differ for every observation, defaults to None
    :type gene_and_sample_decay: InferelatorData, optional
    :returns: Transcriptional output estimates
    :rtype: InferelatorData
    """

    # Check that the data objects passed have the same
    # dimensions & labels
    assert check.indexes_align(
        (expression.gene_names, velocity.gene_names)
    )

    assert check.indexes_align(
        (expression.sample_names, velocity.sample_names)
    )

    # Use the same decay constant for every gene and observation
    if global_decay is not None:

        Debug.vprint(
            "Modeling transcription with fixed decay constant "
            f"{global_decay} for every gene"
        )

        return _global_decay(
            expression,
            velocity,
            global_decay
        )

    # Use the same decay constant for every observation
    # but a different decay constant for every gene
    elif gene_specific_decay is not None:

        assert check.indexes_align(
            (
                expression.gene_names,
                gene_specific_decay.index
            )
        )

        Debug.vprint(
            "Modeling transcription with velocity and decay per gene"
        )

        return _gene_constant_decay(
            expression,
            velocity,
            gene_specific_decay
        )

    # Use a different decay constant for every gene and observation
    # Decay constants must be a [M x N] data object
    elif gene_and_sample_decay is not None:

        assert check.indexes_align(
            (
                expression.sample_names,
                gene_and_sample_decay.sample_names
            )
        )

        Debug.vprint(
            "Modeling transcription with velocity and decay "
            "per sample and per gene"
        )

        return _gene_variable_decay(
            expression,
            velocity,
            gene_and_sample_decay
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
    :type velocity: InferelatorData
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
        np.add(
            velocity.values,
            np.multiply(
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
    decay_constants
):
    """
    Calculate dX/dt + lambda * X
    where values of lambda can differ per gene

    :param expression: Expression data X ([M x N])
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt ([M x N])
    :type velocity: InferelatorData
    :param decay_constants: Decay constant to use ([N x 1])
    :type decay_constants: pd.DataFrame
    :return: dX/dt + lambda * X
    :rtype: InferelatorData
    """

    if np.sum(decay_constants.values < 0) > 0:
        raise ValueError(
            "Decay cannot be negative; "
            f"{np.sum(decay_constants.values < 0)} / {decay_constants.size} "
            " are negative values"
        )

    # dx/dt + \lambda * X = f(A)
    return InferelatorData(
        np.add(
            velocity.values,
            np.multiply(
                expression.values,
                decay_constants.values.flatten()[None, :]
            )
        ),
        gene_names=expression.gene_names,
        sample_names=expression.sample_names,
        meta_data=expression.meta_data
    )


def _gene_variable_decay(
    expression,
    velocity,
    decay_constants
):

    """
    Calculate dX/dt + lambda * X
    where values of lambda can differ per gene

    Note that there is no check for negative decay

    :param expression: Expression data X ([M x N])
    :type expression: InferelatorData
    :param velocity: Velocity data dX/dt ([M x N])
    :type velocity: InferelatorData
    :param decay_constants: Decay constants ([M x N])
    :type decay_constants: InferelatorData
    :return: dX/dt + lambda * X
    :rtype: InferelatorData
    """

    if np.sum(decay_constants.values < 0) > 0:
        raise ValueError(
            "Decay cannot be negative; "
            f"{np.sum(decay_constants.values < 0)} / {decay_constants.size} "
            " are negative values"
        )

    # dx/dt + \lambda * X = f(A)
    return InferelatorData(
        np.add(
            velocity.values,
            np.multiply(
                expression.values,
                decay_constants.values
            )
        ),
        gene_names=expression.gene_names,
        sample_names=expression.sample_names,
        meta_data=expression.meta_data
    )
