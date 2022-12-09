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

    assert check.indexes_align(
        (expression.gene_names, velocity.gene_names)
    )

    assert check.indexes_align(
        (expression.sample_names, velocity.sample_names)
    )

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

    elif gene_and_sample_decay is not None:

        assert check.indexes_align(
            (
                expression.sample_names,
                gene_specific_decay.sample_names
            )
        )

        Debug.vprint(
            "Modeling transcription with velocity and decay per sample and per gene"
        )

        return _gene_variable_decay(
            expression,
            velocity,
            gene_and_sample_decay
        )

    else:
        Debug.vprint(
            "Modeling transcription with velocity only"
        )

        return InferelatorData(
            velocity.values,
            gene_names=expression.gene_names,
            sample_names=expression.sample_names,
            meta_data=expression.meta_data
        )



def _global_decay(
    expression,
    velocity,
    constant
):

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
