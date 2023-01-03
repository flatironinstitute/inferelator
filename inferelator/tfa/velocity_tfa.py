from inferelator.preprocessing.tfa import TFA
from inferelator.utils import (
    InferelatorData,
    Debug,
    Validator as check
)

import numpy as np


class VelocityTFA(TFA):

    def compute_transcription_factor_activity(
        self,
        prior,
        expression_data,
        expression_data_halftau=None,
        keep_self=False,
        tau=None
    ):

        prior, activity_tfs, expr_tfs = self._check_prior(
            prior,
            expression_data,
            keep_self=keep_self
        )

        if len(activity_tfs) > 0:
            activity = self._calculate_activity(
                prior.loc[:, activity_tfs].values,
                expression_data.values
            )
        else:
            raise ValueError(
                "TFA cannot be calculated; prior matrix has no edges"
            )

        return InferelatorData(
            activity,
            gene_names=activity_tfs,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data
        )


def modify_velocity_with_decay(
    velocity_data,
    expression_data,
    constant_decay=None,
    variable_decay=None,
    decay_per_observation=None
):
    """
    Return \alpha = (dX/dt + lambda * X)
    Takes constant lambda (constant_decay),
    lambda that is different for each gene (variable_decay),
    or lambda that is different for each gene and
    observation (decay_per_observation).

    :param velocity_data: Velocity data
    :type velocity_data: InferelatorData
    :param expression_data: Expression data
    :type expression_data: InferelatorData
    :param constant_decay: Constant decay value to be applied to every gene,
        defaults to None
    :type constant_decay: float, optional
    :param variable_decay: Decay constants for every gene,
        defaults to None
    :type variable_decay: np.ndarray, optional
    :param decay_per_observation: Decay constants for every gene and
        observation, defaults to None
    :type decay_per_observation: np.ndarray, optional
    :raises ValueError: Raises if no decay values are passed
    :return: \alpha matrix
    :rtype: InferelatorData
    """

    if constant_decay is not None:
        Debug.vprint(
            "Modeling TFA on fixed decay constant "
            f"{constant_decay} for every gene"
        )

        return InferelatorData(
            np.add(
                velocity_data.values,
                np.multiply(
                    expression_data.values,
                    constant_decay
                )
            ),
            gene_names=expression_data.gene_names,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            name=expression_data.name
        )

    elif variable_decay is not None:

        return InferelatorData(
            np.add(
                velocity_data.values,
                np.multiply(
                    expression_data.values,
                    variable_decay[None, :]
                )
            ),
            gene_names=expression_data.gene_names,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            name=expression_data.name
        )

    elif decay_per_observation is not None:
        Debug.vprint(
            "Modeling TFA on velocity and decay "
            "per sample and per gene"
        )

        assert check.indexes_align(
            (
                expression_data.sample_names,
                decay_per_observation.sample_names
            )
        )

        return InferelatorData(
            np.add(
                velocity_data.values,
                np.multiply(
                    expression_data.values,
                    decay_per_observation.values
                )
            ),
            gene_names=expression_data.gene_names,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            name=expression_data.name
        )

    else:

        raise ValueError(
            "Pass one of constant_decay, "
            "variable_decay, or "
            "decay_per_observation"
        )
