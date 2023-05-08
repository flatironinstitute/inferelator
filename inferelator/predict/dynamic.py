import numpy as np

from inferelator.predict.static import InferelatorStaticEstimator


class InferelatorDynamicEstimator(InferelatorStaticEstimator):
    """

    Inferelator Estimator for dynamic data.

    The inferelator has learned a model connecting gene
    expression to latent activity.

    InferelatorDynamicEstimator predicts time-dependent gene expression
    data from the latent activity space

    Parameters
    ----------
    model : anndata.AnnData, default=None
        Inferelator model object of shape (`n_genes`, `n_features)
    model_file : str, default=None
        Inferelator model object filename (.h5ad)
    tfa_model : inferelator.tfa.TFA, default=None
        The specific TFA model to use for calculations

    Attributes
    ----------
    model_ : anndata.AnnData of shape (`n_genes`, `n_features)
        Inferelator model object
    feature_names_ : pd.Index of shape (`n_features`, )
        Pandas index of feature (TF) names
    gene_names_ : pd.Index of shape (`n_genes`, )
        Pandas index of gene names
    """

    def predict(
        self,
        X,
        decay_constants,
        initial_state=None,
        step_time=1.,
        finite_difference=0.1
    ):
        """
        Apply inferelator model to predict gene expression from latent TFA
        features

        dX/dt = -\\lambda * X + alpha

        Parameters
        ----------
        X : array-like of shape (n_times, n_features)
            New TFA data, where `n_times` is the number of time steps
            and `n_features` is the number of features.

        decay_constants: array-like of shape (n_genes, ) or (n_times, n_genes)
            Decay constants at each time step, where time unit
            is the same as what was used to train the model.

        initial_state : array-like of shape (n_genes, )
            Initial gene expression state for t_0.
            Defaults to zeros.

        step time : float
            Duration of time steps between rows of X, where
            time unit is the same as what was used to train the model

        finite_difference : float
            Forward difference time

        Returns
        -------
        X_new : array-like of shape (n_times, n_genes)
            Predicted gene expression, where `n_times`
            is the number of time steps and `n_genes` is the number of
            genes in the model.
        """

        # Get number of time steps, features, and output genes
        _nt, _nf = X.shape
        _ng = self.coef_.shape[0]

        # Check to make sure the decay constants are shaped right
        if decay_constants.ndim == 2 and decay_constants.shape != (_nt, _ng):
            _shape_mismatch = True
        elif decay_constants.ndim == 1 and decay_constants.shape[0] != _ng:
            _shape_mismatch = True
        else:
            _shape_mismatch = False

        if _shape_mismatch:
            raise ValueError(
                f"decay_constants must be a 1d array ({_ng}, ) "
                f"or a 2d array with ({_nt}, {_ng}); "
                f"{decay_constants.shape} provided"
            )

        _duration = int(np.ceil(step_time * _nt))
        _steps = int(np.ceil(_duration / finite_difference))
        _step_landmark = int(step_time / finite_difference)

        # Output data object
        predicts = np.zeros(
            (_nt, _ng),
            dtype=float
        )

        _dX = np.zeros(
            (2, _ng),
            dtype=float
        )

        # Add initial state if provided
        # Otherwise it will be zeros
        if initial_state is not None:
            predicts[0, :] = initial_state.ravel()
            _dX[0, :] = predicts[0, :]

        for i in range(_steps):

            t = i * finite_difference
            _time_step_row = int(np.floor(t / step_time))

            if decay_constants is not None and decay_constants.ndim == 2:
                _step_decay_state = decay_constants[_time_step_row, :]
            elif decay_constants is not None and decay_constants.ndim == 1:
                _step_decay_state = decay_constants
            else:
                _step_decay_state = None

            # Positive component
            _dX[1, :] = np.maximum(
                super().predict(
                    X[_time_step_row, :]
                ),
                0
            ) * finite_difference

            # Negative component
            if _step_decay_state is not None:
                _dX[1, :] -= np.maximum(
                    np.multiply(
                        _step_decay_state,
                        _dX[0, :]
                    ),
                    0
                ) * finite_difference

            # Add changes to the last stepwise state
            _dX[0, :] += _dX[1, :]

            if i > 0 and i % _step_landmark == 0:
                predicts[_time_step_row, :] = _dX[0, :]

        return predicts
