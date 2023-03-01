from sklearn.base import BaseEstimator
import anndata as ad
import numpy as np

from inferelator.utils import DotProduct
from inferelator.tfa import ActivityOnlyPinvTFA
from inferelator.preprocessing import PreprocessData


class InferelatorStaticEstimator(BaseEstimator):
    """

    Inferelator Estimator for static data.

    The inferelator has learned a model connecting gene
    expression to latent activity.

    InferelatorStaticEstimator transforms gene expression
    data into the latent activity space, and predicts
    gene expression data from the latent activity space

    Parameters
    ----------
    model : anndata.AnnData, default=None
        Inferelator model object of shape (`n_genes`, `n_features)
        or a path to a model object file that can be loaded
    tfa_model : inferelator.tfa.TFA, default=None
        The specific TFA model to use for calculations

    Attributes
    ----------
    model : anndata.AnnData of shape (`n_genes`, `n_features)
        Inferelator model object
    feature_names_ : pd.Index of shape (`n_features`, )
        Pandas index of feature (TF) names
    gene_names_ : pd.Index of shape (`n_genes`, )
        Pandas index of gene names
    """

    full_model_ = None
    current_model_ = None

    @property
    def model(self):
        if self.current_model_ is not None:
            return self.current_model_
        else:
            return self.full_model_

    @model.setter
    def model(self, model):
        self.full_model_ = model
        self.current_model_ = model

    def __init__(
            self,
            model,
            tfa_model=None
    ):

        if isinstance(model, ad.AnnData):
            self.model = model.copy()
        else:
            self.model = ad.read(model)

        self._extract_model_values()
        PreprocessData.set_preprocessing_method(
            **self.model.uns['preprocessing']
        )

        if tfa_model is not None:
            self.tfa_model = tfa_model
        else:
            self.tfa_model = ActivityOnlyPinvTFA

        super().__init__()

    def fit(self, X, y=None):
        """
        Model must be fit with the inferelator workflow

        Parameters
        ----------
        X : Ignored
            Ignored.
        y : Ignored
            Ignored.

        Returns
        -------
        None

        Notes
        -----
        Don't call this, it's not implemented here
        """
        raise NotImplementedError(
            "Fit model with the full inferelator package"
        )

    def predict(self, X):
        """
        Apply inferelator model to predict gene expression from latent TFA
        features

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New TFA data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_genes)
            Predicted gene expression, where `n_samples`
            is the number of samples and `n_genes` is the number of
            genes in the model.
        """

        return DotProduct.dot(X, self.coef_.T)

    def transform(self, X):
        """
        Apply inferelator model to transform gene expression
        into TFA features

        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data, where `n_samples` is the number of samples
            and `n_genes` is the number of genes in the model.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
            Transformed latent TFA, where `n_samples`
            is the number of samples and `n_features` is the number of
            TF features in the model.
        """

        return self.tfa_model._calculate_activity(
            self.coef_,
            X
        )

    def transform_predict(self, X):
        """
        Transform gene expression into TFA features
        and then use those features to predict gene
        expression

        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data, where `n_samples` is the number of samples
            and `n_genes` is the number of genes in the model.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_genes)
            Gene expression data, where `n_samples` is the number of samples
            and `n_genes` is the number of genes in the model.
        """

        return self.predict(
            self.transform(
                X
            )
        )

    def trim(
        self,
        genes=None,
        tfs=None
    ):
        """
        Trim the predictive model to a specific set of genes or
        features. Reference to the full model is retained and so
        this function can be repeatedly called with nonoverlapping
        sets of labels.

        Parameters
        ----------
        genes : array-like of shape (genes, )
            Genes to keep in the model. Will raise a KeyError if any
            genes are provided that do not exist in the model

        tfs : array-like of shape (features, )
            TF features to keep in the model. Will raise a KeyError if any
            TFs are provided that do not exist in the model

        Returns
        -------
        self
        """

        # Return self if nothing is passed in
        if genes is None and tfs is None:
            return self

        gene_idxer, tf_idxer = None, None

        # Get reindexers for current model
        try:
            gene_idxer, tf_idxer = self._get_reindexers(
                genes,
                tfs
            )
        except KeyError:
            pass

        # Get reindexers for the full model if the
        # current model didn't work out
        if gene_idxer is None and tf_idxer is None:
            _cm = self.current_model_
            self.current_model_ = self.full_model_

            try:
                gene_idxer, tf_idxer = self._get_reindexers(
                    genes,
                    tfs
                )
            except KeyError:
                self.current_model_ = _cm
                raise

        self._trim_modelaxis(
            gene_idxer,
            tf_idxer
        )

        return self

    def _extract_model_values(self):
        self.coef_ = self.model.X.copy()
        self.feature_names_ = self.model.var_names.copy()
        self.gene_names_ = self.model.obs_names.copy()

    def _get_reindexers(
        self,
        genes,
        tfs
    ):
        if genes is not None:
            gene_idxer = self._check_reindex(
                genes,
                self._get_reindexer(
                    genes,
                    axis=0
                ),
                0
            )
        else:
            gene_idxer = None

        if tfs is not None:
            tf_idxer = self._check_reindex(
                tfs,
                self._get_reindexer(
                    tfs,
                    axis=1
                ),
                1
            )
        else:
            tf_idxer = None

        return gene_idxer, tf_idxer

    def _get_reindexer(
        self,
        new_labels,
        axis
    ):
        if axis == 0:
            return self.model.obs_names.get_indexer(new_labels)
        elif axis == 1:
            return self.model.var_names.get_indexer(new_labels)
        else:
            raise ValueError(
                f"axis must be 0 or 1; {axis} provided"
            )

    def _trim_modelaxis(
        self,
        axis_0_indexer,
        axis_1_indexer
    ):
        """
        Trim model with reindexers
        """

        model = self.model

        if axis_0_indexer is not None and axis_1_indexer is not None:
            self.current_model_ = model[axis_0_indexer, :][:, axis_1_indexer]
        elif axis_0_indexer is not None:
            self.current_model_ = model[axis_0_indexer, :]
        elif axis_1_indexer is not None:
            self.current_model_ = model[:, axis_1_indexer]

        self.current_model_ = self.current_model_.copy()
        self._extract_model_values()

    @staticmethod
    def _check_reindex(
        new_labels,
        new_indexer,
        axis
    ):
        """
        Make sure there are no missing labels
        """

        if np.any(new_indexer == -1):
            raise KeyError(
                f"All labels must be present in the model on axis {axis}: "
                f"{np.sum(new_indexer == -1)} / {len(new_labels)} are missing"
            )

        return new_indexer
