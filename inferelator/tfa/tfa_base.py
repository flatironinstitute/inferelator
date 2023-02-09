from abc import abstractmethod
import numpy as np

from inferelator.utils import (
    InferelatorData,
    Debug,
    df_set_diag
)

from inferelator.preprocessing import (
    PreprocessData
)

class TFABase:
    """
    TFA calculates transcription factor activity
    """

    def compute_transcription_factor_activity(
        self,
        prior,
        expression_data,
        expression_data_halftau=None,
        keep_self=False,
        tau=None
    ):

        raise NotImplementedError

    def _check_prior(
        self,
        prior,
        expression_data,
        keep_self=False,
        use_expression=True
    ):

        if not keep_self:
            prior = df_set_diag(prior, 0)

        activity_tfs, expr_tfs, drop_tfs = self._determine_tf_status(
            prior,
            expression_data
        )

        if not use_expression:
            drop_tfs = drop_tfs.append(expr_tfs)

        if len(drop_tfs) > 0:
            Debug.vprint(
                f"{len(drop_tfs)} TFs are removed from activity "
                f"as they cannot be estimated",
                level=0
            )
            Debug.vprint(" ".join(drop_tfs), level=1)

        prior = prior.drop(drop_tfs, axis=1)

        return prior, activity_tfs, expr_tfs

    @staticmethod
    def _determine_tf_status(prior, expression_data):

        # These TFs can have activity calculations performed
        # for them because there are priors
        _activity_tfs = ((prior != 0).sum(axis=0) != 0).values

        # These TFs match gene expression only (no activity calculation)
        _expr_tfs = prior.columns.isin(expression_data.gene_names)
        _expr_tfs &= ~_activity_tfs

        activity_names = prior.columns[_activity_tfs]
        expr_names = prior.columns[_expr_tfs]
        drop_names = prior.columns[~(_activity_tfs | _expr_tfs)]

        return activity_names, expr_names, drop_names

    @staticmethod
    @abstractmethod
    def _calculate_activity(
        prior,
        expression_data
    ):
        """
        Calculate activity from prior and expression data

        :param prior: Prior knowledge matrix
        :type prior: np.ndarray
        :param expression_data: Gene expression data,
            already normalized
        :type expression_data: np.ndarray, sp.spmatrix
        """

        raise NotImplementedError


class ActivityExpressionTFA(TFABase):

    def compute_transcription_factor_activity(
        self,
        prior,
        expression_data,
        expression_data_halftau=None,
        keep_self=False
    ):
        """
        Calculate TFA from a prior and expression data object

        :param prior: pd.DataFrame [G x K]
        :param expression_data: InferelatorData [N x G]
        :param expression_data_halftau: InferelatorData [N x G]
        :param keep_self: bool
        :return: InferelatorData [N x K]
        """

        # Figure out which features have enough prior network
        # data to calculate activity and which do not

        trim_prior, activity_tfs, expr_tfs = self._check_prior(
            prior,
            expression_data,
            keep_self=keep_self,
            use_expression=True
        )

        # Make empty array to fill with activity
        activity = np.zeros(
            (expression_data.shape[0], trim_prior.shape[1]),
            dtype=np.float64
        )

        # Calculate activity if there are any features which
        # support activity calculation
        if len(activity_tfs) > 0:
            a_cols = trim_prior.columns.isin(activity_tfs)

            if expression_data_halftau is not None:
                expr = expression_data_halftau
            else:
                expr = expression_data

            activity[:, a_cols] = self._calculate_activity(
                trim_prior.loc[:, activity_tfs].values,
                PreprocessData.preprocess_expression_array(
                    expr.values
                )
            )

        # Use TF expression in place of activity for features
        # which don't have activity
        if len(expr_tfs) > 0:
            e_cols = trim_prior.columns.isin(expr_tfs)
            activity[:, e_cols] = PreprocessData.preprocess_expression_array(
                expression_data.get_gene_data(
                    expr_tfs,
                    force_dense=True
                )
            )

        if expression_data.name is None:
            _data_name = "Activity"
        else:
            _data_name = f"{expression_data.name} Activity"

        activity = InferelatorData(
            activity,
            gene_names=trim_prior.columns,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            name=_data_name
        )

        activity.prior_data = prior.copy()
        activity.tfa_prior_data = trim_prior.loc[:, activity_tfs].copy()

        return activity


class ActivityOnlyTFA(TFABase):

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
            keep_self=keep_self,
            use_expression=False
        )

        if len(activity_tfs) > 0:
            activity = self._calculate_activity(
                prior.loc[:, activity_tfs].values,
                PreprocessData.preprocess_expression_array(
                    expression_data.values
                )
            )
        else:
            raise ValueError(
                "TFA cannot be calculated; prior matrix has no edges"
            )

        if expression_data.name is None:
            _data_name = "Activity"
        else:
            _data_name = f"{expression_data.name} Activity"

        activity = InferelatorData(
            activity,
            gene_names=activity_tfs,
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            name=_data_name
        )

        activity.prior_data = prior.copy()
        activity.tfa_prior_data = prior.loc[:, activity_tfs].copy()

        return activity


class NoTFA(TFABase):
    """ NoTFA creates an activity matrix from the expression data only """

    def compute_transcription_factor_activity(
        self,
        prior,
        expression_data,
        expression_data_halftau=None,
        keep_self=False
    ):

        Debug.vprint(
            "Setting Activity to Expression Values",
            level=1
        )

        tf_gene_overlap = prior.columns[
            prior.columns.isin(expression_data.gene_names)
        ]

        if expression_data.name is None:
            _data_name = "Activity"
        else:
            _data_name = f"{expression_data.name} Activity"

        return InferelatorData(
            PreprocessData.preprocess_expression_array(
                expression_data.get_gene_data(
                    tf_gene_overlap,
                    force_dense=True
                )
            ),
            sample_names=expression_data.sample_names,
            meta_data=expression_data.meta_data,
            gene_names=tf_gene_overlap,
            name=_data_name
        )
