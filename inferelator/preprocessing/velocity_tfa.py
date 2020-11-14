from inferelator.preprocessing.tfa import TFA
from inferelator.utils import InferelatorData


class VelocityTFA(TFA):

    def compute_transcription_factor_activity(self, prior, expression_data, expression_data_halftau=None,
                                              keep_self=False, tau=None):

        prior, activity_tfs, expr_tfs = self._check_prior(prior, expression_data, keep_self=keep_self)

        if len(activity_tfs) > 0:
            activity = self._calculate_activity(prior.loc[:, activity_tfs].values, expression_data)
        else:
            raise ValueError("TFA cannot be calculated; prior matrix has no edges")

        return InferelatorData(activity, gene_names=activity_tfs, sample_names=expression_data.sample_names,
                               meta_data=expression_data.meta_data)


