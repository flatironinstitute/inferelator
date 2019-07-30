import pandas as pd
import numpy as np

from inferelator import utils
from inferelator import default
from inferelator.utils import Validator as check


class ManagePriors(object):
    """
    The ManagePriors class has the functions to manipulate prior and gold standard data which are called from workflow
    This filters, aligns, crossvalidates, shuffles, etc.
    """

    @staticmethod
    def validate_priors_gold_standard(priors_data, gold_standard):
        """
        Validate the priors and the gold standard, then pass them through

        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard data
        :return priors, gold_standard: pd.DataFrame [G x K], pd.DataFrame [G x K]
        """

        try:
            check.index_values_unique(priors_data.index)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate gene(s) in prior index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

        try:
            check.index_values_unique(priors_data.columns)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate tf(s) in prior index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

        try:
            check.index_values_unique(priors_data.index)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate gene(s) in gold standard index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

        try:
            check.index_values_unique(gold_standard.columns)
        except ValueError as v_err:
            utils.Debug.vprint("Duplicate tf(s) in gold standard index", level=0)
            utils.Debug.vprint(str(v_err), level=0)

        return priors_data, gold_standard

    @staticmethod
    def cross_validate_gold_standard(priors_data, gold_standard, cv_split_axis, cv_split_ratio, random_seed):
        """
        Sample the gold standard for crossvalidation, and then remove the new gold standard from the priors (if split
        on an axis)

        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param gold_standard: pd.DataFrame [G x K]
            Gold standard data
        :param cv_split_ratio: float
            The proportion of the priors that should go into the gold standard
        :param cv_split_axis: int
            Splits on rows (when 0), columns (when 1), or on flattened individual data points (when None)
            Note that if this is None, the returned gold standard will be the same as all_data, and the priors will have
            half of the data points of all_data
        :param random_seed: int
            Random seed
        :return priors_data, gold_standard: pd.DataFrame [G x K], pd.DataFrame [G x K]
        """

        assert check.argument_enum(cv_split_axis, (0, 1), allow_none=True)
        assert check.argument_numeric(cv_split_ratio, low=0, high=1)

        if cv_split_axis == 1:
            utils.Debug.vprint("Selecting cv_split_axis of 1 is possible but a very bad idea", level=1)

        utils.Debug.vprint("Resampling GS ({gs}) for crossvalidation".format(gs=gold_standard.shape), level=0)
        gs_to_prior, gold_standard = ManagePriors._split_for_cv(gold_standard, cv_split_ratio, split_axis=cv_split_axis,
                                                                seed=random_seed)

        # If the priors are split on an axis, remove circularity
        if cv_split_axis is not None:
            priors_data, gold_standard = ManagePriors._remove_prior_circularity(priors_data, gold_standard,
                                                                                split_axis=cv_split_axis)
        else:
            if priors_data is not None:
                utils.Debug.vprint("Existing prior is being replaced with a downsampled gold standard")
            priors_data = gs_to_prior

        utils.Debug.vprint("CV prior {pr} and gold standard {gs}".format(pr=priors_data.shape,
                                                                         gs=gold_standard.shape), level=0)

        return priors_data, gold_standard

    @staticmethod
    def filter_to_gene_list(priors_data, expression_matrix, gene_list):
        """
        Filter the expression matrix and priors to the intersection with a provided list of genes
        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param expression_matrix: pd.DataFrame [G x N]
            Expression data matrix
        :param gene_list: list
            List of genes to restrict the modeling to
        :return expression_matrix: pd.DataFrame [g x N]
        """

        utils.Debug.vprint("Filtering expression and priors to {le} genes from list".format(le=len(gene_list)), level=1)
        expression_matrix = expression_matrix.loc[expression_matrix.index.intersection(gene_list), :]
        utils.Debug.vprint("Expression data filtered to {sh}".format(sh=expression_matrix.shape), level=1)
        priors_data = priors_data.loc[priors_data.index.intersection(gene_list), :]
        utils.Debug.vprint("Priors data filtered to {sh}".format(sh=priors_data.shape), level=1)

        if expression_matrix.shape[0] == 0:
            raise ValueError("Expression matrix genes and gene list genes have no overlap")

        if priors_data.shape[0] == 0:
            raise ValueError("Prior genes and gene list genes have no overlap")

        return priors_data, expression_matrix

    @staticmethod
    def filter_to_tf_names_list(priors_data, tf_names):
        """
        Filter the priors the intersection with a provided list of regulators
        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param tf_names: list [k]
            List of regulators to restrict the modeling to
        :return priors_data: pd.DataFrame [G x k]
            Filtered priors on regulators
        """

        tf_keepers = pd.Index(tf_names).intersection(pd.Index(priors_data.columns))
        priors_data = priors_data.loc[:, tf_keepers]

        if priors_data.shape[1] == 0:
            raise ValueError("Prior regulators and regulator list regulators have no overlap")

        return priors_data

    @staticmethod
    def align_priors_to_expression(priors_data, expression_matrix):
        """
        Make sure that the priors align to the expression matrix and fill priors that are created with 0s
        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param expression_matrix: pd.DataFrame [G x N]
            Expression matrix data
        :return priors_data:
            Returns priors_data where genes match expression matrix genes
        """

        if len(priors_data.index.intersection(expression_matrix.index)) == 0:
            raise ValueError("Prior genes and expression matrix genes have no overlap")

        return priors_data.reindex(index=expression_matrix.index).fillna(value=0)

    @staticmethod
    def shuffle_priors(priors_data, shuffle_prior_axis, random_seed):
        """
        Shuffle the labels on the priors on a specific axis
        :param priors_data: pd.DataFrame [G x K]
            Prior data
        :param shuffle_prior_axis: int
            Axis to shuffle. 0 is genes, 1 is regulators, None is skip shuffling.
        :param random_seed: int
            Random seed
        :return priors_data:
            Returns priors_data the data has been shuffled on a specific axis
        """

        assert check.argument_enum(shuffle_prior_axis, [0, 1], allow_none=True)

        if shuffle_prior_axis is None:
            return priors_data
        elif shuffle_prior_axis == 0:
            # Shuffle index (genes) in the priors_data
            utils.Debug.vprint("Randomly shuffling prior [{sh}] gene data".format(sh=priors_data.shape))
            prior_index = priors_data.index.tolist()
            priors_data = priors_data.sample(frac=1, axis=0, random_state=random_seed)
            priors_data.index = prior_index
        elif shuffle_prior_axis == 1:
            # Shuffle columns (TFs) in the priors_data
            utils.Debug.vprint("Randomly shuffling prior [{sh}] TF data".format(sh=priors_data.shape))
            prior_index = priors_data.columns.tolist()
            priors_data = priors_data.sample(frac=1, axis=1, random_state=random_seed)
            priors_data.columns = prior_index

        return priors_data

    @staticmethod
    def _split_for_cv(all_data, split_ratio, split_axis=default.DEFAULT_CV_AXIS, seed=default.DEFAULT_CV_RANDOM_SEED):
        """
        Take a dataframe and split it according to split_ratio on split_axis into two new dataframes. This is for
        crossvalidation splits of a gold standard.

        :param all_data: pd.DataFrame [G x K]
            Existing prior or gold standard data
        :param split_ratio: float
            The proportion of the priors that should go into the gold standard
        :param split_axis: int
            Splits on rows (when 0), columns (when 1), or on flattened individual data points (when None)
            Note that if this is None, the returned gold standard will be the same as all_data, and the priors will have
            half of the data points of all_data
        :param seed: int
            Seed for the random generator
        :return prior_data, gold_standard: pd.DataFrame [G/2 x K], pd.DataFrame [G/2 x K]
            Returns a new prior and gold standard by splitting the old one in half
        """

        assert check.argument_numeric(split_ratio, 0, 1)
        assert check.argument_enum(split_axis, [0, 1], allow_none=True)

        # Split the priors into gold standard based on axis (flatten if axis=None)
        if split_axis is None:
            priors_data, _ = ManagePriors._split_flattened(all_data, split_ratio, seed=seed)
            gold_standard = all_data
        else:
            priors_data, gold_standard = ManagePriors._split_axis(all_data, split_ratio, axis=split_axis, seed=seed)

        return priors_data, gold_standard

    @staticmethod
    def _remove_prior_circularity(priors, gold_standard, split_axis=default.DEFAULT_CV_AXIS):
        """
        Remove all axis labels that occur in the gold standard from the prior
        :param priors: pd.DataFrame [M x N]
        :param gold_standard: pd.DataFrame [m x n]
        :param split_axis: int (0,1)
        :return new_priors: pd.DataFrame [M-m x N]
        :return gold_standard: pd.DataFrame [m x n]
        """

        assert check.argument_enum(split_axis, [0, 1])

        new_priors = priors.drop(gold_standard.axes[split_axis], axis=split_axis, errors='ignore')

        return new_priors, gold_standard

    @staticmethod
    def _split_flattened(data, split_ratio, seed=default.DEFAULT_CV_RANDOM_SEED):
        """
        Instead of splitting by axis labels, split edges and ignore axes
        :param data: pd.DataFrame [M x N]
        :param split_ratio: float
        :param seed:
        :return priors_data: pd.DataFrame [M x N]
        :return gold_standard: pd.DataFrame [M x N]
        """

        assert check.argument_numeric(split_ratio, 0, 1)

        pc = np.sum(data.values != 0)
        gs_count = int(split_ratio * pc)
        idx = ManagePriors._make_shuffled_index(pc, seed=seed)

        pr_idx = data.values[data.values != 0].copy()
        gs_idx = data.values[data.values != 0].copy()

        pr_idx[idx[0:gs_count]] = 0
        gs_idx[idx[gs_count:]] = 0

        gs = data.values.copy()
        pr = data.values.copy()

        gs[gs != 0] = gs_idx
        pr[pr != 0] = pr_idx

        priors_data = pd.DataFrame(pr, index=data.index, columns=data.columns)
        gold_standard = pd.DataFrame(gs, index=data.index, columns=data.columns)

        return priors_data, gold_standard

    @staticmethod
    def _split_axis(priors, split_ratio, axis=default.DEFAULT_CV_AXIS, seed=default.DEFAULT_CV_RANDOM_SEED):
        """
        Split by axis labels on the chosen axis
        :param priors: pd.DataFrame [M x N]
        :param split_ratio: float
        :param axis: [0, 1]
        :param seed:
        :return:
        """

        assert check.argument_numeric(split_ratio, 0, 1)
        assert check.argument_enum(axis, [0, 1])

        pc = priors.shape[axis]
        gs_count = int((1 - split_ratio) * pc)
        idx = ManagePriors._make_shuffled_index(pc, seed=seed)

        if axis == 0:
            axis_idx = priors.index
        elif axis == 1:
            axis_idx = priors.columns
        else:
            raise ValueError("Axis can only be 0 or 1")

        pr_idx = axis_idx[idx[0:gs_count]]
        gs_idx = axis_idx[idx[gs_count:]]

        priors_data = priors.drop(gs_idx, axis=axis)
        gold_standard = priors.drop(pr_idx, axis=axis)

        return priors_data, gold_standard

    @staticmethod
    def _make_shuffled_index(idx_len, seed=default.DEFAULT_CV_RANDOM_SEED):
        idx = list(range(idx_len))
        np.random.RandomState(seed=seed).shuffle(idx)
        return idx
