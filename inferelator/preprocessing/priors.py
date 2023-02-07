import pandas as pd
import numpy as np
import warnings

from inferelator.utils import (
    Validator as check,
    Debug
)

DEFAULT_CV_AXIS = 0
DEFAULT_SEED = 2001


class ManagePriors:
    """
    The ManagePriors class has the functions to manipulate prior
    and gold standard data which are called from workflow
    This filters, aligns, crossvalidates, shuffles, etc.
    """

    @staticmethod
    def validate_priors_gold_standard(
        priors_data,
        gold_standard
    ):
        """
        Validate the priors and the gold standard, then pass them through

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param gold_standard: Gold standard network data [G x K]
        :type gold_standard: pd.DataFrame
        :return priors, gold_standard: Prior network data [G x K],
            Gold standard network data [G x K]
        :rtype: pd.DataFrame, pd.DataFrame
        """

        try:
            check.index_values_unique(priors_data.index)
        except ValueError as v_err:
            Debug.vprint(
                "Duplicate gene(s) in prior index: " + str(v_err),
                level=0
            )

        try:
            check.index_values_unique(priors_data.columns)
        except ValueError as v_err:
            Debug.vprint(
                "Duplicate tf(s) in prior index: " + str(v_err),
                level=0
            )

        try:
            check.index_values_unique(gold_standard.index)
        except ValueError as v_err:
            Debug.vprint(
                "Duplicate gene(s) in gold standard index: " + str(v_err),
                level=0
            )

        try:
            check.index_values_unique(gold_standard.columns)
        except ValueError as v_err:
            Debug.vprint(
                "Duplicate tf(s) in gold standard index" + str(v_err),
                level=0
            )

        return priors_data, gold_standard

    @staticmethod
    def cross_validate_gold_standard(
        priors_data,
        gold_standard,
        cv_split_axis,
        cv_split_ratio,
        random_seed
    ):
        """
        Sample the gold standard for crossvalidation, and then remove the
        new gold standard from the priors (if split on an axis)

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param gold_standard: Gold standard network data [G x K]
        :type gold_standard: pd.DataFrame
        :param cv_split_ratio: The proportion of the priors that should go
            into the gold standard
        :type cv_split_ratio: float
        :param cv_split_axis: Splits on rows (when 0),
            columns (when 1),
            or on flattened individual data points (when None)
            Note that if this is None, the returned gold standard will be
            unchanged, and the priors will have half of the network edges in
            the gold standard. If a different prior network has been passed in
            it will be discarded
        :type cv_split_axis: int, None
        :param random_seed: Random seed
        :type random_seed: int
        :return priors, gold_standard: Prior network data [G x K],
            Gold standard network data [G x K]
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.argument_enum(
            cv_split_axis,
            (0, 1),
            allow_none=True
        )
        assert check.argument_numeric(
            cv_split_ratio,
            low=0,
            high=1
        )

        _gs_shape = gold_standard.shape

        if cv_split_axis == 1:
            warnings.warn(
                "Setting a cv_split_axis of 1 means TFs in the gold standard "
                "will have no prior network knowledge for activity. "
                "This is not advisable."
            )

        gs_to_prior, gold_standard = ManagePriors._split_for_cv(
            gold_standard,
            cv_split_ratio,
            split_axis=cv_split_axis,
            seed=random_seed
        )

        # If the priors are split on an axis, remove circularity from prior
        if cv_split_axis is not None:

            priors_data, gold_standard = ManagePriors._remove_prior_circularity(
                priors_data,
                gold_standard,
                split_axis=cv_split_axis
            )

        else:
            if priors_data is not None:
                warnings.warn(
                    "Existing prior is being replaced with a "
                    "downsampled gold standard "
                    "because cv_split_axis == None"
                )

            priors_data = gs_to_prior

        Debug.vprint(
            f"Gold standard {_gs_shape} split on axis {cv_split_axis}. "
            f"Prior knowledge network {priors_data.shape} "
            f"[{np.sum(np.sum(priors_data != 0))} edges] used for activity "
            f"and gold standard network {gold_standard.shape} "
            f"[{np.sum(np.sum(gold_standard != 0))} edges] used for testing.",
            level=0
        )

        return priors_data, gold_standard

    @staticmethod
    def filter_priors_to_genes(
        priors_data,
        gene_list
    ):
        """
        Filter a prior network dataframe to a gene index

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param gene_list: List or index with genes to filter to
        :type gene_list: list, pd.Index
        :return priors: Prior network data [G x K] filtered on
            targets
        :rtype: pd.DataFrame
        """

        if len(gene_list) == 0:
            raise ValueError(
                "Filtering to a list of 0 genes is not valid"
            )

        if len(priors_data.index) == 0:
            raise ValueError(
                "Filtering a prior matrix of 0 genes is not valid"
            )

        try:
            priors_data = ManagePriors._filter_df_index(
                priors_data,
                gene_list
            )
        except ValueError as err:
            raise ValueError(
                f"{str(err)} when filtering priors for gene list. "
                f"Prior matrix genes (e.g. {str(priors_data.index[0])}) "
                f"and gene list genes (e.g. {str(gene_list[0])}) are "
                "non-overlapping"
            )

        return priors_data

    @staticmethod
    def filter_to_tf_names_list(
        priors_data,
        tf_names
    ):
        """
        Filter the priors to a provided list of regulators

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param tf_names: List or index with tfs (columns) to filter to
        :type tf_names: list, pd.Index
        :return priors: Prior network data [G x K] filtered on
            regulators
        :rtype: pd.DataFrame
        """

        if len(tf_names) == 0:
            raise ValueError(
                "Filtering to a list of 0 TFs is not valid"
            )

        if len(priors_data.columns) == 0:
            raise ValueError(
                "Filtering a prior matrix of 0 TFs is not valid"
            )

        try:
            priors_data = ManagePriors._filter_df_index(
                priors_data,
                tf_names,
                axis=1
            )
        except ValueError as err:
            raise ValueError(
                f"{str(err)} when filtering priors for TF list. "
                f"Prior matrix TFs (e.g. {str(priors_data.columns[0])}) "
                f"and TF list genes (e.g. {str(tf_names[0])}) are "
                "non-overlapping"
            )

        Debug.vprint(
            f"Filtered prior to {priors_data.shape[1]} TFs from the "
            "TF name list",
            level=1
        )

        return priors_data

    @staticmethod
    def _filter_df_index(
        data_frame,
        index_list,
        axis=0
    ):

        if axis == 0:
            new_index = data_frame.index.intersection(
                index_list
            )
        elif axis == 1:
            new_index = data_frame.columns.intersection(
                index_list
            )
        else:
            raise ValueError(
                "_filter_df_index axis must be 0 or 1"
            )

        if len(new_index) == 0:
            raise ValueError(
                "Filtering results in 0-length index"
            )

        return data_frame.reindex(
            new_index,
            axis=axis
        )

    @staticmethod
    def align_priors_to_expression(
        priors_data,
        gene_list
    ):
        """
        Make sure that the priors align to the expression matrix and fill
        priors that are created with 0s

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param gene_list: List or index with genes to align to
        :type gene_list: list, pd.Index
        :return priors: Prior network data [G x K] aligned on genes and
            filled out with 0s
        :rtype: pd.DataFrame
        """

        # Filter to overlap and raise an error if there's no overlap
        try:
            priors_data = ManagePriors._filter_df_index(
                priors_data,
                gene_list
            )
        except ValueError as err:
            raise ValueError(
                f"{str(err)} when aligning priors to expression data. "
                f"Prior matrix genes (e.g. {str(priors_data.index[0])}) "
                f"and expression data genes (e.g. {str(gene_list[0])}) are "
                "non-overlapping"
            )

        # Reindex to align to the full expression gene index
        # and fill out with zeros
        return priors_data.reindex(
            index=gene_list
        ).fillna(value=0)

    @staticmethod
    def shuffle_priors(
        priors_data,
        shuffle_prior_axis,
        random_seed
    ):
        """
        Shuffle the labels on the priors on a specific axis
        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param shuffle_prior_axis: Prior axis to shuffle.
            0 is genes,
            1 is regulators,
            -1 is to shuffle both axes.
            None is skip shuffling.
        :type shuffle_prior_axis: int
        :param random_seed: Random seed
        :type random_seed: int
        :return priors: Prior network data [G x K] shuffled on
            one or both axes
        :rtype: pd.DataFrame
        """

        assert check.argument_enum(
            shuffle_prior_axis,
            [-1, 0, 1],
            allow_none=True
        )

        def _shuffle_axis(pd, axis=0):
            # Shuffle index (genes) in the priors_data
            Debug.vprint(
                f"Randomly shuffling prior {pd.shape} "
                f"{'gene' if axis == 0 else 'TF'} data",
                level=0
            )

            if axis == 0:
                prior_index = pd.index.tolist()
            elif axis == 1:
                prior_index = pd.columns.tolist()

            pd = pd.sample(
                frac=1,
                axis=axis,
                random_state=random_seed
            )

            if axis == 0:
                pd.index = prior_index
            elif axis == 1:
                pd.columns = prior_index

            return pd

        if shuffle_prior_axis is None:
            return priors_data

        elif shuffle_prior_axis == 0:
            priors_data = _shuffle_axis(
                priors_data
            )
        elif shuffle_prior_axis == 1:
            priors_data = _shuffle_axis(
                priors_data, axis=1
            )
        elif shuffle_prior_axis == -1:
            priors_data = _shuffle_axis(
                priors_data
            )
            priors_data = _shuffle_axis(
                priors_data, axis=1
            )

        return priors_data

    @staticmethod
    def add_prior_noise(
        priors_data,
        noise_ratio,
        random_seed
    ):
        """
        Add random edges to the prior. Note that this will binarize the
        prior if it was not already binary.

        :param priors_data: Prior data [G x K]
        :type priors_data: pd.DataFrame
        :param noise_ratio: Ratio of edges to add to the prior
        :type noise_ratio: float
        :param random_seed: Random seed for generator
        :type random_seed: int
        :return: Prior data [G x K]
        :rtype: pd.DataFrame
        """

        assert check.argument_numeric(noise_ratio, low=0, high=1)
        assert check.argument_integer(random_seed)

        rgen = np.random.default_rng(random_seed)

        new_prior = rgen.random(priors_data.shape)
        cutoff = np.quantile(new_prior, noise_ratio, axis=None)

        priors_data = priors_data != 0
        old_prior_sum = priors_data.sum().sum()

        priors_data += new_prior <= cutoff
        priors_data = (priors_data != 0).astype(int)
        new_prior_sum = priors_data.sum().sum()

        Debug.vprint(
            f"Prior {priors_data.shape} [{old_prior_sum}] modified "
            f"to {noise_ratio:.02f} noise [{new_prior_sum}]",
            level=0
        )

        return priors_data


    @staticmethod
    def _split_for_cv(
        all_data,
        split_ratio,
        split_axis=DEFAULT_CV_AXIS,
        seed=DEFAULT_SEED
    ):
        """
        Take a dataframe and split it according to split_ratio on split_axis
        into two new dataframes. This is for crossvalidation splits of a gold
        standard.

        :param all_data: Network edges [G x K] dataframe
        :type all_data: pd.DataFrame
        :param split_ratio: The proportion of the priors that should go
            into the gold standard
        :type split_ratio: float
        :param split_axis: Splits on rows (when 0),
            columns (when 1),
            or on flattened individual data points (when None)
            Note that if this is None, the returned gold standard will be
            unchanged, and the priors will have half of the network edges in
            the gold standard. If a different prior network has been passed in
            it will be discarded
        :type split_axis: int, None
        :param random_seed: Random seed
        :type random_seed: int
        :return: Returns a new prior network [G * (1-split_ratio) x K]
            and gold standard network [G * (split_ratio) x K]
            by splitting the old prior1
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.argument_numeric(split_ratio, 0, 1)
        assert check.argument_enum(split_axis, [0, 1], allow_none=True)

        # Split the priors into gold standard based on axis
        # (flatten if axis=None)
        if split_axis is None:
            priors_data, _ = ManagePriors._split_flattened(
                all_data,
                split_ratio,
                seed=seed
            )

            # Reset the gold standard as the full data
            gold_standard = all_data

        # Split the network on either axis and return the pieces
        else:
            priors_data, gold_standard = ManagePriors._split_axis(
                all_data,
                split_ratio,
                axis=split_axis,
                seed=seed
            )

        return priors_data, gold_standard

    @staticmethod
    def _remove_prior_circularity(
        priors,
        gold_standard,
        split_axis=DEFAULT_CV_AXIS
    ):
        """
        Remove all axis labels that occur in the gold standard from the prior

        :param priors_data: Prior network data [G x K]
        :type priors_data: pd.DataFrame
        :param gold_standard: Gold standard network data [G x K]
        :type gold_standard: pd.DataFrame
        :param split_axis: Remove overlap on rows (when 0),
            or on columns (when 1),
        :type split_axis: int, None
        :return: New prior network data without any values that are in the
            gold standard network on the selected axis
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.argument_enum(split_axis, [0, 1])

        new_priors = priors.drop(
            gold_standard.axes[split_axis],
            axis=split_axis,
            errors='ignore'
        )

        return new_priors, gold_standard

    @staticmethod
    def _split_flattened(
        data,
        split_ratio,
        seed=DEFAULT_SEED
    ):
        """
        Instead of splitting by axis labels, split edges and ignore axes

        :param all_data: Network edges [G x K] dataframe
        :type all_data: pd.DataFrame
        :param split_ratio: The proportion of the network data edges
            that should go into the gold standard
        :type split_ratio: float
        :param random_seed: Random seed
        :type random_seed: int
        :return: Network edges split into two [G x K] dataframes
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.argument_numeric(split_ratio, 0, 1)

        # Get the number of non-zero edges
        # and order them randomly
        pc = np.sum(data.values != 0)
        gs_count = int(split_ratio * pc)
        idx = ManagePriors._make_shuffled_index(pc, seed=seed)

        # Get the nonzero edges as a flattened array
        pr_idx = data.values[data.values != 0]
        gs_idx = pr_idx.copy()

        # Set some of them to zero in one and the rest to zero in the other
        pr_idx[idx[0:gs_count]] = 0
        gs_idx[idx[gs_count:]] = 0

        gs = data.values.copy()
        pr = gs.copy()

        # Replace the nonzero values with some nonzero values and some
        # zero values
        gs[gs != 0] = gs_idx
        pr[pr != 0] = pr_idx

        # Rebuild dataframes
        priors_data = pd.DataFrame(
            pr,
            index=data.index,
            columns=data.columns
        )

        gold_standard = pd.DataFrame(
            gs,
            index=data.index,
            columns=data.columns
        )

        return priors_data, gold_standard

    @staticmethod
    def _split_axis(
        priors,
        split_ratio,
        axis=DEFAULT_CV_AXIS,
        seed=DEFAULT_SEED
    ):
        """
        Split by axis labels on the chosen axis

        :param priors: Network edges [G x K] dataframe
        :type priors: pd.DataFrame
        :param split_ratio: The proportion of the network data axis
            that should go into the gold standard
        :param axis: Split on on rows (when 0), or on columns (when 1)
        :type axis: int
        :param random_seed: Random seed
        :type random_seed: int
        :return: Network edges split into two dataframes on an axis
        :rtype: pd.DataFrame, pd.DataFrame
        """

        assert check.argument_numeric(split_ratio, 0, 1)
        assert check.argument_enum(axis, [0, 1])

        # Get the number of entries on the axis
        # and decide where to cut it for the split
        pc = priors.shape[axis]
        gs_count = int((1 - split_ratio) * pc)
        idx = ManagePriors._make_shuffled_index(
            pc,
            seed=seed
        )

        if axis == 0:
            axis_idx = priors.index
        elif axis == 1:
            axis_idx = priors.columns
        else:
            raise ValueError("Axis can only be 0 or 1")

        # Select the axis labels for the prior
        # and the axis labels for the gold standard,
        # randomly chosen, not from the existing order
        pr_idx = axis_idx[idx[0:gs_count]]
        gs_idx = axis_idx[idx[gs_count:]]

        # Drop the labels from the appropriate axis
        priors_data = priors.drop(gs_idx, axis=axis)
        gold_standard = priors.drop(pr_idx, axis=axis)

        return priors_data, gold_standard

    @staticmethod
    def _make_shuffled_index(
        idx_len,
        seed=DEFAULT_SEED
    ):

        idx = list(range(idx_len))
        np.random.RandomState(seed=seed).shuffle(idx)

        return idx
