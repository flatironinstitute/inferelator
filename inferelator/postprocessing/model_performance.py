import os
import pandas as pd
import numpy as np
import gzip

from inferelator import utils
from inferelator.utils import Validator as check
from inferelator.postprocessing import (
    GOLD_STANDARD_COLUMN,
    CONFIDENCE_COLUMN,
    TARGET_COLUMN,
    REGULATOR_COLUMN,
    TN,
    FN,
    TP,
    FP
)


class RankSummingMetric:
    """
    This class takes a data set that has some rankable values and
    a gold standard for which elements of that data set
    are true and calculates confidence
    """

    # Metric name
    name = "Confidences"

    # Filter methods to align gold standard and confidences
    filter_method = None
    filter_method_lookup = {
        'overlap': 'filter_to_overlap',
        'keep_all_gold_standard': 'filter_to_left_size'
    }

    # Data (wide)
    rankable_data = None
    gold_standard = None
    all_confidences = None

    # Processed data (long)
    confidence_data = None
    filtered_data = None

    # File name
    curve_file_name = None

    def __init__(
        self,
        rankable_data,
        gold_standard,
        filter_method='keep_all_gold_standard'
    ):
        """
        Take rankable data and process it into confidence
        scores which are stored in this object

        :param rankable_data: A list of numeric dataframes
            (with identical axes)
        :type rankable_data: list(pd.DataFrame) [B x [G x K]]
        :param gold_standard: A dataframe which corresponds to known,
            gold-standard data
        :type gold_standard: pd.DataFrame [G x K]
        :param filter_method: The method of aligning the gold-standard
            data to the modeled network results
        :type gold_standard: str
        """

        # Get the filtering method
        assert check.argument_enum(
            filter_method,
            self.filter_method_lookup.keys()
        )

        self.filter_method = getattr(
            self,
            self.filter_method_lookup[filter_method]
        )

        # Explicitly cast the gold standard data to a boolean array [0,1]
        gold_standard = (gold_standard != 0).astype(int)
        self.gold_standard = gold_standard

        # Calculate confidences based on the ranked data
        self.all_confidences = self.compute_combined_confidences(rankable_data)

        # Convert the confidence data to long format
        confidence_data = utils.melt_and_reindex_dataframe(
            self.all_confidences,
            CONFIDENCE_COLUMN,
            idx_name=TARGET_COLUMN,
            col_name=REGULATOR_COLUMN
        )

        # Attach the gold standard
        confidence_data = self.attach_gs_to_confidences(
            confidence_data,
            gold_standard
        )

        # Sort by confidence (descending) and reset the index
        self.confidence_data = confidence_data.sort_values(
            by=CONFIDENCE_COLUMN,
            ascending=False,
            na_position='last'
        )

        self.confidence_data.reset_index(inplace=True)

        # Filter the gold standard and confidences down to a format
        # that can be directly compared
        utils.Debug.vprint(
            f"GS: {gold_standard.sum().sum()} edges; "
            f"Confidences: {(confidence_data[CONFIDENCE_COLUMN] > 0).sum()} "
            "edges",
            level=0
        )

        self.filtered_data = self.filter_method(
            GOLD_STANDARD_COLUMN,
            CONFIDENCE_COLUMN,
            self.confidence_data
        ).copy()

        utils.Debug.vprint(
            "Filtered network data to "
            f"{(self.filtered_data[CONFIDENCE_COLUMN] > 0).sum()} edges",
            level=1
        )

    def score(self):
        raise NotImplementedError

    def all_scores(self):
        return dict([tuple(self.score())])

    @classmethod
    def all_names(cls):
        return tuple([cls.name])

    def auc(self):
        raise NotImplementedError

    def confidence_dataframe(self):
        return self.confidence_data

    def curve_dataframe(self):
        raise NotImplementedError

    def output_curve_pdf(
        self,
        output_dir,
        file_name=None,
        dpi=300,
        figsize=(6, 4)
    ):

        file_name = self.curve_file_name if file_name is None else file_name

        # Plot the curve
        ax = self.output_curve(figsize=figsize)
        fig = ax.get_figure()

        # If there's a file name set, make the output file
        if file_name is not None and output_dir is not None:
            # Save the plot and close
            self.save_figure(
                os.path.join(output_dir, file_name),
                fig, dpi=dpi
            )

        return fig, ax

    def output_curve(
        self,
        ax=None,
        figsize=(6, 4)
    ):
        raise NotImplementedError

    @staticmethod
    def save_figure(file_name, fig, dpi=300):
        """
        Save figure. Gzip if the file name ends with .gz

        :param file_name: File name
        :type file_name: str
        :param fig: Figure object
        :type fig: matplotlib.Figure
        :param dpi: DPI of output
        :type dpi: int
        """
        if file_name is None:
            pass
        elif file_name.lower().endswith(".gz"):
            with gzip.open(file_name, mode='w') as fh:
                fig.savefig(fh, dpi=dpi)
        else:
            fig.savefig(file_name, dpi=dpi)

    @staticmethod
    def attach_gs_to_confidences(confidence_data, gold_standard):
        """
        Outer join the gold standard into the confidence data

        :param confidence_data: Wide dataframe with confidence scores
        :type confidence_data: pd.DataFrame [G*K x n]
        :param gold_standard: Long dataframe with gold standard edges
        :type gold_standard: pd.DataFrame [G x K]
        :return: Long dataframe with scores and gold standard edges
        :rtype: pd.DataFrame
        """

        gold_standard = utils.melt_and_reindex_dataframe(
            gold_standard,
            GOLD_STANDARD_COLUMN,
            idx_name=TARGET_COLUMN,
            col_name=REGULATOR_COLUMN
        )

        return confidence_data.join(
            gold_standard,
            how='outer',
            on=[TARGET_COLUMN, REGULATOR_COLUMN]
        )

    @staticmethod
    def compute_combined_confidences(rankable_data):
        """
        Calculate confidences based on ranking value in all of the data
        frames and summing the ranks

        :param rankable_data: list(pd.DataFrame [M x N])
        :return combine_conf: pd.DataFrame [M x N]
        """
        # Create an 0s dataframe shaped to the data to be ranked
        combine_conf = pd.DataFrame(
            np.zeros(rankable_data[0].shape),
            index=rankable_data[0].index,
            columns=rankable_data[0].columns
        )

        for replicate in rankable_data:
            # Flatten and rank based on the beta error reductions
            ranked_replicate = np.reshape(
                pd.DataFrame(replicate.values.flatten()).rank().values,
                replicate.shape
            )
            # Sum the rankings for each bootstrap
            combine_conf += ranked_replicate

        # Convert rankings to confidence values
        min_element = min(combine_conf.values.flatten())
        denominator = len(rankable_data) * combine_conf.size - min_element

        return (combine_conf - min_element) / denominator

    @staticmethod
    def compute_confusion_matrix(
        data,
        rank_col=CONFIDENCE_COLUMN,
        gs_col=GOLD_STANDARD_COLUMN
    ):

        # Copy off just needed columns
        data = data[[rank_col, gs_col]].copy()

        # Force sort if necessary
        if not RankSummingMetric.is_monotonic_decreasing(data, rank_col):
            _reindex = data.index.copy()
            data.sort_values(
                by=rank_col,
                ascending=False,
                na_position='last',
                inplace=True
            )
        else:
            _reindex = None

        # Get indices for testable edges
        valid_gs_idx = ~pd.isnull(data[gs_col])

        # Find the edges that are in the gold standard
        df = data.loc[valid_gs_idx, [rank_col, gs_col]].copy()
        df[gs_col] = df[gs_col].astype(bool)

        # Calculate cumulative confusion matrix at each row
        df[TP] = (df[gs_col]).astype(int).cumsum()
        df[FP] = (~df[gs_col]).astype(int).cumsum()

        df[TN] = pd.Series(
            (~df[gs_col].iloc[::-1]).astype(int).cumsum(),
            index=df.index
        ).shift(-1, fill_value=0)

        df[FN] = pd.Series(
            (df[gs_col].iloc[::-1]).astype(int).cumsum(),
            index=df.index
        ).shift(-1, fill_value=0)

        # Handle ties
        RankSummingMetric.transform_column(df, rank_col, TP, 'max')
        RankSummingMetric.transform_column(df, rank_col, FP, 'max')
        RankSummingMetric.transform_column(df, rank_col, TN, 'min')
        RankSummingMetric.transform_column(df, rank_col, FN, 'min')

        # Stick confusion results back onto the data and return it
        for col in [TP, FP, TN, FN]:
            data[col] = np.NaN
            data.loc[valid_gs_idx, col] = df[col].astype(data[col].dtype)

        if _reindex is not None:
            data = data.reindex(_reindex)

        return data

    @staticmethod
    def filter_to_left_size(left_column, right_column, data):
        # Return data where one column (left) is not NA
        return data.dropna(subset=[left_column])

    @staticmethod
    def filter_to_overlap(left_column, right_column, data):
        # Return data where both columns are not NA
        return data.dropna(subset=[left_column, right_column])

    @staticmethod
    def transform_column(
        df,
        group_col,
        xform_col,
        xform,
        inplace=True
    ):
        """
        Transform a column based on a grouping column
        Used to handle ties

        :param df: Input dataframe
        :type df: pd.DataFrame
        :param group_col: Column to group by
        :type group_col: str
        :param xform_col: Column to transform
        :type xform_col: str
        :param xform: Transformation to use
        :type xform: str, func
        :param inplace: Transform in place, defaults to True
        :type inplace: bool, optional
        :return: Dataframe if inplace is False, otherwise None
        :rtype: pd.DataFrame, None
        """
        if not inplace:
            df = df.copy()

        df[xform_col] = df[[group_col, xform_col]].groupby(
            group_col
        )[xform_col].transform(
            xform
        )

        if not inplace:
            return df

    @staticmethod
    def is_monotonic_decreasing(data, col):
        data = data.loc[~pd.isnull(data[col]), col]
        return data.is_monotonic_decreasing
