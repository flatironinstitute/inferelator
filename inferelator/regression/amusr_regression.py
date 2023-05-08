import numpy as np
import pandas as pd
import itertools
import functools

from inferelator.distributed.inferelator_mp import MPControl
from inferelator.utils import (
    Debug,
    Validator as check
)
from .base_regression import (
    _MultitaskRegressionWorkflowMixin,
    BaseRegression,
    PreprocessData,
    PROGRESS_STR
)
from .amusr_math import run_regression_EBIC

DEFAULT_prior_weight = 1.0

TOL = 1e-2
REL_TOL = None


class AMuSR_regression(BaseRegression):

    X = None  # list(InferelatorData)
    Y = None  # list(InferelatorData)
    priors = None  # list(pd.DataFrame [G, K]) OR pd.DataFrame [G, K]
    tfs = None  # pd.Index OR list
    genes = None  # pd.Index OR list

    K = None  # int
    G = None  # int
    n_tasks = None  # int
    prior_weight = DEFAULT_prior_weight  # float
    remove_autoregulation = True  # bool

    lambda_Bs = None
    lambda_Ss = None
    Cs = None
    Ss = None

    tol = None
    rel_tol = None

    use_numba = False

    regression_function = staticmethod(run_regression_EBIC)

    def __init__(self,
                 X,
                 Y,
                 tfs=None,
                 genes=None,
                 priors=None,
                 prior_weight=DEFAULT_prior_weight,
                 remove_autoregulation=True,
                 lambda_Bs=None,
                 lambda_Ss=None,
                 Cs=None,
                 Ss=None,
                 tol=TOL,
                 rel_tol=REL_TOL,
                 use_numba=False
    ):
        """
        Set up a regression object for multitask regression
        :param X: Design activity data for each task
        :type X: list(InferelatorData)
        :param Y: Response expression data for each task
        :type Y: list(InferelatorData)
        :param priors: Prior network data for each task
        :type priors: list(pd.DataFrame [G, K])
        :param prior_weight: Weight for existing edges in prior network
        :type prior_weight: numeric
        :param remove_autoregulation: bool
        :param lambda_Bs: list(float) [None]
        :param lambda_Ss: list(float) [None]
        :param Cs: list(float) [None]
        :param Ss: list(float) [None]
        :param use_numba: bool [False]
        """

        # Check input types are correct
        assert check.argument_type(X, list)
        assert check.argument_type(Y, list)
        assert check.argument_type(tfs, list, allow_none=True)
        assert check.argument_type(genes, list, allow_none=True)
        assert check.argument_numeric(prior_weight)
        assert check.argument_numeric(tol)
        assert check.argument_numeric(rel_tol, allow_none=True)

        assert len(X) == len(Y)

        # Set the data into the regression object
        self.X = scale_list_of_data(X)
        self.Y = scale_list_of_data(Y)
        self.n_tasks = len(X)

        # Set the priors and weight into the regression object
        self.priors = priors
        self.prior_weight = float(prior_weight)

        # Construct a list of regulators if they are not passed in
        if tfs is None:
            self.tfs = [design.gene_names.tolist() for design in X]
        else:
            self.tfs = tfs

        # Construct a list of genes if they are not passed in
        if genes is None:
            self.genes = filter_genes_on_tasks(
                [resp.gene_names for resp in Y],
                'union'
            )
            self.genes = genes_tasks(self.genes, self.Y)
        else:
            self.genes = genes

        # Set the regulators and targets into the regression object
        self.K, self.G = len(self.tfs), len(self.genes)
        self.remove_autoregulation = remove_autoregulation

        # Set the regularization coefficients into the regression object
        self.lambda_Bs = lambda_Bs
        self.lambda_Ss = lambda_Ss
        self.Cs = Cs
        self.Ss = Ss

        # Set the tolerances into the regression object
        self.tol = tol
        self.rel_tol = rel_tol

        self.use_numba = use_numba

    def regress(self):
        """
        Execute multitask (AMUSR)
        :return: list
            Returns a list of regression results that the amusr_regression
            pileup_data can process
        """

        return MPControl.map(
            _amusr_wrapper,
            itertools.repeat(self.X, self.G),
            _y_generator(self.Y, self.genes),
            itertools.repeat(self.priors, self.G),
            itertools.repeat(self.prior_weight, self.G),
            range(self.G),
            itertools.repeat(self.G, self.G),
            itertools.repeat(self.tfs, self.G),
            Cs=self.Cs,
            Ss=self.Ss,
            lambda_Bs=self.lambda_Bs,
            lambda_Ss=self.lambda_Ss,
            tol=self.tol,
            rel_tol=self.rel_tol,
            use_numba=self.use_numba,
            scatter=[self.X, self.priors]
        )

    def pileup_data(self, run_data):

        weights = []
        rescaled_weights = []

        for k in range(self.n_tasks):
            results_k = []
            for res in run_data:
                try:
                    results_k.append(res[k])
                except (KeyError, IndexError):
                    pass

            results_k = pd.concat(results_k)

            weights_k = _format_weights(
                results_k,
                'weights',
                self.Y[k].gene_names,
                self.tfs[k]
            )

            rescaled_weights_k = _format_weights(
                results_k,
                'resc_weights',
                self.Y[k].gene_names,
                self.tfs[k]
            )

            rescaled_weights_k[rescaled_weights_k < 0.] = 0

            weights.append(weights_k)
            rescaled_weights.append(rescaled_weights_k)

        return weights, rescaled_weights


def _amusr_wrapper(
    x_data,
    y_data_stack,
    prior,
    prior_weight,
    j,
    nG,
    tfs,
    remove_autoregulation=True,
    **kwargs
):
    """ Wrapper for multiprocessing AMuSR """

    y_data, gene = y_data_stack

    Debug.vprint(
        PROGRESS_STR.format(gn=gene[0], i=j, total=nG),
        level=0 if j % 1000 == 0 else 2
    )

    x, y, tf_t, tasks = [], [], [], []

    if remove_autoregulation:
        tfs = np.asarray(tfs)
        tf_keep = np.ones_like(tfs, dtype=bool)
        for g in gene:
            tf_keep &= tfs != g
        tfs = tfs[:, np.any(tf_keep, axis=0)]

    # Reorder the task data so it matches the genes that were passed in
    for k, y_task in y_data:
        x.append(x_data[k].get_gene_data(tfs[k]))  # list([N, K])
        y.append(y_task)
        tf_t.append(tfs[k])
        tasks.append(k)  # [T,]

    prior = format_prior(
        prior,
        gene,
        tasks,
        prior_weight,
        tfs=tfs
    )

    return run_regression_EBIC(
        x,
        y,
        tf_t,
        tasks,
        gene, prior,
        **kwargs
    )


def _y_generator(y_data, genes):

    nG = len(genes)

    for i in range(nG):
        y, y_genes = [], []

        for k, g in genes[i]:

            if g in y_data[k].gene_names:

                y.append((
                    k,
                    y_data[k].get_gene_data(
                        g,
                        force_dense=True
                    ).reshape(-1, 1)
                ))

                y_genes.append(g)

        yield y, y_genes


def format_prior(priors, gene, tasks, prior_weight, tfs=None):
    """
    Returns weighted priors for one gene
    :param priors: list(pd.DataFrame [G x K]) or pd.DataFrame [G x K]
        Either a list of prior data or a single data frame to use for all tasks
    :param gene: str
        The gene to select from the priors
    :param tasks: list(int)
        A list of task IDs
    :param prior_weight: float, int
        How much to weight the priors
    :return prior_out: np.ndarray [K x T]
        The weighted priors for a specific gene in each task
    """

    assert check.argument_type(priors, (list, pd.DataFrame), allow_none=True)
    assert check.argument_type(tasks, list)
    assert check.argument_numeric(prior_weight)

    if priors is None:
        return None

    if not isinstance(priors, (tuple, list)):
        priors = [priors] * (max(tasks) + 1)

    if not isinstance(gene, (tuple, list)):
        gene = [gene] * len(priors)

    if tfs is None:
        tfs = [p.columns.tolist() for p in priors]

    priors_out = [
        _weight_prior(
            _reindex_to_gene(
                priors[k],
                gene[i],
                tfs[k]
            ).loc[gene[i], :].values,
            prior_weight
        ) for i, k in enumerate(tasks)
    ]

    return np.transpose(np.vstack(priors_out))


def _reindex_to_gene(p, g, tfs):
        p = p.reindex([g])
        p = p.reindex(tfs, axis=1) if tfs is not None else p
        p = p.fillna(0.0)
        return p


def _weight_prior(prior, prior_weight):
    """
    Weight priors
    :param prior: pd.Series [K] or np.ndarray [K,]
        The prior information
    :param prior_weight: numeric
        How much to weight priors. If this is 1, do not weight prior
        terms differently
    :return prior: pd.Series [K] or np.ndarray [K,]
        Weighted priors
    """

    # Set non-zero priors to 1 and zeroed priors to 0
    prior = (prior != 0).astype(float)

    # Divide by the prior weight
    prior /= prior_weight

    # Set zeroed priors to 1
    prior[prior == 0] = 1.0

    # Reweight priors
    prior = prior / prior.sum() * len(prior)
    return prior


def _format_weights(df, col, targets, regs):
    """
    Reformat the edge table (target -> regulator) that's output by amusr
    into a pivoted table that the rest of the inferelator workflow can handle

    :param df: pd.DataFrame
        An edge table (regulator -> target) with columns
        containing model values
    :param col:
        Which column to pivot into values
    :param targets: list
        A list of target genes (the index of the output data)
    :param regs: list
        A list of regulators (the columns of the output data)
    :return out: pd.DataFrame [G x K]
        A [targets x regulators] dataframe pivoted from the edge dataframe
    """

    # Make sure that the value column is all numeric
    df[col] = pd.to_numeric(df[col])

    # Pivot an edge table into a matrix of values
    out = pd.pivot_table(
        df,
        index='target',
        columns='regulator',
        values=col,
        fill_value=0.
    ).astype(float)

    # Reindex to a [targets x regulators] dataframe and fill anything
    # missing with 0s
    out = out.reindex(targets).reindex(regs, axis=1)
    out = out.fillna(value=0.)

    return out


def scale_list_of_data(X):
    """
    Scale a list of data objects so that each has mean 0 and unit variance

    :param X: list(InferelatorData) [T]
    :return X: list(InferelatorData) [T]
    """

    assert check.argument_type(X, list)

    return [PreprocessData.preprocess_design(xk) for xk in X]


class AMUSRRegressionWorkflowMixin(_MultitaskRegressionWorkflowMixin):
    """
    Multi-Task AMuSR regression

    https://doi.org/10.1371/journal.pcbi.1006591
    """

    prior_weight = 1

    # Model hyperparameters
    lambda_Bs = None
    lambda_Ss = None
    heuristic_Cs = None

    tol = TOL
    relative_tol = REL_TOL

    _r_class = AMuSR_regression

    def set_regression_parameters(
        self,
        prior_weight=None,
        lambda_Bs=None,
        lambda_Ss=None,
        heuristic_Cs=None,
        tol=None,
        relative_tol=None
    ):
        """
        Set regression parameters for AmUSR.

        :param prior_weight: Weight for edges that are present in the prior
            network. Non-prior edges have a weight of 1. Set this to 1 to
            weight prior and non-prior edges equally, Defaults to 1.
        :type prior_weight: numeric
        :param lambda_Bs: Lambda_B values to search during model selection.
            If not set, lambda_B will be chosen using the heuristic
            lambda_b = c * sqrt(d log p / n) from Castro 2019
            Defaults to not set. Must be provided if lambda_S is set.
        :type lambda_Bs: list(floats) or np.ndarray(floats)
        :param lambda_Ss: Lambda_S values to search during model selection.
            If not set, lambda_S will be chosen using the heuristic
            0.5 < lambda_s/lambda_b < 1 from Castro 2019
            Defaults to not set.
        :type lambda_Ss: list(floats) or np.ndarray(floats)
        :param heuristic_Cs: c values to search during model selection.
            Values of c to calculate lambda_b = c * sqrt(d log p / n),
            Defaults to np.logspace(np.log10(0.01), np.log10(10), 20)[::-1].
            Does not have an effect if lambda_B is provided.
        :type heuristic_Cs: list(floats) or np.ndarray(floats)
        :param tol: Convergence tolerance for amusr regression
        :type tol: float
        :param relative_tol: Relative convergence tolerance for
            amusr regression
        :type relative_tol: float
        """

        self._set_with_warning("prior_weight", prior_weight)
        self._set_with_warning("lambda_Bs", lambda_Bs)
        self._set_with_warning("lambda_Ss", lambda_Ss)
        self._set_with_warning("heuristic_Cs", heuristic_Cs)
        self._set_without_warning("tol", tol)
        self._set_without_warning("relative_tol", relative_tol)

    def run_bootstrap(self, bootstrap_idx):
        x, y, tfs = [], [], []

        # Select the appropriate bootstrap from each task and stash the
        # data into X and Y
        for k in range(self._n_tasks):

            if bootstrap_idx is not None:
                x.append(self._task_design[k].get_bootstrap(
                    self._task_bootstraps[k][bootstrap_idx]
                ))
                y.append(self._task_response[k].get_bootstrap(
                    self._task_bootstraps[k][bootstrap_idx]
                ))

            else:
                x.append(self._task_design[k])
                y.append(self._task_response[k])

            tfs.append(self._task_design[k].gene_names.tolist())

        regress = self._r_class(
            x,
            y,
            tfs=tfs,
            genes=self._task_genes,
            priors=self._task_priors,
            prior_weight=self.prior_weight,
            lambda_Bs=self.lambda_Bs,
            lambda_Ss=self.lambda_Ss,
            Cs=self.heuristic_Cs,
            tol=self.tol,
            rel_tol=self.relative_tol,
            use_numba=self.use_numba
        )

        return regress.run()


def filter_genes_on_tasks(list_of_indexes, task_expression_filter):
    """
    Take a list of indexes and filter them based on the method specified in
    task_expression_filter to a single index

    :param list_of_indexes: list(pd.Index)
    :param task_expression_filter: str or int
    :return filtered_genes: pd.Index
    """

    filtered_genes = list_of_indexes[0]

    # If task_expression_filter is a number only keep genes that appear
    # in that number of tasks or more
    if isinstance(task_expression_filter, int):
        filtered_genes = pd.concat(
            list(map(lambda x: x.to_series(), list_of_indexes))
        ).value_counts()
        _filtered_idx = filtered_genes >= task_expression_filter
        filtered_genes = filtered_genes[_filtered_idx].index

    # If task_expression_filter is "intersection" only keep genes in all tasks
    elif task_expression_filter == "intersection":
        filtered_genes = functools.reduce(
            lambda x, y: x.intersection(y),
            list_of_indexes
        )

    # If task_expression_filter is "union" keep genes that are in any task
    elif task_expression_filter == "union":
        filtered_genes = functools.reduce(
            lambda x, y: x.union(y),
            list_of_indexes
        )
    else:
        raise ValueError(
            f"{task_expression_filter} is not an allowed "
            "task_expression_filter value"
        )

    return filtered_genes


def genes_tasks(list_of_genes, list_of_data):
    """
    Take a list of genes and find them in the task
    response data gene_names.

    Returns (task #, gene_id) tuples

    :param list_of_genes: List of gene IDs
    :type list_of_genes: list, pd.Index
    :param list_of_data: list of response data
    :type list_of_data: list(InferelatorData)
    :return: List where each element is genes that
        should be learned together as (task #, gene_id)
        tuples
    :rtype: list(list(tuple(int, str)))
    """

    genes_tasks = []
    for g in list_of_genes:
        genes_group = [(i, g)
                       for i, d in enumerate(list_of_data)
                       if g in d.gene_names]

        if len(genes_group) > 0:
            genes_tasks.append(genes_group)

    return genes_tasks
