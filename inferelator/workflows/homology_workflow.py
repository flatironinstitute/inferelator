import functools
import pandas as pd
import numpy as np

from inferelator.regression.amusr_regression import filter_genes_on_tasks

from .amusr_workflow import MultitaskLearningWorkflow


class MultitaskHomologyWorkflow(MultitaskLearningWorkflow):

    _regulator_expression_filter = "union"

    _homology_group_key = None

    _tf_homology = None
    _tf_homology_group_key = None
    _tf_homology_gene_key = None

    def set_homology(
        self,
        homology_group_key=None,
        tf_homology_map=None,
        tf_homology_map_group_key=None,
        tf_homology_map_gene_key=None
    ):
        """
        Set the gene metadata key that identifies genes to group by homology

        :param homology_group_key: Gene metadata column which identifies
            homology group
        :type homology_group_key: str, optional
        :param tf_homology_map
        """

        self._set_with_warning(
            '_homology_group_key',
            homology_group_key
        )

        self._set_with_warning(
            '_tf_homology',
            tf_homology_map
        )

        self._set_with_warning(
            '_tf_homology_group_key',
            tf_homology_map_group_key
        )

        self._set_with_warning(
            '_tf_homology_gene_key',
            tf_homology_map_gene_key
        )

    def startup_finish(self):
        super().startup_finish()
        self.homology_groupings()

    def homology_groupings(self):

        # Get all the homology groups and put them in a list
        _all_groups = functools.reduce(
            lambda x, y: x.union(y),
            [
                pd.Index(t._adata.var[self._homology_group_key])
                for t in self._task_objects
            ]
        )

        # Build a dict, keyed by homology group ID
        # Values are a list of (task, gene_id) tuples
        # for that homology group
        _group_dict = {g: [] for g in _all_groups}

        for i, t in enumerate(self._task_objects):

            for k, gene in zip(
                t._adata.var[self._homology_group_key],
                t._adata.var_names
            ):
                _group_dict[k].append((i, gene))

        # Unpack them into a list of lists
        self._task_genes = [v for _, v in _group_dict.items()]

    def _align_design_response(self):

        # Dict keyed by gene ID
        # Value is the common homology ID
        tf_homology_renamer = dict(zip(
            self._tf_homology[self._tf_homology_gene_key],
            self._tf_homology[self._tf_homology_group_key]
        ))

        # Get the homology IDs for the design data
        current_task_groups = [
            pd.Index([tf_homology_renamer[x] for x in y.gene_names])
            for y in self._task_design
        ]

        # Get all the homology IDs
        all_task_groups = filter_genes_on_tasks(
            current_task_groups,
            'union'
        )

        n_features = len(all_task_groups)

        for i in range(len(self._task_design)):
            design_data = self._task_design[i]

            _has_mapping = [
                k in tf_homology_renamer.keys()
                for k in design_data.gene_names
            ]

            _homology_map = [
                tf_homology_renamer[x]
                for x in design_data.gene_names[_has_mapping]
            ]

            _integer_map_new = [
                all_task_groups.get_loc(x)
                for x in _homology_map
            ]

            _integer_map_old = np.arange(design_data.shape[1])[_has_mapping]

            _new_data = np.zeros((design_data.shape[0], n_features),
                                 dtype=design_data.values.dtype)
            _new_names = list(map(lambda x: f"TF_ZERO_{x}", range(n_features)))

            for i, loc in zip(_integer_map_old, _integer_map_new):
                _new_data[:, loc] = design_data.values[:, i]
                _new_names[loc] = design_data.gene_names[i]

            design_data.replace_data(
                _new_data,
                new_gene_metadata=design_data.gene_data.reindex(_new_names).fillna(0)
            )

    def _get_tf_homology(self):

        pass
