import unittest
import copy
import pandas as pd
import numpy as np
import pandas.testing as pdt
import numpy.testing as npt

from inferelator.utils import InferelatorData
from inferelator import inferelator_workflow
from inferelator.workflows.homology_workflow import MultitaskHomologyWorkflow

TF_T1 = ['tf1', 'tf2', 'tf3']
GENE_T1 = ['gene1', 'gene2']

TF_T2 = ['newtf1', 'newtf2', 'newtf3']
GENE_T2 = ['newgene1', 'newgene3']

DES = [InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=TF_T1)),
       InferelatorData(pd.DataFrame(np.array([[1, 1, 3], [0, 0, 2], [0, 0, 1]]).astype(float), columns=TF_T2))]

RES = [InferelatorData(pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=GENE_T1)),
       InferelatorData(pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]).astype(float), columns=GENE_T2))]

PRIORS = [pd.DataFrame([[0, 1, 1], [1, 0, 1]], index=GENE_T1, columns=TF_T1),
          pd.DataFrame([[0, 1, 1], [1, 0, 1]], index=GENE_T2, columns=TF_T2)]

HOMOLOGY_MAP_GENES = pd.DataFrame(
    [['gene1', 'hg1'],
     ['gene2', 'hg2'],
     ['newgene_1', 'hg1'],
     ['newgene3', 'hg3']],
     columns=['Gene', 'Homology']
)

HOMOLOGY_MAP_TFS = pd.DataFrame(
    [['tf1', 'hg1'],
     ['tf2', 'hg2'],
     ['tf3', 'hg3'],
     ['newtf1', 'hg1'],
     ['newtf2', 'hg2'],
     ['newtf3', 'hg3']],
     columns=['TF', 'Homology']
)

class TestHomologyMap(unittest.TestCase):

    def setUp(self) -> None:

        self.workflow = inferelator_workflow(workflow=MultitaskHomologyWorkflow, regression="amusr")
        self.workflow.create_output_dir = lambda *x: None

        self.workflow._n_tasks = 2
        self.workflow._task_bootstraps = [[np.arange(3)], [np.arange(3)]]
        self.workflow._task_design = copy.deepcopy(DES)
        self.workflow._task_response = copy.deepcopy(RES)
        self.workflow._task_priors = copy.deepcopy(PRIORS)
        self.workflow.num_bootstraps = 1
        self.workflow.gold_standard = PRIORS[0].copy()

        self.workflow._tf_homology = HOMOLOGY_MAP_TFS.copy()
        self.workflow._tf_homology_group_key = 'Homology'
        self.workflow._tf_homology_gene_key = 'TF'

        return super().setUp()

    def test_align_design_stretch(self):

        self.workflow._tf_homology['Homology'] = ['hg1', 'hg2', 'hg3', 'hg4', 'hg5', 'hg6']

        self.assertEqual(len(self.workflow._task_design), 2)
        self.assertEqual(self.workflow._task_design[0].shape, (3, 3))
        self.assertEqual(self.workflow._task_design[1].shape, (3, 3))

        self.workflow._align_design_response()
        self.assertEqual(len(self.workflow._task_design), 2)
        self.assertEqual(self.workflow._task_design[0].shape, (3, 6))
        self.assertEqual(self.workflow._task_design[1].shape, (3, 6))

        npt.assert_array_equal(
            self.workflow._task_design[1].values[:, 0:3],
            np.zeros_like(self.workflow._task_design[1].values[:, 0:3])
        )

        npt.assert_array_equal(
            self.workflow._task_design[0].values[:, 0:3],
            DES[0].values
        )

        npt.assert_array_equal(
            self.workflow._task_design[0].values[:, 3:6],
            np.zeros_like(self.workflow._task_design[0].values[:, 3:6])
        )

        npt.assert_array_equal(
            self.workflow._task_design[1].values[:, 3:6],
            DES[0].values
        )

        pdt.assert_index_equal(
            self.workflow._task_design[0].gene_names,
            pd.Index(['tf1', 'tf2', 'tf3', 'TF_ZERO_3', 'TF_ZERO_4', 'TF_ZERO_5'])
        )

        pdt.assert_index_equal(
            self.workflow._task_design[1].gene_names,
            pd.Index(['TF_ZERO_0', 'TF_ZERO_1', 'TF_ZERO_2', 'newtf1', 'newtf2', 'newtf3'])
        )

    def test_align_design_overlap(self):

        self.assertEqual(len(self.workflow._task_design), 2)
        self.assertEqual(self.workflow._task_design[0].shape, (3, 3))
        self.assertEqual(self.workflow._task_design[1].shape, (3, 3))

        self.workflow._align_design_response()
        self.assertEqual(len(self.workflow._task_design), 2)
        self.assertEqual(self.workflow._task_design[0].shape, (3, 3))
        self.assertEqual(self.workflow._task_design[1].shape, (3, 3))

        npt.assert_array_equal(
            self.workflow._task_design[0].values,
            DES[0].values
        )

        npt.assert_array_equal(
            self.workflow._task_design[1].values,
            DES[0].values
        )

        pdt.assert_index_equal(
            self.workflow._task_design[0].gene_names,
            pd.Index(HOMOLOGY_MAP_TFS['TF'].iloc[0:3].tolist())
        )

        pdt.assert_index_equal(
            self.workflow._task_design[1].gene_names,
            pd.Index(HOMOLOGY_MAP_TFS['TF'].iloc[3:6].tolist())
        )

    def test_regression(self):

        beta, resc_beta, _, _ = self.workflow.run_regression()

        pdt.assert_frame_equal(
            beta[0][0],
            pd.DataFrame([[0., 0., -1.], [0., 0., -1.]], columns=TF_T1, index=GENE_T1),
            check_names=False
        )

        pdt.assert_frame_equal(
            beta[1][0],
            pd.DataFrame([[0., 0., -1.], [0., 0., -1.]], columns=TF_T2, index=GENE_T2),
            check_names=False
        )

        pdt.assert_frame_equal(
            resc_beta[0][0],
            pd.DataFrame([[0., 0., 1.], [0., 0., 1.]], columns=TF_T1, index=GENE_T1),
            check_names=False
        )

        pdt.assert_frame_equal(
            resc_beta[1][0],
            pd.DataFrame([[0., 0., 1.], [0., 0., 1.]], columns=TF_T2, index=GENE_T2),
            check_names=False
        )
