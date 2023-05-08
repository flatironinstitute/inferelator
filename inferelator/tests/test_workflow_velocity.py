import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt

from inferelator.workflow import inferelator_workflow
from inferelator.workflows.velocity_workflow import VelocityWorkflow
from inferelator.utils.inferelator_data import InferelatorData

# Start, f(A), DC
GENE_DATA = [
    (0, 0.1, 0.05),
    (10, 1, 0.1),
    (100, 10, 0.25),
    (100, 0.1, 0.05),
    (10, 0.01, 0.05),
    (0, 0.01, 1)
]

GENE_NAMES = [
    f"Gene{x}" for x in range(len(GENE_DATA))
]

def _expr_velo(t, decay=None):

    _ed = np.zeros((len(t), len(GENE_DATA)))
    _velos = np.zeros((len(t), len(GENE_DATA)))

    if decay is None:
        _dcs = np.asarray([x[2] for x in GENE_DATA])
    else:
        _dcs = np.asarray([decay for x in GENE_DATA])

    _vs = np.asarray([x[1] for x in GENE_DATA])
    for i, _dt in enumerate(np.insert(np.diff(t), 0, 0)):

        if i == 0:
            _ed[i, :] = [x[0] for x in GENE_DATA]
            _velos[i, :] = [x[1] for x in GENE_DATA]
        else:
            _velos[i, :] = _dt * (-1 * _dcs * _ed[i-1, :] + _vs)
            _ed[i, :] = _velos[i, :] + _ed[i-1, :]

    return _ed, _velos


TEST_TIME = np.linspace(0, 10, 100)
TEST_EXPR, TEST_VELOS = (
    InferelatorData(
        x,
        gene_names=GENE_NAMES
    )
    for x in _expr_velo(TEST_TIME)
)
TEST_DECAYS = pd.DataFrame(
    [x[2] for x in GENE_DATA],
    index=GENE_NAMES,
    columns=["DC"]
)


class TestVelocityWorkflow(unittest.TestCase):

    def setUp(self) -> None:

        self.worker = inferelator_workflow(
            'base',
            VelocityWorkflow
        )

        self.worker.data = TEST_EXPR.copy()
        self.worker._velocity_data = TEST_VELOS.copy()
        self.worker._decay_constants = TEST_DECAYS.copy()

        return super().setUp()

    def test_combine_no_decay(self):

        self.worker._decay_constants = None

        _combined = self.worker._combine_expression_velocity(
            TEST_EXPR,
            TEST_VELOS
        )

        npt.assert_equal(
            _combined.values,
            TEST_VELOS.values
        )

        self.worker._decay_constants = None

        _combined = self.worker._combine_expression_velocity(
            TEST_EXPR,
            TEST_VELOS
        )

        npt.assert_equal(
            _combined.values,
            TEST_VELOS.values
        )

    def test_combine_fixed_decay(self):

        self.worker._global_decay_constant = 0.1

        _correct_combined = TEST_VELOS.values.copy()
        _correct_combined += TEST_EXPR.values * 0.1

        _combined = self.worker._combine_expression_velocity(
            TEST_EXPR,
            TEST_VELOS
        )

        npt.assert_equal(
            _correct_combined,
            _combined.values
        )

    def test_combine_gene_decay(self):

        _correct_combined = TEST_VELOS.values.copy()
        _correct_combined += TEST_EXPR.values * TEST_DECAYS.values.flatten()[None, :]

        _combined = self.worker._combine_expression_velocity(
            TEST_EXPR,
            TEST_VELOS
        )

        npt.assert_equal(
            _correct_combined,
            _combined.values
        )