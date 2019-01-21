import unittest
from .. import prior
import pandas as pd
import os

"""
Fixing travis to properly test this module is more work than turning the travis test for this module off
So prior.py isn't protected by CI
"""

def should_skip(environment_flags=[("TRAVIS", "true"), ("SKIP_KVS_TESTS", "true")]):
    for (flag, value) in environment_flags:
        if (flag in os.environ and os.environ[flag] == value):
            return True
    # default
    return False

class TestPrior(unittest.TestCase):

    def setup_test_files(self):
        self.motifs = [('chr1', '10', '15', 'TF1', '.', '-'),
                       ('chr1', '18', '22', 'TF1', '.', '+'),
                       ('chr2', '50', '56', 'TF1', '.', '+'),
                       ('chr1', '100', '107', 'TF2', '.', '-'),
                       ('chr1', '103', '108', 'TF3', '.', '+'),
                       ('chr1', '150', '154', 'TF4', '.', '+')]

        self.tss = [('chr1', '20', '21', 'gene1', '.', '+'),
                    ('chr1', '20', '21', 'gene2', '.', '-'),
                    ('chr1', '120', '121', 'gene3', '.', '+')]

        self.genes = [('chr1', '20', '45', 'gene2', '.', '+'),
                 ('chr1', '5', '21', 'gene1', '.', '-'),
                 ('chr1', '120', '150', 'gene3', '.', '+')]

        self.target_genes = ['gene1', 'gene2', 'gene3']
        self.regulators = regulators = ['TF1', 'TF2', 'TF3', 'TF4']

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_empty_tfs_and_targets_lists(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs, self.genes, [], [], 'closest', 100)
        self.assertEqual(prior_object.make_prior().size, 0)

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_closest_zero_distance(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest', 0)
        expected_prior = pd.DataFrame([[1, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_closest_zero_distance_genes_with_multiple_tss_at_different_locations(self):
        self.setup_test_files()
        self.tss.append(('chr1', '100', '101', 'gene1', '.', '+'),)
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest', 0)
        expected_prior = pd.DataFrame([[1, 1, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_closest_zero_distance_genes_with_multiple_tss_at_same_location(self):
        self.setup_test_files()
        self.tss.append(('chr1', '19', '20', 'gene1', '.', '+'),)
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest', 0)
        expected_prior = pd.DataFrame([[2, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_window_TSS_zero_distance(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'window', 0)

        expected_prior = pd.DataFrame([[1, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_window_geneBody_zero_distance(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.genes,
                                   self.target_genes,
                                   self.regulators,
                                   'window', 0)
        expected_prior = pd.DataFrame([[2, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_closestTSS_default(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest')
        expected_prior = pd.DataFrame([[2, 0, 0, 0],
                                       [2, 0, 0, 0],
                                       [0, 1, 1, 1]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_closestTSS_ignore_downstream(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest', ignore_downstream = True)
        expected_prior = pd.DataFrame([[2, 0, 0, 0],
                                       [1, 0, 0, 1],
                                       [0, 1, 1, 0]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_windowGeneBody_1000(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.genes,
                                   self.target_genes,
                                   self.regulators,
                                   'window', 1000)
        expected_prior = pd.DataFrame([[2, 1, 1, 1],
                                       [2, 1, 1, 1],
                                       [2, 1, 1, 1]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))

    @unittest.skipIf(should_skip(), "Skipping this test on Travis CI.")
    def test_prior_number_of_targets_2(self):
        self.setup_test_files()
        prior_object = prior.Prior(self.motifs,
                                   self.tss,
                                   self.target_genes,
                                   self.regulators,
                                   'closest', number_of_targets = 2)
        expected_prior = pd.DataFrame([[2, 1, 1, 1],
                                       [2, 1, 1, 1],
                                       [0, 1, 1, 1]],
                                       index = ['gene1', 'gene2', 'gene3'],
                                       columns = ['TF1', 'TF2', 'TF3', 'TF4'])
        self.assertTrue(prior_object.make_prior().equals(expected_prior))
