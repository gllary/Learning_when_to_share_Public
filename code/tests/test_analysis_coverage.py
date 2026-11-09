"""Coverage follows accepted calibration even when the residual is zero."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import plot_analysis_figures as plots


class CoverageContract(unittest.TestCase):
    def test_unequal_fold_sizes_and_zero_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coefficients = []
            for fold, (n, beta) in enumerate(zip([2, 3, 4, 5, 6], [1, 0, .25, 0, .5])):
                path = root / f'results/matbench/full/predictions/outer_{fold}/example.npz'
                path.parent.mkdir(parents=True)
                np.savez(path, sample_ids=np.arange(n).astype(str), beta=beta,
                         prediction=np.ones(n), anchor_prediction=np.ones(n))
                coefficients.append(dict(dataset='matbench', task='example',
                                         outer_fold=fold + 1, beta=beta))
            with patch.object(plots, 'ROOT', root), patch.object(plots, 'ORDER', ['example']), \
                    patch.object(plots, 'BETA', coefficients, create=True):
                row, = plots.coverage_records('matbench')
            self.assertEqual(row['n'], 20)
            self.assertEqual(row['accepted_n'], 12)
            self.assertEqual(row['coverage_percent'], 60)

    def test_zero_beta_rejects_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'results/jarvis/full/predictions/example.npz'
            path.parent.mkdir(parents=True)
            np.savez(path, sample_ids=np.arange(7).astype(str), beta=0)
            with patch.object(plots, 'ROOT', root), \
                    patch.object(plots, 'JV', [{'task': 'example'}], create=True), \
                    patch.object(plots, 'BETA', [dict(dataset='jarvis', task='example',
                                                    outer_fold='', beta=0)], create=True):
                row, = plots.coverage_records('jarvis')
            self.assertEqual(row['accepted_n'], 0)
            self.assertEqual(row['coverage_percent'], 0)


if __name__ == '__main__':
    unittest.main()
