"""Direct frozen external evaluation contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/scripts'))
sys.path.insert(0, str(ROOT / 'code'))
from evaluate_external import COHORTS, evaluate, validate_contract
from rebuild_paper_tables import render
from hpsafemoe.external.evaluation import regression_metrics, sha256, validate_prediction


class FrozenExternalContract(unittest.TestCase):
    def test_full_populations_and_published_metrics(self):
        results = [evaluate(slug) for slug in COHORTS]
        self.assertEqual([r['n'] for r in results], [23, 248, 95, 18])
        self.assertEqual(sum(r['n'] for r in results), 384)
        self.assertEqual(sum(r['frozen_beta_references_verified'] for r in results), 30)
        self.assertTrue(all(r['external_calibration_samples'] == 0 for r in results))

    def test_frozen_contract_is_required(self):
        cfg = json.loads((ROOT / 'results/external/hea95/config.json').read_text())
        for key, value in [('external_calibration_samples', 1), ('beta_refitting', True),
                           ('model_training', True), ('evaluation_samples', 90),
                           ('route_temperature', .5), ('labels_used_for', 'fitting')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(cfg)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_contract(changed)

    def test_metrics_use_fixed_predictions(self):
        truth = np.array([0., 1., 2.])
        prediction = truth + 2.
        before = prediction.copy()
        result = regression_metrics(prediction, truth)
        self.assertEqual(result['mae'], 2.)
        self.assertEqual(result['rmse'], 2.)
        self.assertEqual(result['r2'], -5.)
        np.testing.assert_array_equal(prediction, before)

    def test_exact_prediction_identity_and_ensemble(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'predictions.npz'
            outer = np.tile([1., 2.], (5, 1))
            values = dict(sample_ids=np.array(['a', 'b']), stage1=outer.mean(axis=0),
                          stage2=outer.mean(axis=0), outer_stage1=outer, outer_stage2=outer)
            np.savez(p, **values)
            validate_prediction(p, ['a', 'b'])
            with self.assertRaises(ValueError):
                validate_prediction(p, ['b', 'a'])
            values['stage2'] = np.array([1., 2.1])
            np.savez(p, **values)
            with self.assertRaises(ValueError):
                validate_prediction(p, ['a', 'b'])

    def test_table_matches_evaluated_metrics(self):
        rows = [line for line in render('external').splitlines() if line.startswith('| ')]
        self.assertEqual(len(rows), len(COHORTS) + 1)
        for slug, row in zip(COHORTS, rows[1:]):
            result = evaluate(slug)
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            self.assertEqual(cells[0], result['dataset'])
            self.assertEqual(int(cells[1]), result['n'])
            self.assertEqual(cells[2], result['unit'])
            for column, model, metric, places in (
                (3, 'stage1', 'mae', 8), (4, 'stage2', 'mae', 8),
                (6, 'stage1', 'rmse', 8), (7, 'stage2', 'rmse', 8),
                (8, 'stage1', 'r2', 6), (9, 'stage2', 'r2', 6),
            ):
                self.assertAlmostEqual(float(cells[column]),
                                       result['metrics'][model][metric], places=places)

    def test_evaluation_is_read_only(self):
        files = [ROOT / 'results/external' / slug / path for slug in COHORTS
                 for path in ('predictions/final/predictions.npz', 'results/metrics.json',
                              'data/labels.csv', 'results/per_sample_frozen.csv')]
        before = {str(p): sha256(p) for p in files}
        for slug in COHORTS:
            evaluate(slug)
        self.assertEqual(before, {str(p): sha256(p) for p in files})

    def test_shared_label_metadata_preserves_sample_population(self):
        base = ROOT / 'results/external/hea95'
        cfg = json.loads((base / 'config.json').read_text())
        groups = json.loads((base / 'data/shared_label_groups.json').read_text())
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(len(group) == 2 and set(group) <= set(cfg['sample_ids']) for group in groups))
        self.assertEqual(cfg['evaluation_samples'], 95)

if __name__ == '__main__':
    unittest.main()
