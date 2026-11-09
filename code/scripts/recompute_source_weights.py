"""Rebuild Fig. 2 / Supplementary Fig. 2 statistics from saved member products."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from analysis_coverage import accepted_count
ROOT = Path(__file__).resolve().parents[2]

def compute(dataset, record_root=None):
    base = ROOT / 'records/analysis/source_weights' / dataset if record_root is None else Path(record_root)
    columns = json.loads((ROOT / f'records/expert_and_routing/{dataset}/model/expert_columns.json').read_text())
    tasks = columns['task_order']
    rows = []
    checks = []
    for task in tasks:
        total = np.zeros(len(tasks))
        n = changed_n = adopted_n = 0
        for outer in range(5) if dataset == 'matbench' else [None]:
            rel = f'outer_{outer}/{task}' if outer is not None else task
            with np.load(ROOT / f'results/{dataset}/full/predictions/{rel}.npz', allow_pickle=False) as z:
                ids = z['sample_ids'].astype(str)
                changed = z['prediction'] != z['anchor_prediction']
                beta = float(np.ravel(z['beta'])[0])
                raw = z['raw_prediction']
            n += len(ids)
            changed_n += int(changed.sum())
            adopted_n += accepted_count(beta, len(ids))
            if dataset == 'matbench':
                with np.load(base / f'outer_{outer}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z[task + '_ids'].astype(str), ids)
                    np.testing.assert_array_equal(z[task + '_changed'], changed)
                    mass = z[task + '_source_mass']
                    assert mass.shape == (len(ids), len(tasks))
                    assert np.isfinite(mass).all() and (mass >= 0).all()
                    np.testing.assert_allclose(mass.sum(axis=1), beta, atol=2e-06, rtol=0)
                    assert np.count_nonzero(mass[:, tasks.index(task)]) == 0
                    total += mass[changed].sum(axis=0)
            elif beta:
                member_raw = []
                for member in range(10):
                    with np.load(base / f'member_{member}.npz', allow_pickle=False) as z:
                        mass = z[task + '_source_sum']
                        assert mass.shape == (len(tasks),)
                        assert np.isfinite(mass).all() and (mass >= 0).all()
                        np.testing.assert_allclose(mass.sum(), beta * int(changed.sum()), atol=0.002, rtol=2e-06)
                        assert mass[tasks.index(task)] == 0
                        total += mass / 10
                        member_raw.append(z[task + '_raw'])
                replay = np.mean(member_raw, axis=0)
                np.testing.assert_allclose(replay, raw, rtol=2e-05, atol=1e-05)
        percent = 100 * total / total.sum() if total.sum() > 0 else np.full(len(total), np.nan)
        for source, mass, value in zip(tasks, total, percent):
            rows.append(dict(target_task=task, source_task=source, diagnostic_n=n, changed_n=changed_n, calibrated_route_attention_mass=float(mass), normalized_source_percent=float(value)))
        top = np.argsort(-np.nan_to_num(percent, nan=-np.inf))[:3] if total.sum() > 0 else []
        checks.append({'task': task, 'n': n, 'adopted': adopted_n, 'coverage_percent': 100 * adopted_n / n, 'top3_sources': [tasks[i] for i in top], 'top3_percent': [float(percent[i]) for i in top], 'top3_sum_percent': float(sum((percent[i] for i in top))) if len(top) else None})
    return (pd.DataFrame(rows), checks)

def check(frame, dataset):
    expected = pd.read_csv(ROOT / f'results/analysis/source_weights/{dataset}.csv')
    keys = ['target_task', 'source_task']
    a = frame.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    assert list(a.columns) == list(b.columns) and a.shape == b.shape
    for k in keys:
        assert a[k].equals(b[k])
    np.testing.assert_allclose(a.drop(columns=keys), b.drop(columns=keys), rtol=1e-12, atol=1e-10, equal_nan=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['all', 'matbench', 'jarvis'], default='all')
    p.add_argument('--record-root', type=Path, help='Alternative cache for one dataset')
    p.add_argument('--check', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not args.check and args.output is None:
        p.error('Choose --check or --output')
    if args.record_root and args.dataset == 'all':
        p.error('--record-root requires one dataset')
    out = None
    if args.output:
        out = args.output.resolve()
        if not out.is_relative_to(ROOT / 'runs'):
            p.error('Output must be under runs/')
        out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for ds in ['matbench', 'jarvis'] if args.dataset == 'all' else [args.dataset]:
        frame, summary = compute(ds, args.record_root)
        if args.check:
            check(frame, ds)
        if out:
            frame.to_csv(out / f'{ds}.csv', index=False)
            (out / f'{ds}_task_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        counts[ds] = {'cells': len(frame), 'tasks': len(summary), 'test_records': sum((x['n'] for x in summary))}
    print(json.dumps({'passed': True, 'checked_archived_results': args.check, 'datasets': counts}, indent=2))
if __name__ == '__main__':
    main()
