"""Redraw the current English numerical figures using only released data.

Run from the repository root. --figures uses current English figure numbers;
--output must be a new directory under runs/. Figure numbers match the manuscript.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from analysis_coverage import accepted_count
ROOT = Path(__file__).resolve().parents[2]
ORDER = ['expt_is_metal', 'glass', 'mp_is_metal', 'dielectric', 'expt_gap', 'jdft2d', 'log_gvrh', 'log_kvrh', 'mp_e_form', 'mp_gap', 'perovskites', 'phonons', 'steels']
GROUPS = ['composition_stoichiometry', 'electronic_chemistry', 'size_mass', 'structure_geometry_density', 'symmetry', 'target_property']
GLABELS = ['Composition and\nstoichiometry', 'Electronic and\nchemical\nproperties', 'Atomic size\nand mass', 'Structural\ngeometry\nand density', 'Crystal\nsymmetry', 'Target\nproperty']
STEMS = {'fig2': 'fig2_source_allocation', 'fig3': 'fig3_ablation', 'fig4': 'fig4_external_errors', 's1': 'matbench_fold_boxplots', 's2': 'supp_fig_s2_source_allocation', 's3': 'supp_fig_s3_descriptor_cka', 's4': 'supp_fig_s4_pca', 's5': 'supp_fig_s5_permutation_reference'}
LABELS = {'fig2': 'Fig. 2', 'fig3': 'Fig. 3', 'fig4': 'Fig. 4', **{f's{i}': f'Supplementary Fig. {i}' for i in range(1, 6)}}
BLUE = '#247DA8'
TEAL = '#168579'
ORANGE = '#8063A2'
AMBER = '#CE8842'
INK = '#26343E'
HEAT = LinearSegmentedColormap.from_list('HP_SafeMoE', ['#F0F6F7', '#A9CEDC', '#4A9EB9', '#17698F', '#073D68'])
INPUT_FILES = set()

def read_json(relative):
    path = ROOT / relative
    INPUT_FILES.add(relative)
    return json.loads(path.read_text())

def read_csv(relative):
    path = ROOT / relative
    INPUT_FILES.add(relative)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def input_name(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

def save_csv(name, rows):
    if not rows:
        return
    with (DATA / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def num(value):
    return float(value)

def load_inputs():
    global MB, JV, EXT, FOLDS, BETA, WEIGHTS, CKA, PCA, PRED, NAMES
    matbench = {r['task']: r for r in read_json('results/tables/matbench.json')['tasks']}
    jarvis = {r['task']: r for r in read_json('results/tables/jarvis.json')['tasks']}
    WEIGHTS = {ds: read_csv(f'results/analysis/source_weights/{ds}.csv') for ds in ['matbench', 'jarvis']}
    jorder = list(dict.fromkeys((r['target_task'] for r in WEIGHTS['jarvis'])))
    MB = [{'task': t, 'RawFull_gain': matbench[t]['models']['NO_SAFETY_CALIBRATION']['normalized_gain_percent'], 'Full_gain': matbench[t]['models']['FULL']['normalized_gain_percent']} for t in ORDER]
    JV = [{'task': t, 'stage1_provider': jarvis[t]['stage1_model']} for t in jorder]
    FOLDS, BETA = ([], [])
    for t in ORDER:
        for model, arm in [('FULL', 'Full'), ('NO_CROSS', 'NoCross')]:
            gains = matbench[t]['models'][model]['fold_gains_percent']
            if len(gains) != 5:
                raise ValueError(f'Expected five outer folds: {t}/{model}')
            FOLDS.extend(({'dataset': 'matbench', 'task': t, 'arm': arm, 'outer_fold': i + 1, 'normalized_gain_percent': g} for i, g in enumerate(gains)))
        for i in range(5):
            c = read_json(f'results/matbench/full/calibration/outer_{i}/{t}.json')
            BETA.append({'dataset': 'matbench', 'task': t, 'outer_fold': i + 1, 'beta': c['beta']})
    for t in jorder:
        c = read_json(f'results/jarvis/full/calibration/{t}.json')
        BETA.append({'dataset': 'jarvis', 'task': t, 'outer_fold': '', 'beta': c['beta']})
    if any((r['beta'] not in (0, 0.25, 0.5, 0.75, 1) for r in BETA)):
        raise ValueError('Unexpected calibration coefficient')
    PRED = {slug: read_csv(f'results/external/{slug}/results/per_sample_frozen.csv') for slug in ['ds2_248', 'core23', 'hea95', 'b2_18']}
    EXT = [{'slug': slug, 'dataset': name} for slug, name in [('ds2_248', 'DS2-248'), ('core23', 'Core23'), ('hea95', 'HEA95'), ('b2_18', 'B2-18')]]
    NAMES = {t: 'matbench_' + t for t in ORDER}
    NAMES.update({t: t[4:] + ' (R)' if t.startswith('reg_') else t[4:] + ' (C)' for t in jorder})

def coverage_records(dataset):
    tasks = ORDER if dataset == 'matbench' else [r['task'] for r in JV]
    rows = []
    for task in tasks:
        n = accepted = 0
        for outer in range(5) if dataset == 'matbench' else [None]:
            rel = f'outer_{outer}/{task}' if outer is not None else task
            path = f'results/{dataset}/full/predictions/{rel}.npz'
            INPUT_FILES.add(path)
            with np.load(ROOT / path, allow_pickle=False) as z:
                size = len(z['sample_ids'])
                beta = np.asarray(z['beta'])
                expected = next((r['beta'] for r in BETA if r['dataset'] == dataset and r['task'] == task and (r['outer_fold'] == (outer + 1 if outer is not None else ''))))
                if not np.all(beta == expected):
                    raise ValueError(f'Prediction/calibration coefficient mismatch: {rel}')
                accepted += accepted_count(beta, size)
                n += size
        rows.append({'dataset': dataset, 'task': task, 'n': n, 'accepted_n': accepted, 'coverage_percent': 100 * accepted / n, 'beta': expected if dataset == 'jarvis' else ''})
    return rows

def source_matrix(which, order):
    index = {t: i for i, t in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    for r in WEIGHTS[which]:
        value = r['normalized_source_percent']
        matrix[index[r['target_task']], index[r['source_task']]] = float(value) if value else np.nan
    for row in matrix:
        if np.isfinite(row).any():
            np.testing.assert_allclose(np.nansum(row), 100, atol=1e-06, rtol=0)
    return matrix

def save(fig, stem, meta):
    key = stem
    for ext in FORMATS:
        metadata = {'Date': None} if ext == 'svg' else {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
        fig.savefig(FIG / (STEMS[key] + '.' + ext), dpi=DPI, bbox_inches=None if key in ('s1', 's5') else 'tight', pad_inches=0 if key in ('s1', 's5') else 0.12, metadata=metadata)
    plt.close(fig)

def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#B7C3CA')
    ax.tick_params(length=2, color='#B7C3CA')

def heat(ax, a, rows, cols, vmax=100, annot=0, fontsize=None):
    cm = HEAT.copy()
    cm.set_bad('#E7EAED')
    im = ax.imshow(a, cmap=cm, vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(cols)), cols)
    ax.tick_params(length=0, labelsize=9)
    ax.spines[:].set_visible(False)
    fs = fontsize or (6.2 if a.shape[1] > 25 else 8)
    ax.set_xticks(np.arange(-0.5, a.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, a.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.48)
    ax.tick_params(which='minor', length=0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            label = 'N/A' if not np.isfinite(v) else '0' if v == 0 else f'{v:.1f}' if vmax > 1 else f'{v:.2f}'
            ax.text(j, i, label, ha='center', va='center', fontsize=fs, color='white' if np.isfinite(v) and v > vmax * 0.52 else INK)
    return im

def fig2():
    a = source_matrix('matbench', ORDER)
    labels = [f'{i + 1:02d}  {NAMES[t]}' for i, t in enumerate(ORDER)]
    fig = plt.figure(figsize=(12.2, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.3], wspace=0.2)
    ax = fig.add_subplot(gs[0])
    im = heat(ax, a, labels, [f'{i + 1:02d}' for i in range(13)], annot=5)
    ax.set_xlabel('Source expert index (matched to target task index)')
    ax.set_ylabel('Target task')
    ax.set_title('a  Non-target source weights', loc='left', pad=12)
    cb = fig.colorbar(im, cax=fig.add_axes([0.21, 0.075, 0.6, 0.022]), orientation='horizontal')
    cb.set_label('Calibrated normalized source weights (%)')
    coverage = coverage_records('matbench')
    bx = fig.add_subplot(gs[1])
    c = np.array([num(r['coverage_percent']) for r in coverage])
    bx.barh(range(13), c, color=TEAL, height=0.68)
    bx.set_ylim(12.5, -0.5)
    bx.set_yticks([])
    bx.set_xlim(0, 121)
    bx.set_xticks([0, 50, 100])
    bx.set_xlabel('Coverage (%)')
    bx.set_title('b  Coverage', loc='left', pad=12)
    clean(bx)
    for i, v in enumerate(c):
        bx.text(v + 2, i, f'{v:.1f}', va='center', fontsize=9)
    fig.subplots_adjust(left=0.21, right=0.97, top=0.91, bottom=0.24)
    save(fig, 'fig2', {'matrix_shape': list(a.shape), 'test_records': sum((r['n'] for r in coverage)), 'all_cells_annotated': True, 'zero_mass_rows': [ORDER[i] for i in range(13) if not np.isfinite(a[i]).any()]})
    save_csv('fig2_source_weights.csv', WEIGHTS['matbench'])
    save_csv('fig2_coverage.csv', coverage)

def fig3():
    lookup = {(r['task'], r['arm'], int(r['outer_fold'])): num(r['normalized_gain_percent']) for r in FOLDS if r['dataset'] == 'matbench'}
    delta = np.array([[lookup[t, 'Full', f] - lookup[t, 'NoCross', f] for f in range(1, 6)] for t in ORDER])
    raw = np.array([num(r['RawFull_gain']) for r in MB])
    full = np.array([num(r['Full_gain']) for r in MB])
    beta = np.array([[num(next((r['beta'] for r in BETA if r['dataset'] == 'matbench' and r['task'] == t and (int(r['outer_fold']) == f)))) for f in range(1, 6)] for t in ORDER])
    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 0.82], wspace=0.23)
    ax = fig.add_subplot(gs[0])
    ax.axvline(0, c='#98A5AF', lw=0.8, ls='--')
    means = delta.mean(1)
    ax.barh(range(13), means, color=np.where(means >= 0, TEAL, AMBER), height=0.64, xerr=np.stack([means - delta.min(1), delta.max(1) - means]), error_kw={'ecolor': '#43545F', 'elinewidth': 1.0, 'capsize': 3, 'capthick': 1.0})

    def labelnum(v):
        return f'{v:.4f}' if 0 < abs(v) < 0.01 else f'{v:.3f}'
    for i, v in enumerate(means):
        end = delta[i].max() + 0.65 if v >= 0 else delta[i].min() - 0.65
        ax.text(end, i, labelnum(v), va='center', ha='left' if v >= 0 else 'right', fontsize=8.4, color=INK)
    ax.set_xlim(min(-6, delta.min() - 5), delta.max() + 8)
    ax.set_yticks(range(13), [NAMES[t] for t in ORDER])
    ax.set_ylim(12.5, -0.5)
    ax.set_xlabel('Full − No-cross\nNormalized gain increment\n(percentage points)')
    ax.set_title('a  Gains from cross-model\n    sharing', loc='left', fontsize=11)
    clean(ax)
    ax.text(0.98, 0.02, 'Bars: five-fold mean\nError bars: five-fold range', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    bx = fig.add_subplot(gs[1])
    bx.axvline(0, c='#98A5AF', lw=0.8, ls='--')
    bx.barh(np.arange(13) - 0.18, raw, color=AMBER, height=0.34, label='Uncalibrated')
    bx.barh(np.arange(13) + 0.18, full, color=BLUE, height=0.34, label='Full')
    bx.set_ylim(12.5, -0.5)
    bx.set_yticks(range(13), [])
    bx.set_xlabel('Normalized gain relative to Stage 1 (%)')
    bx.set_title('b  Calibration of the same\n    residual proposal', loc='left', fontsize=11)
    bx.legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, -0.24), ncol=2)
    clean(bx)
    for values, offset in [(raw, -0.18), (full, 0.18)]:
        for i, v in enumerate(values):
            bx.text(v + (0.65 if v >= 0 else -0.65), i + offset, labelnum(v), va='center', ha='left' if v >= 0 else 'right', fontsize=8.1, color=INK)
    bx.set_xlim(min(raw.min(), full.min()) - 8, max(raw.max(), full.max()) + 8)
    cx = fig.add_subplot(gs[2])
    im = heat(cx, beta, [''] * 13, [str(i) for i in range(1, 6)], vmax=1, annot=0)
    cx.set_xlabel('Outer test fold')
    cx.set_title('c  Task-level calibration\n    coefficients', loc='left', fontsize=11)
    cb = fig.colorbar(im, cax=fig.add_axes([0.752, 0.065, 0.228, 0.022]), orientation='horizontal', ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label('$\\beta$')
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.92)
    save(fig, 'fig3', {'fold_points': 65, 'full_minus_nocross_macro_pp': float(delta.mean()), 'beta_shape': list(beta.shape), 'no_significance_testing': True})
    save_csv('fig3_fold_gain_differences.csv', [{'task': t, 'outer_fold': j + 1, 'full_minus_nocross_pp': delta[i, j], 'beta': beta[i, j]} for i, t in enumerate(ORDER) for j in range(5)])
    panels = {'panel_a': [{'task': t, 'mean_full_minus_nocross_pp': float(means[i]), 'fold_min': float(delta[i].min()), 'fold_max': float(delta[i].max()), 'fold_values': delta[i].tolist()} for i, t in enumerate(ORDER)], 'panel_b': [{'task': t, 'raw_full_gain_percent': float(raw[i]), 'full_gain_percent': float(full[i])} for i, t in enumerate(ORDER)], 'panel_c': [{'task': t, 'beta_by_outer_fold': beta[i].tolist()} for i, t in enumerate(ORDER)]}
    (DATA / 'fig3_plot_data.json').write_text(json.dumps(panels, ensure_ascii=False, indent=2))
    save_csv('fig3_calibration_gain.csv', panels['panel_b'])
BOX_TASKS = [('expt_is_metal', 'Experimental metallicity', 'ROC–AUC', 3), ('glass', 'Glass-forming ability', 'ROC–AUC', 3), ('mp_is_metal', 'DFT metallicity', 'ROC–AUC', 3), ('dielectric', 'Refractive index', 'MAE', 2), ('expt_gap', 'Experimental band gap', 'MAE (eV)', 2), ('jdft2d', 'Exfoliation energy', 'MAE (meV/atom)', 0), ('log_gvrh', 'Shear modulus', 'MAE [log$_{10}$(G/GPa)]', 3), ('log_kvrh', 'Bulk modulus', 'MAE [log$_{10}$(K/GPa)]', 3), ('mp_e_form', 'Formation energy', 'MAE (eV/atom)', 4), ('mp_gap', 'DFT PBE band gap', 'MAE (eV)', 3), ('perovskites', 'Perovskite formation energy', 'MAE (eV/cell)', 3), ('phonons', 'Last phonon DOS peak', 'MAE (cm$^{-1}$)', 0), ('steels', 'Steel yield strength', 'MAE (MPa)', 0)]
BOX_COLORS = ['#7C8794', '#247DA8']
BOX_NAMES = ['Target model', 'HP-SafeMoE']
BOX_VARIANTS = ['STAGE1', 'FULL']

def point_offsets(values, span):
    """Separate nearby points horizontally; leave all metric values unchanged."""
    offsets = np.zeros(len(values))
    placed = []
    for i in np.argsort(values, kind='stable'):
        for lane in [0.0, 0.075, -0.075, 0.15, -0.15]:
            if all((abs(values[i] - values[j]) > 0.065 * span or abs(lane - offsets[j]) >= 0.074 for j in placed)):
                offsets[i] = lane
                break
        placed.append(i)
    return offsets

def fold_boxplots():
    from evaluate_benchmarks import task_score
    source_table = {r['task']: r for r in read_json('results/tables/matbench.json')['tasks']}
    data = {}
    for task in ORDER:
        manifest_path = f'data/matbench/raw/manifests/{task}.npz'
        INPUT_FILES.add(manifest_path)
        with np.load(ROOT / manifest_path, allow_pickle=False) as z:
            ids = z['sample_ids'].astype(str)
            targets = z['targets'].astype(float)
            folds = [z[f'outer_{i}_test'] for i in range(5)]
        for variant in BOX_VARIANTS:
            data[f'{task}/{variant}'] = []
        for outer, positions in enumerate(folds):
            path = f'results/matbench/full/predictions/outer_{outer}/{task}.npz'
            INPUT_FILES.add(path)
            with np.load(ROOT / path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['sample_ids'].astype(str), ids[positions])
                for variant, key in [('STAGE1', 'anchor_prediction'), ('FULL', 'prediction')]:
                    data[f'{task}/{variant}'].append(task_score(targets[positions], z[key].astype(float), source_table[task]['metric']))
    with plt.rc_context():
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 8.5, 'text.color': '#253442', 'axes.labelcolor': '#354453', 'xtick.color': '#4A5562', 'ytick.color': '#4A5562', 'axes.linewidth': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.default': 'regular'})
        fig = plt.figure(figsize=(9.1, 11.7), facecolor='white')
        grid = fig.add_gridspec(5, 3, left=0.085, right=0.985, bottom=0.05, top=0.934, hspace=0.75, wspace=0.49)
        rows, summaries = ([], [])
        for index, (task, name, metric_label, decimals) in enumerate(BOX_TASKS):
            ax = fig.add_subplot(grid[index // 3, index % 3])
            values = [np.array(data[f'{task}/{variant}'], dtype=float) for variant in BOX_VARIANTS]
            assert all((v.shape == (5,) and np.isfinite(v).all() for v in values))
            assert np.isclose(values[0].mean(), source_table[task]['stage1'], atol=1e-12, rtol=0)
            assert np.isclose(values[1].mean(), source_table[task]['models']['FULL']['score'], atol=1e-12, rtol=0)
            all_vals = np.concatenate(values)
            span = float(np.ptp(all_vals))
            for pos, (variant, vals, color) in enumerate(zip(BOX_VARIANTS, values, BOX_COLORS)):
                quartiles = np.quantile(vals, [0.25, 0.5, 0.75], method='linear')
                stats = {'q1': quartiles[0], 'med': quartiles[1], 'q3': quartiles[2], 'whislo': vals.min(), 'whishi': vals.max(), 'fliers': []}
                ax.bxp([stats], positions=[pos], widths=0.42, patch_artist=True, manage_ticks=False, showfliers=False, boxprops={'facecolor': to_rgba(color, 0.2), 'edgecolor': color, 'linewidth': 0.95, 'zorder': 3}, medianprops={'color': color, 'linewidth': 1.4, 'zorder': 4}, whiskerprops={'color': color, 'linewidth': 0.75}, capprops={'color': color, 'linewidth': 0.75})
                ax.scatter(pos + point_offsets(vals, span), vals, s=23, color=color, edgecolor='white', linewidth=0.55, zorder=5)
                for fold, score in enumerate(vals):
                    rows.append({'task': 'matbench_' + task, 'variant': variant, 'method': BOX_NAMES[pos], 'outer_fold': fold, 'metric': source_table[task]['metric'], 'unit': source_table[task]['unit'], 'score': float(score)})
                summaries.append({'task': task, 'variant': variant, 'mean': float(vals.mean()), 'sd_ddof_1': float(vals.std(ddof=1)), 'min': float(vals.min()), 'q1': float(quartiles[0]), 'median': float(quartiles[1]), 'q3': float(quartiles[2]), 'max': float(vals.max())})
            ax.set_ylim(float(all_vals.min() - span * 0.14), float(all_vals.max() + span * 0.16))
            ax.set_xlim(-0.49, 1.49)
            ax.set_xticks([0, 1], BOX_NAMES)
            ax.tick_params(axis='x', length=0, pad=7)
            ax.tick_params(axis='y', length=3, width=0.6, pad=3)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals}f'))
            ax.set_ylabel(metric_label, labelpad=5)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#BCC6CE')
            ax.text(-0.23, 1.25, chr(97 + index), transform=ax.transAxes, fontsize=12.5, fontweight='bold', va='bottom')
            ax.text(0, 1.25, name, transform=ax.transAxes, fontsize=10.3, fontweight='bold', va='bottom')
            ax.text(0, 1.105, 'matbench_' + task, transform=ax.transAxes, fontsize=8.7, color='#77838F', va='bottom')
        legend = fig.add_subplot(grid[4, 1:])
        legend.axis('off')
        handles = [Patch(facecolor=to_rgba(c, 0.2), edgecolor=c, label=n) for c, n in zip(BOX_COLORS, BOX_NAMES)]
        legend.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.55), frameon=False, ncol=2, fontsize=10, handlelength=1.8, columnspacing=2.2)
        save(fig, 's1', {'tasks': len(BOX_TASKS), 'folds_per_method': 5, 'points': len(rows), 'quantile_method': 'linear', 'whiskers': 'minimum and maximum'})
    save_csv('supplementary_fig_1_source.csv', rows)
    (DATA / 'supplementary_fig_1_box_statistics.json').write_text(json.dumps(summaries, indent=2) + '\n')

def supp2():
    order = [r['task'] for r in JV]
    a = source_matrix('jarvis', order)
    fig = plt.figure(figsize=(15.8, 12.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[9.5, 0.65, 1.45], wspace=0.13)
    ax = fig.add_subplot(gs[0])
    labels = [f'{i + 1:02d}  ' + NAMES[t] for i, t in enumerate(order)]
    im = heat(ax, a, labels, [f'{i + 1:02d}' for i in range(34)], vmax=20, annot=10)
    ax.tick_params(axis='x', labelsize=7, rotation=90)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Source expert index (matched to target task index)')
    ax.set_title('a  JARVIS-DFT non-target source weights', loc='left', pad=14)
    cb = fig.colorbar(im, cax=fig.add_axes([0.28, 0.08, 0.56, 0.018]), orientation='horizontal')
    cb.set_label('Calibrated normalized source weights (%)')
    coverage = coverage_records('jarvis')
    beta = [num(r['beta']) for r in coverage]
    bx = fig.add_subplot(gs[1])
    heat(bx, np.array(beta)[:, None], [''] * 34, ['$\\beta$'], vmax=1, annot=0)
    bx.set_title('b\nCalibration', loc='center', pad=14, fontsize=10)
    cx = fig.add_subplot(gs[2])
    cover = [num(r['coverage_percent']) for r in coverage]
    cx.barh(range(34), cover, color=TEAL, height=0.7)
    cx.set_ylim(33.5, -0.5)
    cx.set_yticks([])
    cx.set_xlim(0, 124)
    cx.set_xticks([0, 100])
    cx.set_xlabel('Coverage (%)')
    cx.set_title('c  Coverage', loc='center', pad=14, fontsize=10)
    clean(cx)
    for i, v in enumerate(cover):
        cx.text(v + 2, i, f'{v:.0f}', va='center', fontsize=8)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.22, top=0.94)
    save(fig, 's2', {'matrix_shape': list(a.shape), 'test_records': sum((r['n'] for r in coverage)), 'all_cells_annotated': True, 'color_scale_percent': [0, 20], 'zero_mass_rows': [order[i] for i in range(34) if not np.isfinite(a[i]).any()]})
    save_csv('supplementary_fig_2_task_key.csv', [{'number': i + 1, 'task': r['task'], 'name': NAMES[r['task']], 'Stage1_model': r['stage1_provider']} for i, r in enumerate(JV)])
    save_csv('supplementary_fig_2_source_weights.csv', WEIGHTS['jarvis'])
    save_csv('supplementary_fig_2_calibration_coverage.csv', coverage)

def fig4():
    fig, axs = plt.subplots(2, 4, figsize=(15.1, 7.7))
    units = ['eV', 'dimensionless', '$\\log_{10}(E/\\mathrm{GPa})$', '$\\log_{10}(E/\\mathrm{GPa})$']
    names = ['Experimental band gap', 'Refractive index', "Derived Young's modulus", "Derived Young's modulus"]
    audit = []
    for i, r in enumerate(EXT):
        rows = PRED[r['slug']]
        y = np.array([num(v['target']) for v in rows])
        s = np.array([num(v['stage1']) for v in rows])
        f = np.array([num(v['stage2']) for v in rows])
        a = axs[0, i]
        errors = [np.abs(s - y), np.abs(f - y)]
        viol = a.violinplot(errors, positions=[0, 1], widths=0.78, showmeans=False, showmedians=False, showextrema=False, bw_method='scott')
        for body, col in zip(viol['bodies'], [ORANGE, BLUE]):
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(0.88)
        a.boxplot(errors, positions=[0, 1], widths=0.16, showfliers=False, patch_artist=True, boxprops={'facecolor': 'white', 'edgecolor': '#33454F', 'linewidth': 0.8}, medianprops={'color': '#1E2E38', 'linewidth': 1.6}, whiskerprops={'color': '#33454F', 'linewidth': 0.8}, capprops={'color': '#33454F', 'linewidth': 0.8})
        a.set_xticks([0, 1], ['Stage 1', 'Full'])
        a.set_ylabel('Absolute error (' + units[i] + ')', fontfamily=FONT)
        a.set_ylim(bottom=0)
        a.set_title(chr(97 + i) + '  ' + r['dataset'] + '\n' + names[i] + ', n = ' + str(len(y)), loc='left', fontsize=11)
        clean(a)
        b = axs[1, i]
        delta = np.abs(f - y) - np.abs(s - y)
        idx = np.argsort(y, kind='stable')
        b.axhline(0, c='#8B9DA9', lw=0.8)
        b.bar(np.arange(1, len(y) + 1), delta[idx], width=0.92, color=np.where(delta[idx] < 0, TEAL, AMBER), linewidth=0)
        b.set_xlim(0.3, len(y) + 0.7)
        b.set_xlabel('Samples ordered by\nincreasing target value')
        b.set_ylabel('Paired change in absolute error\n(' + units[i] + ')', fontfamily=FONT)
        b.set_title(chr(101 + i) + '  Sample-wise correction effects', loc='left', fontsize=11)
        clean(b)
        b.text(0.98, 0.03, f'Error reduced: {int((delta < 0).sum())}/{len(y)}', transform=b.transAxes, ha='right', va='bottom', fontsize=9)
        audit.append({'dataset': r['dataset'], 'n': len(y), 'mae_stage1': float(np.mean(np.abs(s - y))), 'mae_full': float(np.mean(np.abs(f - y))), 'improved': int((delta < 0).sum())})
        save_csv('fig4_' + r['slug'] + '.csv', rows)
    fig.tight_layout(h_pad=2.4, w_pad=2.3)
    save(fig, 'fig4', {'datasets': audit, 'all_samples_shown_in_paired_bars': True, 'top': 'violin and IQR box'})
import pandas as pd
from matplotlib.ticker import MaxNLocator
from recompute_representation_analysis import load_protocol, CLASSIFICATION, GROUPS as FULL_GROUPS, GROUP_LABELS as FULL_LABELS, write_csv as full_write_csv, write_json as full_write_json
FULL_STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'HP-SafeMoE-full-data-v1', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#26343E', 'text.color': '#26343E', 'axes.unicode_minus': False}

def save_full(fig, name, source, details):
    key = 's' + name.rsplit('_', 1)[1]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b = artist.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or (b.y1 > fig.bbox.height + 1):
                outside.append(artist.get_text())
    assert not outside, (name, outside)
    for ext in [e for e in ['pdf', 'svg', 'png'] if e in FORMATS]:
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else {}
        fig.savefig(FIG / (STEMS[key] + '.' + ext), dpi=DPI, metadata=meta)
    INPUT_FILES.add(input_name(REPRESENTATION_ROOT / source))
    plt.close(fig)

def full_figure(number):
    with plt.rc_context({**matplotlib.rcParamsDefault, **FULL_STYLE}):
        if number == 4:
            for task in ORDER:
                INPUT_FILES.add(input_name(REPRESENTATION_DATA / 'pca' / f'{task}.csv.gz'))
            pca_plot()
        else:
            summary = pd.read_csv(REPRESENTATION_ROOT / 'representation_summary.csv')
            (descriptor_cka if number == 3 else permutation_plot)(summary)

def descriptor_cka(summary):
    groups = list(FULL_GROUPS) + ['target_property']
    mat = summary.pivot(index='task', columns='descriptor_group', values='linear_cka_mean').reindex(index=ORDER, columns=groups).to_numpy()
    fig = plt.figure(figsize=(8.2, 6.5))
    ax = fig.add_axes([0.27, 0.23, 0.71, 0.64])
    cm = LinearSegmentedColormap.from_list('HPfull', ['#F0F5F7', '#8FBDCD', '#267E9B', '#0C4268'])
    cm.set_bad('#E7EBEE')
    im = ax.imshow(mat, cmap=cm, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(13), ['matbench_' + x for x in ORDER], fontsize=8)
    ax.set_xticks(range(6), FULL_LABELS, fontsize=8)
    ax.tick_params(length=0, pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    annotations = []
    for i in range(13):
        for j in range(6):
            val = mat[i, j]
            label = f'{val:.2f}' if np.isfinite(val) else 'N/A'
            annotations.append(label)
            ax.text(j, i, label, ha='center', va='center', fontsize=8, color='white' if val > 0.6 else '#26343E')
    fig.text(0.27, 0.945, 'Association of frozen expert representations\nwith material information', fontsize=12, va='top')
    cb = fig.colorbar(im, cax=fig.add_axes([0.34, 0.085, 0.57, 0.025]), orientation='horizontal')
    cb.set_label('Mean linear CKA across five full test folds', fontsize=9)
    save_full(fig, 'supplementary_fig_3', 'representation_summary.csv', {'n_cells': 78, 'numeric_cells': int(np.isfinite(mat).sum()), 'na_cells': int(np.isnan(mat).sum()), 'annotations': annotations})
    full_write_csv(DATA / 'supplementary_fig_3_source.csv', summary[['task', 'descriptor_group', 'n_total', 'linear_cka_mean', 'linear_cka_sd', 'status']])

def pca_plot():
    cfg = load_protocol()
    fig, axes = plt.subplots(5, 3, figsize=(9.5, 11.7), layout='constrained')
    count = 0
    panels = []
    for i, task in enumerate(ORDER):
        ax = axes.flat[i]
        d = pd.read_csv(REPRESENTATION_DATA / 'pca' / f'{task}.csv.gz')
        count += len(d)
        order = np.random.default_rng(20261108 + i).permutation(len(d))
        d = d.iloc[order]
        size = 8 if len(d) < 1500 else 3.5 if len(d) < 5000 else 1.6
        if task in CLASSIFICATION:
            for cl, color, marker in [(0, '#247DA8', 'o'), (1, '#8063A2', '^')]:
                v = d[d.target == cl]
                ax.scatter(v.pc1, v.pc2, c=color, s=size, marker=marker, alpha=0.6, linewidths=0, rasterized=True, label=f'Class {cl}')
            ax.legend(frameon=False, fontsize=6.8, markerscale=1.5, loc='best')
        else:
            s = ax.scatter(d.pc1, d.pc2, c=d.target, cmap='viridis', s=size, alpha=0.72, linewidths=0, rasterized=True)
            cbar = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
            cbar.ax.tick_params(labelsize=6.5)
        ax.set_title(f'{chr(97 + i)}  matbench_{task}\n(n = {len(d):,})', loc='left', fontsize=8)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticks([v for v in ax.get_xticks() if xlim[0] <= v <= xlim[1]])
        ax.set_yticks([v for v in ax.get_yticks() if ylim[0] <= v <= ylim[1]])
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#BCC9D0')
            ax.spines[side].set_linewidth(0.6)
        panels.append({'task': task, 'outer_fold': 0, 'records': len(d), 'all_records_plotted': True})
    for ax in axes.flat[13:]:
        ax.set_visible(False)
    save_full(fig, 'supplementary_fig_4', 'pca_metadata.csv', {'n_samples': count, 'panels': panels, 'all_first_outer_fold_records_plotted': True})
    full_write_json(DATA / 'supplementary_fig_4_source.json', {'inputs': [input_name(REPRESENTATION_DATA / 'pca' / f'{t}.csv.gz') for t in ORDER], 'panels': panels})

def permutation_plot(summary):
    from supp_fig5_heatmap import render_heatmap
    folds = pd.read_csv(REPRESENTATION_ROOT / 'representation_descriptor_cka.csv')
    details = render_heatmap(summary, folds, ORDER, FIG / STEMS['s5'], formats=FORMATS, dpi=S5_DPI)
    INPUT_FILES.add(input_name(REPRESENTATION_ROOT / 'representation_summary.csv'))
    INPUT_FILES.add(input_name(REPRESENTATION_ROOT / 'representation_descriptor_cka.csv'))
    full_write_csv(DATA / 'supplementary_fig_5_source.csv', summary[['task', 'descriptor_group', 'n_total', 'permutation_z_mean', 'permutation_z_sd', 'status']])
    full_write_csv(DATA / 'supplementary_fig_5_folds_source.csv', folds[['task', 'outer_fold', 'descriptor_group', 'n_samples', 'permutation_z', 'status']])

def main():
    global FIG, DATA, FORMATS, DPI, S5_DPI, FONT, REPRESENTATION_ROOT, REPRESENTATION_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', nargs='+', choices=['all', *STEMS], default=['all'])
    parser.add_argument('--formats', nargs='+', choices=['png', 'svg', 'pdf'], default=['png', 'svg', 'pdf'])
    parser.add_argument('--dpi', type=int, default=None, help='Default: 360 for s5, 350 for other figures')
    parser.add_argument('--representation-root', type=Path, help='Fresh recomputation directory under runs/ (optional)')
    args = parser.parse_args()
    REPRESENTATION_ROOT = args.representation_root.resolve() if args.representation_root else ROOT / 'results/analysis/representation'
    REPRESENTATION_DATA = REPRESENTATION_ROOT if args.representation_root else ROOT / 'data/analysis/representation'
    FIG = args.output.resolve()
    if not FIG.is_relative_to(ROOT / 'runs'):
        parser.error('Output must be a new directory under runs/')
    if args.dpi is not None and args.dpi <= 0:
        parser.error('--dpi must be positive')
    if FIG.exists():
        parser.error('Output directory already exists; choose a new path under runs/')
    FIG.mkdir(parents=True, exist_ok=False)
    DATA = FIG / 'plot_records'
    DATA.mkdir()
    FORMATS, DPI = (list(dict.fromkeys(args.formats)), args.dpi or 350)
    S5_DPI = args.dpi or 360
    try:
        FONT = fm.FontProperties(fname=fm.findfont('Arial', fallback_to_default=False)).get_name()
    except ValueError:
        FONT = 'DejaVu Sans'
    plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'lines.linewidth': 1.0, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'HP-SafeMoE-English-20261107', 'axes.unicode_minus': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    load_inputs()
    selected = list(STEMS) if 'all' in args.figures else list(dict.fromkeys(args.figures))
    functions = {'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 's1': fold_boxplots, 's2': supp2, 's3': lambda: full_figure(3), 's4': lambda: full_figure(4), 's5': lambda: full_figure(5)}
    for key in selected:
        if key in ('s3', 's4', 's5'):
            with plt.rc_context({'font.size': 8, 'mathtext.fontset': 'dejavusans', 'text.color': '#293139', 'axes.labelcolor': '#293139', 'xtick.color': '#59636C', 'ytick.color': '#293139'}):
                functions[key]()
        else:
            functions[key]()
        print(LABELS[key], flush=True)
    (DATA / 'figure_map.json').write_text(json.dumps({k: {'manuscript_figure': LABELS[k], 'files': [STEMS[k] + '.' + ext for ext in FORMATS]} for k in selected}, indent=2) + '\n')
    print(FIG)
if __name__ == '__main__':
    main()
