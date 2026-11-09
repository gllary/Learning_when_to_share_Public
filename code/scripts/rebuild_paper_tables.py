"""Render only selected final results; output to stdout or a new runs/ file."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def render(dataset):
    x=json.loads((ROOT/f'results/tables/{dataset}.json').read_text())
    if dataset=='external':
        assert x['evaluation_mode']=='frozen_external_validation'
        out=['# Zero-shot External Validation', '',
             'All cohort samples are evaluated. Model weights and benchmark task-level beta remain fixed.',
             'External labels are used only for evaluation. MAE and RMSE are lower-is-better; R2 is higher-is-better.', '',
             '| Dataset | N | Unit | Stage1 MAE | Final Stage2 MAE | MAE improvement (%) | Stage1 RMSE | Stage2 RMSE | Stage1 R2 | Stage2 R2 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,t in x['datasets'].items():
            assert t['evaluation_mode']=='frozen_external_validation' and t['external_calibration_samples']==0
            a=t['external_validation_metrics']['stage1'];b=t['external_validation_metrics']['stage2']
            out.append(f"| {name} | {t['n']} | {t['unit']} | {a['mae']:.8f} | {b['mae']:.8f} | {100*(a['mae']-b['mae'])/a['mae']:+.4f} | {a['rmse']:.8f} | {b['rmse']:.8f} | {a['r2']:.6f} | {b['r2']:.6f} |")
    else:
        names=list(x['tasks'][0]['models'])
        labels={'FULL':'Full','NO_CROSS':'No-cross Ablation','NO_SAFETY_CALIBRATION':'No-safety-calibration Ablation'}
        out=['| Task | Metric | Stage1 | '+' | '.join(labels[n] for n in names)+' |','|---|---|---:|'+'---:|'*len(names)]
        for t in x['tasks']:
            out.append('| '+t['task']+' | '+t['metric']+' | '+f"{t['stage1']:.8g}"+' | '+' | '.join(f"{t['models'][n]['score']:.8g}" for n in names)+' |')
    return '\n'.join(out)+'\n'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['matbench','jarvis','external']);a=p.parse_args();print(render(a.dataset))
