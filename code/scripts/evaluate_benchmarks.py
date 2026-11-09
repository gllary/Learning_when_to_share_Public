"""Verify selected benchmark metrics against fixed per-sample predictions."""
import json
import importlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hpsafemoe.stage2_calibration.matbench import AUCPlan

def read(p):return json.loads(p.read_text())
def task_score(y,p,metric):
    if metric.upper()=='ACC':return float(np.mean((p>.5)==y))
    if 'AUC' in metric.upper():return AUCPlan(y,p).score()
    return float(np.mean(np.abs(y-p)))

def main():
    checks=[];prediction_rows=0
    for ds in ('jarvis','matbench'):
        calibration=importlib.import_module('hpsafemoe.stage2_calibration.'+ds)
        tasks=read(ROOT/f'results/tables/{ds}.json')['tasks']
        arms=[r for r in read(ROOT/'manifests/benchmark_experiments.json') if r['dataset']==ds]
        for arm in arms:
            model=arm['result_key']
            for t in tasks:
                name=t['task'];root=ROOT/arm['directory']/'predictions';values=[];basevalues=[]
                folds=[None] if ds=='jarvis' else range(5)
                for outer in folds:
                    p=root/f'{name}.npz' if outer is None else root/f'outer_{outer}/{name}.npz'
                    with np.load(p,allow_pickle=False) as z:
                        ids=z['sample_ids'].astype(str).tolist();a=z['anchor_prediction'].astype(np.float64);q=z['prediction'].astype(np.float64)
                        raw=z['raw_prediction'].astype(np.float64);saved_beta=z['beta']
                    certpath=ROOT/arm['directory']/'calibration'/p.relative_to(root).with_suffix('.json')
                    beta=read(certpath)['beta'];assert np.all(saved_beta==beta)
                    np.testing.assert_array_equal(q,calibration.apply_beta(a,raw,beta))
                    if ds=='jarvis':
                        assert ids==read(ROOT/f'data/jarvis/stage1_pool/data/tasks/{name}/manifest.json')['role_ids']['test']
                        labels=read(ROOT/f'data/jarvis/stage1_pool/data/tasks/{name}/test.json')
                        y=np.asarray([labels[i] for i in ids],dtype=np.float64)
                    else:
                        path=ROOT/f'data/matbench/raw/manifests/{name}.npz'
                        with np.load(path,allow_pickle=False) as z:
                            idkey='sample_ids' if 'sample_ids' in z.files else 'ids'
                            keys=z[idkey].astype(str).tolist();lookup={s:i for i,s in enumerate(keys)}
                            assert ids==[keys[i] for i in z[f'outer_{outer}_test']]
                            y=z['targets'][[lookup[s] for s in ids]].astype(np.float64)
                    prediction_rows+=len(ids)
                    values.append(task_score(y,q,t['metric']));basevalues.append(task_score(y,a,t['metric']))
                actual=float(np.mean(values));expected=t['models'][model]['score']
                assert np.isclose(actual,expected,rtol=1e-11,atol=1e-12),(ds,model,name,actual,expected)
                assert np.isclose(np.mean(basevalues),t['stage1'],rtol=1e-11,atol=1e-12)
                checks.append({'dataset':ds,'model':model,'task':name,'score':actual})
    print(json.dumps({'passed':True,'metric_cells':len(checks),'prediction_rows':prediction_rows,'ids_and_beta_application_exact':True,'new_model_inference':False,'published_results_modified':False}))
if __name__=='__main__':main()
