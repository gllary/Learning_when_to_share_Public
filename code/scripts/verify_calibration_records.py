"""Check OOF calibration coefficients and bootstrap quantiles."""
import hashlib,importlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def main():
 count=rows=0
 for e in read(ROOT/'manifests/benchmark_experiments.json'):
  if e['safety']=='none':continue
  ds=e['dataset'];module=importlib.import_module('hpsafemoe.stage2_calibration.'+ds)
  base=ROOT/e['directory'];tasks={t['task']:t for t in read(ROOT/f'results/tables/{ds}.json')['tasks']}
  for path in (base/'calibration').rglob('*.json'):
   cert=read(path);task=cert['task'];classification=tasks[task]['metric'].lower() in ('acc','roc_auc','auc')
   rel=path.relative_to(base/'calibration').with_suffix('.npz');record=base/'calibration_records'/rel
   assert sha(record)==cert['calibration_record_sha256'],str(record)
   with np.load(record,allow_pickle=False) as z:
    a=z['anchor'];r=z['raw'];y=z['y_train_oof' if ds=='jarvis' else 'y_oof']
    assignment=z['meta_fold'];q=z['crossfit_prediction'];draws=z['bootstrap_gains']
   seed=cert['calibration_seed' if ds=='jarvis' else 'meta_seed']
   np.testing.assert_array_equal(assignment,module.meta_partitions(y,classification,seed))
   for fold_record in cert['meta_folds']:
    held=assignment==fold_record['held_fold'];fit=~held
    alpha,_=module.select_alpha(a[fit],r[fit],y[fit],classification)
    assert alpha==fold_record['alpha']
    np.testing.assert_array_equal(q[held],module.apply_beta(a[held],r[held],alpha))
   alpha,_=module.select_alpha(a,r,y,classification);assert alpha==cert['alpha_all_oof']
   if ds=='jarvis':gain=module.normalized_gain(module.losses(y,a,classification),module.losses(y,q,classification))
   else:gain=module.normalized_gain(module.score(y,a,classification),module.score(y,q,classification),classification)
   assert np.isclose(gain,cert['certificate']['G'],rtol=1e-12,atol=1e-12)
   assert len(draws)==5000 and np.isfinite(draws).all()
   assert float(np.quantile(draws,.05))==cert['certificate']['L']
   accepted=module.accept_liberal(gain,cert['certificate']['L'])
   assert cert['task_rule_passed']==accepted and cert['beta']==(alpha if accepted else 0)
   count+=1;rows+=len(y)
 assert count==164
 print(json.dumps({'passed':True,'records':count,'oof_rows':rows,'new_bootstrap_sampling':False,'test_labels_read':False,'published_results_modified':False}))
if __name__=='__main__':main()
