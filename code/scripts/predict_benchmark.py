"""Frozen benchmark inference with saved beta. No training or test-label reads."""
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2)

def label_free_fold(pool,outer,stats):
 from hpsafe_sota.data.manifest import TaskManifest
 from hpsafe_sota.expert_inputs.cache import TaskRoleCache,export_directory,validate_expert_export
 from hpsafe_sota.expert_inputs.data import FoldInputs
 from hpsafe_sota.expert_inputs.protocol import CANONICAL_TASKS
 class IdentityManifest(TaskManifest):
  def __init__(self,path):
   self.path=path
   with np.load(path,allow_pickle=False) as z:self._arrays={k:z[k] for k in z.files if k!='targets'}
   self.metadata=json.loads(str(self._arrays.pop('metadata_json')[0]));self.declared_digest=str(self._arrays.pop('semantic_digest')[0])
  @property
  def digest(self):return self.declared_digest
 fold=object.__new__(FoldInputs);fold.pool=pool;fold.tasks=CANONICAL_TASKS;fold.outer_fold=outer
 fold.task_index={task:i for i,task in enumerate(fold.tasks)};fold.task_types={};fold.roles={};fold.stats={}
 for task in fold.tasks:
  m=IdentityManifest(ROOT/f'data/matbench/manifests/{task}.npz');exports={}
  fold.task_types[task]=m.metadata['definition']['task_type']
  for spec in pool.experts:
   p=export_directory(spec,task,outer,'test')
   exports[spec.expert_id]=validate_expert_export(p,manifest=m,spec=spec,outer_fold=outer,role='test',verify_hashes=True) if (p/'metadata.json').exists() else None
  ids=m.ids('outer_test',outer);anchor=pool.anchor_for(task)
  assert exports[anchor] is not None and exports[anchor].available.all()
  fold.roles[task]={'test':TaskRoleCache(task,'test',outer,'outer_test',ids,np.zeros(len(ids)),anchor,exports)}
  fold.stats[task]=fold._parse_stats(task,stats[task])
 return fold

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,help='Experiment ID from the release registry')
 p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--outer',type=int,choices=range(5))
 args=p.parse_args();cfg=next(x for x in read(ROOT/'manifests/benchmark_experiments.json') if x['experiment_id']==args.experiment)
 output=Path(args.output).resolve();assert output.is_relative_to(ROOT/'runs');output.mkdir(parents=True,exist_ok=False)
 os.environ['HPSAFEMOE_RUN_DIR']=str(output)
 import torch
 torch.set_num_threads(4);ds=cfg['dataset'];arch=cfg['architecture'];base=ROOT/cfg['directory']
 records=read(ROOT/cfg['weights_manifest'])['models'];nc=cfg['strict_no_cross']
 if ds=='jarvis':
  assert not nc and cfg['safety']=='fixed_task_liberal'
  records=sorted(records,key=lambda r:r['path'])
 else:
  records=sorted([r for r in records if ('/ablation_no_cross_expert/' in r['path'])==nc],key=lambda r:r['path'])
 beta_files=sorted((base/'calibration').rglob('*.json'));locks={str(f.relative_to(ROOT)):sha(f) for f in beta_files}
 assert locks
 save(output/'stage2/selection_locked.json',{'published_beta_files':locks,'refitted':False,'test_labels_read':False})
 model_module=importlib.import_module(f'hpsafemoe.stage2_proposal.{ds}.{arch}.model')
 calibrator=importlib.import_module('hpsafemoe.stage2_calibration.'+ds)
 if ds=='jarvis':
  assert args.outer is None,'JARVIS has no outer index'
  inputs=importlib.import_module(f'hpsafemoe.stage2_proposal.jarvis.{arch}.input_statistics').Inputs()
  predict=importlib.import_module(f'hpsafemoe.stage2_proposal.jarvis.{arch}.predictor').predict
  outers=[None]
 else:
  from hpsafe_sota.expert_inputs.protocol import CANONICAL_TASKS,load_expert_pool
  pool=load_expert_pool(ROOT/'data/matbench/stage1_pool/pool/experts.yaml')
  module=importlib.import_module(f'hpsafemoe.stage2_proposal.matbench.{arch}.training')
  outers=range(5) if args.outer is None else [args.outer]
 commitments={}
 for outer in outers:
  members=records if outer is None else [r for r in records if f'/outer_{outer}/' in r['path']]
  assert len(members)==10;predictions={}
  for rec in members:
   path=ROOT/rec['path'];assert sha(path)==rec['sha256']
   q=torch.load(path,map_location='cpu',weights_only=False);mc=q.get('model_config',q.get('config'));state=q.get('state_dict',q.get('model'))
   model=model_module.CrossExpertProposalModel(model_module.ProposalConfig(**mc)).to(args.device)
   model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
   if ds=='jarvis':
    norm=q['normalization'];inputs.inner=q['inner'];inputs.stats={t:tuple(v) for t,v in norm['label_stats'].items()}
    inputs.token_stats=norm['token_stats'];inputs.uncertainty_stats=norm['uncertainty_stats']
    for task in inputs.tasks:predictions.setdefault(task,[]).append(predict(inputs,model,task,'test',args.device))
   else:
    if not predictions:fold=label_free_fold(pool,outer,q['normalization']);assert tuple(fold.tasks)==tuple(mc['task_names'])
    else:fold.stats={t:fold._parse_stats(t,q['normalization'][t]) for t in fold.tasks}
    for task in fold.tasks:predictions.setdefault(task,[]).append(module._predict(fold=fold,model=model,task=task,role='test',device=torch.device(args.device)))
   del model,q,state
  for task,values in predictions.items():
   a=values[0]['anchor_prediction'];ids=values[0]['sample_ids']
   assert all(np.array_equal(a,v['anchor_prediction']) and np.array_equal(ids,v['sample_ids']) for v in values)
   raw=np.mean(np.stack([v['raw_prediction'] for v in values]),axis=0)
   rel=Path(task+'.json') if outer is None else Path(f'outer_{outer}/{task}.json');beta=read(base/'calibration'/rel)['beta']
   dest=output/'predictions'/rel.with_suffix('.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('xb') as f:np.savez_compressed(f,sample_ids=ids,anchor_prediction=a,raw_prediction=raw,prediction=calibrator.apply_beta(a,raw,beta))
   commitments[str(dest.relative_to(output))]=sha(dest)
 save(output/'predictions_committed.json',{'files':commitments,'experiment_id':cfg['experiment_id'],'test_labels_read':False,'new_training':False,'beta_refitted':False})
 print(output)
if __name__=='__main__':main()
