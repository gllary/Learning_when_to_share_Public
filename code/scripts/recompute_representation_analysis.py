"""Recompute all retained representation analyses from complete frozen test folds.

No training; the fixed 20261108 protocol determines every descriptor, PCA fit
and permutation. --check recomputes all 70000 null values and verifies archives.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'data/analysis/representation'
EXPECTED=ROOT/'results/analysis/representation'
PACKAGE=None
TASK_ORDER = ['expt_is_metal','glass','mp_is_metal','dielectric','expt_gap','jdft2d',
              'log_gvrh','log_kvrh','mp_e_form','mp_gap','perovskites','phonons','steels']
GROUPS = {
 'composition_stoichiometry': ['n_elements','composition_entropy','max_atomic_fraction','min_atomic_fraction','metal_fraction','transition_metal_fraction','alkali_fraction','alkaline_fraction','halogen_fraction','chalcogen_fraction','rare_earth_fraction'],
 'electronic_chemistry': [p+'_'+s for p in ['atomic_number','electronegativity','period','group'] for s in ['mean','std','min','max','range']],
 'size_mass': [p+'_'+s for p in ['atomic_mass','atomic_radius'] for s in ['mean','std','min','max','range']],
 'structure_geometry_density': ['n_sites','volume','volume_per_atom','density','lattice_a','lattice_b','lattice_c','lattice_length_anisotropy','lattice_alpha','lattice_beta','lattice_gamma'],
 'symmetry': ['spacegroup_number'],
}
GROUP_LABELS = ['Composition and\nstoichiometry','Electronic and\nchemical properties','Atomic size\nand mass','Structural geometry\nand density','Crystal\nsymmetry','Target\nproperty']
CLASSIFICATION = {'glass','expt_is_metal','mp_is_metal'}

def write_json(path, value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_protocol():
 p=ROOT/'records/analysis/representation/protocol.json'
 return json.loads(p.read_text())

def write_csv(path, frame):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.name.endswith('.gz'):
  with path.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z, io.TextIOWrapper(z,encoding='utf-8',newline='') as f:
   frame.to_csv(f,index=False,float_format='%.17g',lineterminator='\n')
 else:frame.to_csv(path,index=False,float_format='%.17g',lineterminator='\n')

def manifest(input_root,task):
 with np.load(Path(input_root)/'data/matbench/raw/manifests'/f'{task}.npz',allow_pickle=False) as z:
  return {k:z[k] for k in ['sample_ids','raw_indices','targets','structure_hashes','composition_hashes','metadata_json']+[f'outer_{f}_test' for f in range(5)]}

def full_fold(input_root,task,fold):
 cfg=load_protocol();m=manifest(input_root,task);spec=cfg['tasks'][task]
 p=Path(input_root)/'data/matbench/stage1_pool/exports'/spec['expert_id']/task/f'outer_{fold}/test'
 md=json.loads((p/'metadata.json').read_text());ix=m[f'outer_{fold}_test']
 arrays={k:np.load(p/f'{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['sample_ids','hidden','prediction','available']}
 assert np.array_equal(arrays['sample_ids'].astype(str),m['sample_ids'][ix].astype(str))
 assert arrays['hidden'].shape==(len(ix),spec['hidden_dim'])
 assert arrays['available'].all() and np.isfinite(arrays['hidden']).all() and np.isfinite(arrays['prediction']).all()
 return arrays,np.asarray(m['targets'][ix],dtype=np.float64),ix,m,md

def stable_seed(*parts):
 return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'big')

def preprocess(values,*,drop_constant=True):
 x=np.asarray(values,dtype=np.float64)
 observed=np.isfinite(x).any(axis=0);median=np.zeros(x.shape[1]);median[observed]=np.nanmedian(np.where(np.isfinite(x[:,observed]),x[:,observed],np.nan),axis=0)
 filled=np.where(np.isfinite(x),x,median);mean=filled.mean(axis=0);scale=filled.std(axis=0)
 keep=scale>load_protocol()['preprocessing']['constant_threshold']
 safe_scale=np.where(keep,scale,1.0)
 transformed=(filled-mean)/safe_scale
 return (transformed[:,keep] if drop_constant else transformed),{'median':median,'mean':mean,'scale':safe_scale,'keep':keep,'all_missing':~observed}

def cka(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 a=a-a.mean(axis=0);b=b-b.mean(axis=0)
 denominator=np.linalg.norm(a.T@a,'fro')*np.linalg.norm(b.T@b,'fro')
 return float(np.sum((a.T@b)**2)/denominator) if denominator>0 else np.nan

def permutation(a,b,task,fold,group):
 cfg=load_protocol()['permutation'];seed=stable_seed(cfg['base_seed'],task,fold,group)
 a=a-a.mean(axis=0);b=b-b.mean(axis=0)
 denominator=float(np.linalg.norm(a.T@a,'fro')*np.linalg.norm(b.T@b,'fro'))
 assert denominator>0
 observed=float(np.sum((a.T@b)**2)/denominator)
 rng=np.random.default_rng(seed)
 null=np.array([float(np.sum((a.T@b[rng.permutation(len(b))])**2)/denominator) for _ in range(cfg['repeats'])])
 mean=float(null.mean());sd=float(null.std(ddof=cfg['null_sd_ddof']));assert sd>0
 row={'task':task,'outer_fold':fold,'descriptor_group':group,'n_samples':len(a),'linear_cka':observed,
      'permutation_null_mean':mean,'permutation_null_sd':sd,'permutation_z':(observed-mean)/sd,
      'one_sided_permutation_p':float((1+(null>=observed).sum())/(len(null)+1)),
      'n_permutations':len(null),'permutation_seed':seed,'descriptor_dim':b.shape[1],'status':'computed'}
 return row,pd.DataFrame({'task':task,'outer_fold':fold,'descriptor_group':group,'permutation_index':np.arange(1,len(null)+1),'linear_cka':null})

def base_task(input_root,task):
 cfg=load_protocol();rows=[];nulls=[];checks=[];pca_metadata=[];inputs=[]
 for fold in range(5):
  start=time.perf_counter();arrays,y,positions,m,md=full_fold(input_root,task,fold)
  hidden=np.asarray(arrays['hidden'],dtype=np.float64);values,pp=preprocess(hidden)
  components=min(cfg['pca']['max_components_for_cka'],values.shape[1],len(y)-1);assert components>=2
  from sklearn.decomposition import PCA
  model=PCA(n_components=components,svd_solver=cfg['pca']['solver']);reduced=model.fit_transform(values)
  assert np.isfinite(reduced).all()
  out=PACKAGE/'reduced_representations'/task;out.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(out/f'outer_{fold}.npz',sample_ids=arrays['sample_ids'],global_positions=positions,target=y,scores=reduced,
    components=model.components_,explained_variance_ratio=model.explained_variance_ratio_,median=pp['median'],mean=pp['mean'],scale=pp['scale'],keep=pp['keep'])
  b,_=preprocess(y[:,None]);r,null=permutation(reduced,b,task,fold,'target_property');r.update(hidden_dim=hidden.shape[1],reduced_dim_for_cka=components)
  rows.append(r);nulls.append(null)
  checks.append({'task':task,'outer_fold':fold,'samples':len(y),'all_inputs_available_and_finite':True,'sample_id_alignment_verified':True})
  pca_metadata.append({'task':task,'outer_fold':fold,'n_samples':len(y),'hidden_dim':hidden.shape[1],'nonconstant_dimensions':values.shape[1],'reduced_dim':components,'cka_components_explained_variance':float(model.explained_variance_ratio_.sum()),'pc1_explained_variance':float(model.explained_variance_ratio_[0]),'pc2_explained_variance':float(model.explained_variance_ratio_[1])})
  if fold==cfg['pca']['plot_outer_fold']:
   write_csv(PACKAGE/'pca'/f'{task}.csv.gz',pd.DataFrame({'task':task,'outer_fold':fold,'sample_id':arrays['sample_ids'].astype(str),'global_position':positions,'pc1':reduced[:,0],'pc2':reduced[:,1],'target':y}))
  relative=Path('data/matbench/stage1_pool/exports')/cfg['tasks'][task]['expert_id']/task/f'outer_{fold}/test'
  inputs.extend({'path':str(relative/v['filename']),'role':k,'task':task,'outer_fold':fold} for k,v in md['files'].items() if k in ['hidden','sample_ids','available','prediction'])
  print(json.dumps({'phase':'base','task':task,'fold':fold,'n':len(y),'seconds':time.perf_counter()-start}),flush=True)
 write_csv(PACKAGE/'folds'/f'{task}_target.csv',pd.DataFrame(rows))
 write_csv(PACKAGE/'folds'/f'{task}_target_null.csv.gz',pd.concat(nulls,ignore_index=True));write_csv(PACKAGE/'folds'/f'{task}_pca_metadata.csv',pd.DataFrame(pca_metadata))
 write_json(PACKAGE/'verification/base'/f'{task}.json',{'complete':True,'checks':checks})
 write_json(PACKAGE/'input_manifest'/f'{task}_exports.json',inputs)

def descriptor_matrix(frame,group):
 cfg=load_protocol();x=frame[GROUPS[group]].to_numpy(dtype=float)
 if group=='symmetry':
  labels=frame.crystal_system.astype(str).to_numpy();onehot=np.column_stack([labels==c for c in cfg['symmetry']['categories']]).astype(float)
  x=np.column_stack([x,onehot])
 return preprocess(x)[0]

def descriptor_task(task):
 cfg=load_protocol();path=INPUT/'descriptors'/f'{task}.csv.gz';frame=pd.read_csv(path)
 rows=[];nulls=[];check=[]
 for fold in range(5):
  with np.load(PACKAGE/'reduced_representations'/task/f'outer_{fold}.npz',allow_pickle=False) as z:
   ids=z['sample_ids'].astype(str);reduced=z['scores'];target=z['target'];positions=z['global_positions']
  part=frame.set_index('sample_id').loc[ids].copy();assert np.array_equal(part.global_position,positions);assert np.allclose(part.target,target,rtol=1e-13,atol=1e-13)
  for group in GROUPS:
   basic={'task':task,'outer_fold':fold,'descriptor_group':group,'n_samples':len(part),'hidden_dim':cfg['tasks'][task]['hidden_dim'],'reduced_dim_for_cka':reduced.shape[1]}
   if cfg['tasks'][task]['input_type']=='composition' and group in ['structure_geometry_density','symmetry']:
    rows.append({**basic,'status':'not_applicable','descriptor_dim':0});continue
   matrix=descriptor_matrix(part,group)
   if matrix.shape[1]==0:rows.append({**basic,'status':'no_variation','descriptor_dim':0});continue
   row,null=permutation(reduced,matrix,task,fold,group);row.update(hidden_dim=basic['hidden_dim'],reduced_dim_for_cka=reduced.shape[1]);rows.append(row);nulls.append(null)
  check.append({'fold':fold,'all_sample_ids_and_targets_aligned':True,'n':len(part)})
  print(json.dumps({'phase':'descriptor_analysis','task':task,'fold':fold,'n':len(part)}),flush=True)
 write_csv(PACKAGE/'folds'/f'{task}_descriptors.csv',pd.DataFrame(rows));write_csv(PACKAGE/'folds'/f'{task}_descriptor_null.csv.gz',pd.concat(nulls,ignore_index=True))
 write_json(PACKAGE/'verification/descriptor_analysis'/f'{task}.json',{'complete':True,'checks':check})

def aggregate():
 cfg=load_protocol();cka_rows=[];pca_rows=[];all_null=[]
 for task in cfg['tasks']:
  for kind in ['target','descriptors']:cka_rows.append(pd.read_csv(PACKAGE/'folds'/f'{task}_{kind}.csv'))
  pca_rows.append(pd.read_csv(PACKAGE/'folds'/f'{task}_pca_metadata.csv'))
  for kind in ['target','descriptor']:all_null.append(pd.read_csv(PACKAGE/'folds'/f'{task}_{kind}_null.csv.gz'))
 frame=pd.concat(cka_rows,ignore_index=True)
 write_csv(PACKAGE/'representation_descriptor_cka.csv',frame)
 write_csv(PACKAGE/'pca_metadata.csv',pd.concat(pca_rows,ignore_index=True))
 write_csv(PACKAGE/'permutation_null_values.csv.gz',pd.concat(all_null,ignore_index=True))
 summaries=[]
 for (task,group),f in frame.groupby(['task','descriptor_group'],sort=False):
  ok=f[f.status=='computed'];r={'task':task,'descriptor_group':group,'n_total':int(f.n_samples.sum()),'n_folds':len(f),'valid_folds':len(ok),'status':'computed' if len(ok)==5 else ('not_applicable' if len(ok)==0 else 'partial')}
  for c in ['linear_cka','permutation_z','permutation_null_mean','one_sided_permutation_p']:
   r[c+'_mean']=float(ok[c].mean()) if len(ok) else np.nan;r[c+'_sd']=float(ok[c].std(ddof=1)) if len(ok)>1 else np.nan
  summaries.append(r)
 write_csv(PACKAGE/'representation_summary.csv',pd.DataFrame(summaries))

TABLE_TEMPLATES = {17: ('\\begingroup\n\\fontsize{8}{9.5}\\selectfont\n\\setlength{\\tabcolsep}{2pt}\n\\renewcommand{\\arraystretch}{1.13}\n\\setlength{\\LTpre}{8pt}\n\\setlength{\\LTpost}{5pt}\n\\setlength{\\HPavailablewidth}{\\dimexpr\\linewidth-8\\tabcolsep\\relax}\n\\begin{longtable}{@{}>{\\Centering\\arraybackslash}m{0.31386861\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.12408759\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.17518248\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.20437956\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.18248175\\HPavailablewidth}@{}}\n\\caption{\\textbf{Sample counts and representation dimensions}}\\label{supp-tab:17}\\\\\n\\toprule\n\\textbf{Task} & \\textbf{Outer fold} & \\textbf{Analysis records} & \\textbf{Original representation dimensions} & \\textbf{Reduced CKA dimensions} \\\\\n\\midrule\n\\endfirsthead\n\\caption[]{\\textbf{Sample counts and representation dimensions (continued)}}\\\\\n\\toprule\n\\textbf{Task} & \\textbf{Outer fold} & \\textbf{Analysis records} & \\textbf{Original representation dimensions} & \\textbf{Reduced CKA dimensions} \\\\\n\\midrule\n\\endhead\n\\midrule\n\\multicolumn{5}{r}{\\textit{Continued on next page}}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n', '\\end{longtable}\n{\\fontsize{8}{10}\\selectfont\\noindent CKA denotes linear centred kernel alignment. All test samples are used in each outer fold.\\par}\n\\endgroup\n\\par\\medskip\n'), 18: ('\\begingroup\n\\fontsize{8}{9.5}\\selectfont\n\\setlength{\\tabcolsep}{2pt}\n\\renewcommand{\\arraystretch}{1.13}\n\\setlength{\\LTpre}{8pt}\n\\setlength{\\LTpost}{5pt}\n\\setlength{\\HPavailablewidth}{\\dimexpr\\linewidth-14\\tabcolsep\\relax}\n\\begin{longtable}{@{}>{\\Centering\\arraybackslash}m{0.17813765\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.06477733\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.22267206\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.09716599\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.10931174\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.11336032\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.10121457\\HPavailablewidth}>{\\Centering\\arraybackslash}m{0.11336032\\HPavailablewidth}@{}}\n\\caption{\\textbf{Fold-resolved CKA and permutation references}}\\label{supp-tab:18}\\\\\n\\toprule\n\\textbf{Task} & \\textbf{Outer fold} & \\textbf{Descriptor group} & \\textbf{Analysis records} & \\textbf{Linear CKA} & \\textbf{Permutation mean} & \\textbf{Permutation z-score} & \\textbf{One-sided permutation probability} \\\\\n\\midrule\n\\endfirsthead\n\\caption[]{\\textbf{Fold-resolved CKA and permutation references (continued)}}\\\\\n\\toprule\n\\textbf{Task} & \\textbf{Outer fold} & \\textbf{Descriptor group} & \\textbf{Analysis records} & \\textbf{Linear CKA} & \\textbf{Permutation mean} & \\textbf{Permutation z-score} & \\textbf{One-sided permutation probability} \\\\\n\\midrule\n\\endhead\n\\midrule\n\\multicolumn{8}{r}{\\textit{Continued on next page}}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n', '\\end{longtable}\n{\\fontsize{8}{10}\\selectfont\\noindent CKA denotes linear centred kernel alignment. One-sided permutation probabilities are not adjusted for multiple comparisons. A dash indicates that no corresponding value is available.\\par}\n\\endgroup\n\\par\\medskip\n')}

def make_tables(destination):
    meta=pd.read_csv(destination/'pca_metadata.csv')
    c=pd.read_csv(destination/'representation_descriptor_cka.csv')
    order={t:i for i,t in enumerate(TASK_ORDER)}
    meta=meta.assign(order=meta.task.map(order)).sort_values(['order','outer_fold'])
    a=meta[['task','outer_fold','n_samples','hidden_dim','reduced_dim']].copy()
    groups=list(GROUPS)+['target_property']
    c=c[c.status=='computed'].assign(order=c.task.map(order),group_order=c.descriptor_group.map({g:i for i,g in enumerate(groups)})).sort_values(['order','outer_fold','group_order'])
    b=c[['task','outer_fold','descriptor_group','n_samples','linear_cka','permutation_null_mean','permutation_z','one_sided_permutation_p']].copy()
    assert a.shape==(65,5) and b.shape==(350,8)
    labels=dict(zip(groups,[s.replace('\n',' ') for s in GROUP_LABELS]))
    def task_tex(t):return r'\HPtask{'+('matbench_'+t).replace('_',r'\_\allowbreak{}')+'}'
    rows17=[[task_tex(r.task),str(r.outer_fold+1),str(r.n_samples),str(r.hidden_dim),str(r.reduced_dim)] for r in a.itertuples()]
    rows18=[[task_tex(r.task),str(r.outer_fold+1),labels[r.descriptor_group],str(r.n_samples),f'{r.linear_cka:.6f}',f'{r.permutation_null_mean:.6f}',f'{r.permutation_z:.3f}',f'{r.one_sided_permutation_p:.6f}'] for r in b.itertuples()]
    for n,frame,rows in [(17,a,rows17),(18,b,rows18)]:
        write_csv(destination/f'table_{n}.csv',frame)
        prefix,suffix=TABLE_TEMPLATES[n]
        (destination/f'table_{n}.tex').write_text(prefix+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+suffix)

def compare_frame(actual,expected,keys):
    a=pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    b=pd.read_csv(expected).sort_values(keys).reset_index(drop=True)
    assert list(a.columns)==list(b.columns) and a.shape==b.shape,(str(actual),'schema')
    delta=0.0
    for column in a:
        if pd.api.types.is_numeric_dtype(b[column]):
            x=a[column].to_numpy(dtype=float);y=b[column].to_numpy(dtype=float)
            np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=str(actual)+'/'+column)
            finite=np.isfinite(x)&np.isfinite(y)
            if finite.any():delta=max(delta,float(np.abs(x[finite]-y[finite]).max()))
        else:np.testing.assert_array_equal(a[column].fillna('').astype(str),b[column].fillna('').astype(str))
    return {'rows':len(a),'columns':len(a.columns),'max_absolute_difference':delta}

def check_results(output):
    checks={}
    for name,keys in [('representation_descriptor_cka.csv',['task','outer_fold','descriptor_group']),('representation_summary.csv',['task','descriptor_group']),('pca_metadata.csv',['task','outer_fold']),('pca_coordinates.csv',['task','sample_id']),('table_17.csv',['task','outer_fold']),('table_18.csv',['task','outer_fold','descriptor_group'])]:
        checks[name]=compare_frame(output/name,EXPECTED/name,keys)
    checks['permutation_null_values']=compare_frame(output/'permutation_null_values.csv.gz',ROOT/'data/analysis/representation/permutation_null_values.csv.gz',['task','outer_fold','descriptor_group','permutation_index'])
    # Both table schemas and values are checked against their released CSV files above.
    for task in TASK_ORDER:
        compare_frame(output/'pca'/f'{task}.csv.gz',ROOT/'data/analysis/representation/pca'/f'{task}.csv.gz',['sample_id'])
        for fold in range(5):
            with np.load(output/'reduced_representations'/task/f'outer_{fold}.npz',allow_pickle=False) as a,np.load(ROOT/'data/analysis/representation/reduced_representations'/task/f'outer_{fold}.npz',allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for field in a.files:
                    if a[field].dtype.kind in 'biufc':np.testing.assert_allclose(a[field],b[field],rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=task+'/'+field)
                    else:np.testing.assert_array_equal(a[field],b[field])
    return checks

def run(output,check):
    global PACKAGE
    PACKAGE=output
    metadata=json.loads((INPUT/'metadata.json').read_text())
    assert metadata.get('task_records')==408062 and metadata.get('outer_test_folds')==65, 'Full input metadata required; sampled inputs are unsupported'
    for task in load_protocol()['tasks']:base_task(ROOT,task)
    for task in load_protocol()['tasks']:descriptor_task(task)
    aggregate()
    coords=pd.concat([pd.read_csv(output/'pca'/f'{t}.csv.gz') for t in TASK_ORDER],ignore_index=True)
    write_csv(output/'pca_coordinates.csv',coords)
    make_tables(output)
    c=pd.read_csv(output/'representation_descriptor_cka.csv');m=pd.read_csv(output/'pca_metadata.csv');null=pd.read_csv(output/'permutation_null_values.csv.gz')
    assert len(m)==65 and m.n_samples.sum()==408062 and len(coords)==81618
    assert len(c)==390 and (c.status=='computed').sum()==350 and (c.status=='not_applicable').sum()==40
    assert len(null)==70000 and (c.loc[c.status=='computed','n_permutations']==200).all()
    checks=check_results(output) if check else {}
    record={'passed':True,'checked_archived_results':check,'task_records':408062,'tasks':13,'outer_test_folds':65,'cka_rows':390,'computed_rows':350,'not_applicable_rows':40,'permutations_per_group':200,'permutation_values':70000,'pca_records':81618,'table_17_shape':[65,5],'table_18_shape':[350,8],'models_retrained':False,'checks':checks}
    write_json(output/'execution.json',record)
    print(json.dumps(record,indent=2))

def main():
    global INPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--input',type=Path,default=INPUT,help='Full descriptors and metadata; frozen exports are reused from this release')
    p.add_argument('--output',type=Path,help='New directory under runs/')
    args=p.parse_args();INPUT=args.input.resolve()
    if not args.check and args.output is None:p.error('Choose --check or --output')
    if args.output:
        output=args.output.resolve()
        if not output.is_relative_to(ROOT/'runs'):p.error('Output must be under runs/')
        output.mkdir(parents=True,exist_ok=False);run(output,args.check)
    else:
        with tempfile.TemporaryDirectory(prefix='hpsafe-full-') as tmp:run(Path(tmp),True)
if __name__=='__main__':main()
