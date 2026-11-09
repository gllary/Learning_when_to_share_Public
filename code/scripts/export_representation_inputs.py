"""Validate complete frozen inputs, optionally regenerate all raw descriptors."""
from __future__ import annotations
import argparse, csv, gzip, io, json, math, shutil, tempfile, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import recompute_representation_analysis as analysis
from recompute_representation_analysis import ROOT, INPUT, GROUPS, full_fold, load_protocol, manifest, write_json
PACKAGE=None
PROPERTIES=None
CATEGORIES=None

def settings():
 global PROPERTIES,CATEGORIES
 if PROPERTIES is None:
  cfg=load_protocol();CATEGORIES={k:set(v) for k,v in cfg['element_categories'].items()}
  PROPERTIES={r['symbol']:r for r in json.loads((ROOT/'data/analysis/representation/element_properties.json').read_text())}
 return load_protocol()

def weighted(values,weights,name):
 a=np.array([np.nan if v is None else float(v) for v in values]);valid=np.isfinite(a)
 if not valid.any():return {name+'_'+s:np.nan for s in ['mean','std','min','max','range']}
 a=a[valid];w=weights[valid];w=w/w.sum();mean=float(w@a)
 return {name+'_mean':mean,name+'_std':float(np.sqrt(w@((a-mean)**2))),name+'_min':float(a.min()),name+'_max':float(a.max()),name+'_range':float(np.ptp(a))}

def describe(raw,cfg):
 s=Structure.from_dict(raw) if isinstance(raw,dict) else None
 c=s.composition if s is not None else Composition(raw)
 amounts=c.get_el_amt_dict();symbols=sorted(amounts);w=np.array([amounts[x] for x in symbols],dtype=float);w/=w.sum()
 result={'input_type':'structure' if s is not None else 'composition','formula':c.reduced_formula,
  'n_elements':len(symbols),'composition_entropy':float(-(w*np.log(np.clip(w,1e-15,None))).sum()),
  'max_atomic_fraction':float(w.max()),'min_atomic_fraction':float(w.min())}
 for name,group in CATEGORIES.items():result[name+'_fraction']=float(sum(x for e,x in zip(symbols,w) if e in group))
 for prop in ['atomic_number','atomic_mass','electronegativity','atomic_radius','period','group']:
  result.update(weighted([PROPERTIES[e][prop] for e in symbols],w,prop))
 for name in GROUPS['structure_geometry_density']+['spacegroup_number']:result[name]=np.nan
 result['crystal_system']='not_applicable';result['symmetry_status']='not_applicable'
 if s is not None:
  a,b,c=s.lattice.abc;alpha,beta,gamma=s.lattice.angles
  result.update(n_sites=len(s),volume=float(s.volume),volume_per_atom=float(s.volume/len(s)),density=float(s.density),
   lattice_a=float(a),lattice_b=float(b),lattice_c=float(c),lattice_length_anisotropy=float(max(a,b,c)/min(a,b,c)),
   lattice_alpha=float(alpha),lattice_beta=float(beta),lattice_gamma=float(gamma))
  try:
   analyzer=SpacegroupAnalyzer(s,symprec=cfg['symmetry']['symprec_angstrom'],angle_tolerance=cfg['symmetry']['angle_tolerance_degrees'])
   result.update(spacegroup_number=int(analyzer.get_space_group_number()),crystal_system=analyzer.get_crystal_system(),symmetry_status='ok')
  except Exception as exc:
   result['crystal_system']='unresolved';result['symmetry_status']=type(exc).__name__+': '+str(exc)
 return result

def prepare_task(input_root,task):
 warnings.filterwarnings('ignore',category=UserWarning);warnings.filterwarnings('ignore',category=DeprecationWarning)
 cfg=settings();start=time.perf_counter();m=manifest(input_root,task)
 definition=json.loads(str(m['metadata_json'].flat[0]))['definition'];n=len(m['sample_ids'])
 raw=Path(input_root)/'data/matbench/raw_inputs'/definition['raw_path']
 assert np.array_equal(m['raw_indices'],np.arange(n))
 fold_of=np.full(n,-1,dtype=int)
 for fold in range(5):
  ix=m[f'outer_{fold}_test'];assert np.all(fold_of[ix]<0);fold_of[ix]=fold
 assert np.all(fold_of>=0)
 destination=PACKAGE/'descriptors'/f'{task}.csv.gz'
 done=PACKAGE/'verification/descriptors'/f'{task}.json'
 destination.parent.mkdir(parents=True,exist_ok=True)
 temporary=destination.with_suffix('.partial')
 count=0;writer=None;errors=[];missing={};identity=[]
 with temporary.open('wb') as output, gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0) as compressed, io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream, gzip.open(raw,'rb') as handle:
  for pos,(material,target) in enumerate(ijson.items(handle,'data.item',use_float=True)):
   assert np.isclose(float(target),m['targets'][pos],rtol=1e-10,atol=1e-10)
   row={'task':task,'sample_id':str(m['sample_ids'][pos]),'global_position':pos,'raw_index':int(m['raw_indices'][pos]),'outer_fold':int(fold_of[pos]),'target':float(target)}
   row.update(describe(material,cfg))
   if row['symmetry_status'] not in ['ok','not_applicable']:errors.append({'sample_id':row['sample_id'],'error':row['symmetry_status']})
   if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
   writer.writerow(row);count+=1
   for key,value in row.items():
    if isinstance(value,float) and not np.isfinite(value):missing[key]=missing.get(key,0)+1
   if count%10000==0:print(json.dumps({'phase':'descriptors','task':task,'complete':count,'total':n}),flush=True)
 assert count==n;temporary.replace(destination)
 record={'task':task,'n':n,'symmetry_failures':errors,'missing_values':missing,'seconds':time.perf_counter()-start,
  'raw_labels_and_official_fold_alignment_verified':True,'all_records_retained':True}
 write_json(done,record);print(json.dumps({'phase':'descriptors_complete','task':task,'n':n,'seconds':record['seconds']}),flush=True)
 return record


def validate_inputs(directory):
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['task_records']==408062 and metadata['outer_test_folds']==65
    for entry in metadata['input_files']:
        assert (ROOT/entry['path']).is_file(),entry['path']
    counts=[]
    for task,spec in load_protocol()['tasks'].items():
        m=manifest(ROOT,task);d=pd.read_csv(directory/'descriptors'/f'{task}.csv.gz')
        assert len(d)==len(m['sample_ids'])==spec['n'] and d.sample_id.is_unique
        np.testing.assert_array_equal(d.sample_id.astype(str),m['sample_ids'].astype(str))
        np.testing.assert_array_equal(d.global_position,np.arange(len(d)))
        np.testing.assert_allclose(d.target,m['targets'],rtol=1e-13,atol=1e-13)
        seen=[]
        for fold in range(5):
            arrays,y,ix,_,_=full_fold(ROOT,task,fold)
            assert len(ix)==spec['fold_sizes'][fold]
            np.testing.assert_array_equal(d.iloc[ix].outer_fold,fold)
            seen.extend(ix.tolist())
        np.testing.assert_array_equal(np.sort(seen),np.arange(len(d)))
        counts.append({'task':task,'records':len(d),'fold_sizes':spec['fold_sizes']})
    return {'passed':True,'task_records':sum(r['records'] for r in counts),'tasks':counts,'input_files_present':len(metadata['input_files']),'raw_inputs_and_frozen_exports_reused':True}

def execute(output,descriptors,check):
    global PACKAGE,ijson,Composition,Structure,SpacegroupAnalyzer
    if descriptors:
        import ijson
        from pymatgen.core import Composition,Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        PACKAGE=output
        for task in load_protocol()['tasks']:prepare_task(ROOT,task)
    elif output:
        shutil.copytree(INPUT/'descriptors',output/'descriptors')
    if output:shutil.copy2(INPUT/'metadata.json',output/'metadata.json')
    record=validate_inputs(output or INPUT)
    if descriptors and check:
        for task in load_protocol()['tasks']:
            analysis.compare_frame(output/'descriptors'/f'{task}.csv.gz',INPUT/'descriptors'/f'{task}.csv.gz',['sample_id'])
    record['descriptors_recomputed']=descriptors
    record['checked_archived_inputs']=check
    if output:write_json(output/'execution.json',record)
    print(json.dumps(record,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--descriptors',action='store_true',help='Regenerate every descriptor from all raw records using frozen element properties')
    args=p.parse_args()
    if not args.check and args.output is None:p.error('Choose --check or --output')
    if args.output:
        out=args.output.resolve()
        if not out.is_relative_to(ROOT/'runs'):p.error('Output must be under runs/')
        out.mkdir(parents=True,exist_ok=False);execute(out,args.descriptors,args.check)
    elif args.descriptors:
        with tempfile.TemporaryDirectory(prefix='hpsafe-descriptors-') as tmp:execute(Path(tmp),True,args.check)
    else:execute(None,False,args.check)
if __name__=='__main__':main()
