"""Reproduce all four Supplementary Table 15 paired-error intervals from saved per-cohort RNG states."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'records/analysis/external_intervals/protocol.json'
NAMES={'core23':'Core-23','ds2_248':'DS2-248','hea95':'HEA-95','b2_18':'B2-18'}

def compute():
    protocol=json.loads(PROTOCOL.read_text());B=protocol['replicates']
    initial_states=json.loads((PROTOCOL.parent/'rng_states.json').read_text())
    rows=[];draws={};states={};assignments=[]
    for slug in protocol['dataset_order']:
        frame=pd.read_csv(ROOT/f'results/external/{slug}/results/per_sample_frozen.csv');y=frame.target.to_numpy();a=frame.stage1.to_numpy();p=frame.stage2.to_numpy();diff=abs(p-y)-abs(a-y);n=len(y)
        assert frame.sample_id.nunique()==n and np.isfinite(np.column_stack([y,a,p])).all()
        rng=np.random.default_rng();rng.bit_generator.state=initial_states[slug]
        states[slug]=rng.bit_generator.state
        if slug=='hea95':
            pairs=json.loads((ROOT/'records/analysis/external_intervals/hea95_shared_label_groups.json').read_text());lookup={v:i for i,v in enumerate(frame.sample_id)};used={v for pair in pairs for v in pair};assert len(used)==sum(map(len,pairs))
            groups=[[lookup[v] for v in pair] for pair in pairs]+[[i] for i,v in enumerate(frame.sample_id) if v not in used];assert len(groups)==90 and sorted(i for g in groups for i in g)==list(range(n))
            sums=np.array([diff[g].sum() for g in groups]);sizes=np.array([len(g) for g in groups])
            ix=rng.integers(0,len(groups),(B,len(groups)));boot=sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)
            assignments=[{'group':gi,'sample_id':str(frame.sample_id.iloc[i]),'group_size':len(g),'original_row':i} for gi,g in enumerate(groups) for i in g]
        else:
            idx=rng.integers(0,n,(B,n));boot=diff[idx].mean(axis=1)
        low,high=np.quantile(boot,protocol['quantiles'],method=protocol['quantile_method']);draws[slug]=boot
        rows.append({'dataset':NAMES[slug],'n':n,'paired_mae_difference':float(diff.mean()),'lower':float(low),'upper':float(high),'better':int((diff<0).sum()),'same':int((diff==0).sum()),'worse':int((diff>0).sum()),'stage1_mean_signed_error':float((a-y).mean()),'full_mean_signed_error':float((p-y).mean())})
    return pd.DataFrame(rows),draws,states,assignments

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    if not args.check and args.output is None:p.error('Choose --check or --output')
    out=None
    if args.output:
        out=args.output.resolve()
        if not out.is_relative_to(ROOT/'runs'):p.error('Output must be under runs/')
        out.mkdir(parents=True,exist_ok=False)
    frame,draws,states,groups=compute();maxdelta=None
    if args.check:
        expected=pd.read_csv(ROOT/'results/analysis/external_intervals/s15_full_precision.csv').set_index('dataset');actual=frame.set_index('dataset');fields=[c for c in actual.columns if c in expected.columns]
        np.testing.assert_allclose(actual[fields],expected.loc[actual.index,fields],rtol=0,atol=1e-10)
        maxdelta=float(np.max(np.abs(actual[fields].to_numpy()-expected.loc[actual.index,fields].to_numpy())))
    if out:
        frame.to_csv(out/'s15_full_precision.csv',index=False,float_format='%.17g');np.savez_compressed(out/'bootstrap_distributions.npz',**draws)
        (out/'rng_states.json').write_text(json.dumps(states,indent=2)+'\n');pd.DataFrame(groups).to_csv(out/'hea95_group_assignments.csv',index=False)
    print(json.dumps({'passed':True,'checked_archived_results':args.check,'datasets':4,'replicates':20000,'seed':20261031,'hea95_groups':len({x['group'] for x in groups}),'max_absolute_difference':maxdelta},indent=2))
if __name__=='__main__':main()
