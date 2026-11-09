"""One target-free 273-column cache, streamed in bounded batches."""
import argparse
import json
import os
import io
import numpy as np
import pandas as pd
from common import protocol_hash
from common import ROOT,atomic_json,sha256
from cache_contract import contract,verify

def featurizer(workers):
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty,Stoichiometry,ValenceOrbital,IonProperty
    from matminer.featurizers.structure import SiteStatsFingerprint,StructuralHeterogeneity,ChemicalOrdering,StructureComposition,MaximumPackingEfficiency
    f=MultipleFeaturizer([
      SiteStatsFingerprint.from_preset('CoordinationNumber_ward-prb-2017'),
      SiteStatsFingerprint.from_preset('LocalPropertyDifference_ward-prb-2017'),
      StructuralHeterogeneity(),MaximumPackingEfficiency(),ChemicalOrdering(),
      StructureComposition(Stoichiometry()),StructureComposition(ElementProperty.from_preset('magpie')),
      StructureComposition(ValenceOrbital(props=['frac'])),StructureComposition(IonProperty(fast=True))])
    f.set_n_jobs(workers);f.set_chunksize(16)
    assert len(f.feature_labels())==273,len(f.feature_labels())
    return f

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    final=ROOT/'data'/'descriptors.npy';receipt=ROOT/'data'/'descriptors.receipt.json'
    if final.exists() and receipt.exists():
        prior=json.loads(receipt.read_text());verify(prior,'descriptors',[(final,prior['sha256'])]);return
    if (ROOT/'data/descriptors.building.npy').exists():raise RuntimeError('An incomplete descriptor cache exists; inspect before restarting')
    from official_preprocessing import tree_structure
    ids=np.load(ROOT/'data'/'universe_ids.npy').astype(str)
    structures=json.loads((ROOT/'data'/'structures.json').read_text());f=featurizer(a.workers)
    cache=np.lib.format.open_memmap(ROOT/'data'/'descriptors.building.npy',mode='w+',dtype='float64',shape=(len(ids),273))
    for start in range(0,len(ids),256):
        stop=min(start+256,len(ids));ss=[]
        for j in ids[start:stop]:
            ss.append(tree_structure(structures[j]))
        values=f.featurize_many(ss,ignore_errors=True,pbar=False)
        # The published contribution writes CSV, then uses pandas.read_csv.
        frame=pd.DataFrame(np.asarray(values,dtype='float64'),columns=f.feature_labels())
        parsed=pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        cache[start:stop]=parsed.to_numpy(dtype='float64');cache.flush()
        atomic_json(ROOT/'status'/'descriptors.json',{'completed':stop,'total':len(ids),'status':'building'})
        print('DESCRIPTORS_PROGRESS',stop,len(ids),flush=True)
    os.replace(ROOT/'data'/'descriptors.building.npy',final)
    atomic_json(receipt,{'cache_contract':contract('descriptors'),'features':f.feature_labels(),'rows':len(ids),'columns':273,'missing_values':int(np.isnan(cache).sum()),'sha256':sha256(final),'labels_read':False,'dtype':'float64','preprocessing':'Atoms.pymatgen_converter_unitcell_pandas_CSV_roundtrip','source_sha256':sha256(ROOT/'official_preprocessing.py'),'protocol_sha256':protocol_hash()})
    atomic_json(ROOT/'status'/'descriptors.json',{'completed':len(ids),'total':len(ids),'status':'complete'})

if __name__=='__main__':main()
