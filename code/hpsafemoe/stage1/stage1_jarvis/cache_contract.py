"""Bind feature caches to their geometry, source code and data protocol."""
import hashlib
import json
from pathlib import Path
from common import ROOT,sha256,protocol_hash

SUFFIXES={'.py','.json','.yaml','.yml','.csv','.txt','.pyx','.pxd','.h','.c','.cpp'}
def contract(kind):
    files=[ROOT/'official_preprocessing.py']
    mapping={
      'cogn':['prepare_graphs.py','src/hpsafe_cogn_stage1'],
      'congn':['prepare_graphs.py','src/hpsafe_cogn_stage1'],
      'alignn':['alignn_store.py','vendor_ext/alignn_legacy/alignn/graphs.py'],
      'descriptors':['prepare_descriptors.py','tree_reference.py'],
      'atomnet':['strong_backend.py','strong_reference.py','vendor_ext/atomnet/AtomNet/dataset','vendor_ext/atomnet/AtomNet/atomic_descriptor.py'],
      'prism':['strong_backend.py','strong_reference.py','vendor_ext/prism/dataset'],
      'gsmnet':['strong_backend.py','strong_reference.py','vendor_ext/gsmnet/data.py','vendor_ext/gsmnet/grit/transform','vendor_ext/gsmnet/functions']}
    for name in mapping[kind]:
        p=ROOT/name
        if p.is_dir():files.extend(q for q in p.rglob('*') if q.is_file() and q.suffix in SUFFIXES and '__pycache__' not in q.parts and 'build' not in q.parts)
        else:files.append(p)
    actual_code_root=Path(__file__).resolve().parent
    hashes={}
    for p in files:
        relative=p.relative_to(ROOT)
        source=p if p.is_file() else actual_code_root/relative
        assert source.is_file(),source
        hashes[str(relative)]=sha256(source)
    hashes['cache_schema']='target-free-v1'
    payload=dict(kind=kind,protocol_sha256=protocol_hash(),universe_sha256=sha256(ROOT/'data/universe_ids.npy'),structures_sha256=sha256(ROOT/'data/structures.json'),sources=hashes)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def verify(receipt,kind,files=()):
    assert receipt.get('cache_contract') == contract(kind), 'Invalid '+kind+' cache'
    for path,expected in files:
        path=Path(path)
        assert path.is_file() and not path.is_symlink(),path
        assert sha256(path)==expected, 'Corrupt cache: '+str(path)
    return receipt
