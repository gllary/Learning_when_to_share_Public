"""Bind one complete, pinned scalar-ALIGNN implementation (never mix releases)."""
import sys
from common import ROOT

COMMIT = 'dcd73d187ddb331e73c0277f1815fa3d8268e098'

def bind():
    source=ROOT/'vendor_ext/alignn_legacy'
    if not (source/'alignn/train.py').is_file():
        raise RuntimeError('Pinned scalar ALIGNN source has not been installed')
    sys.path.insert(0,str(ROOT/'runtime/alignn_deps'))
    sys.path.insert(0,str(source))
    import alignn
    assert str(source.resolve()) in str(__import__('pathlib').Path(alignn.__file__).resolve())

def configuration(task):
    """Capture the actual official train_prop_model defaults, not a retyping."""
    task={'n_powerfact':'n-powerfact','p_powerfact':'p-powerfact','n_Seebeck':'n-Seebeck','p_Seebeck':'p-Seebeck'}.get(task,task)
    bind()
    import alignn.train_props as entry
    from alignn.config import TrainingConfig
    saved=entry.train_dgl;captured={}
    class Captured(Exception):pass
    def capture(config):
        captured.update(config)
        raise Captured()
    try:
        entry.train_dgl=capture
        try:entry.train_prop_model(prop=task, dataset='dft_3d')
        except Captured:pass
    finally:entry.train_dgl=saved
    assert captured and captured['model']['name']=='alignn'
    return TrainingConfig(**captured)
