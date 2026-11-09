"""Portable I/O for the frozen JARVIS input contract; writes only to a new run."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
RELEASE = Path(__file__).resolve().parents[4]
ROOT = Path(os.environ.get('HPSAFEMOE_RUN_DIR', str(RELEASE / 'runs/unconfigured'))).resolve()
SOURCE_ROOT = RELEASE / 'data/jarvis/stage1_pool'
PUBLIC_ROOT = RELEASE / 'data/jarvis/public_alignn'
PUBLIC_TASKS = ('reg_avg_elec_mass', 'reg_avg_hole_mass', 'reg_n_powerfact', 'reg_max_efg')
CONTEXT = {}

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path).resolve()
    if 'HPSAFEMOE_RUN_DIR' not in os.environ or not path.is_relative_to(RELEASE / 'runs'):
        raise ValueError('Set HPSAFEMOE_RUN_DIR to a new directory under runs/')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
    temp.replace(path)

def fold(jid):
    return int.from_bytes(hashlib.blake2b(('jarvis34-oof-v1:' + jid).encode(), digest_size=8).digest(), 'little') % 5

def registry():
    return json.loads((RELEASE / 'data/jarvis/registry.json').read_text())

def manifest(task):
    return json.loads((SOURCE_ROOT / 'data/tasks' / task / 'manifest.json').read_text())

def output_dir(task, heldout):
    return SOURCE_ROOT / 'artifacts' / task / ('full' if heldout < 0 else f'fold{heldout}')

def validate_expert_protocol(task, config):
    identity = json.loads((RELEASE / 'checkpoint/jarvis/stage1/protocol_identity.json').read_text())
    assert config.get('protocol_sha256') == identity['source_protocol_canonical_sha256']
    assert config.get('retrained_from_scratch') is True
    roles = manifest(task)['role_ids']
    inner = config['fold']
    expected = set(roles['train']) if inner < 0 else {i for i in roles['train'] if fold(i) != inner}
    assert set(config['trained_ids']) == expected
    assert config['test_labels_read'] is False
    return 'frozen-official-role-expert'

def require_zero_u(u, a):
    if Path(u).exists() or Path(a).exists():
        raise ValueError('Unexpected uncertainty/availability overlay in zero-U experiment')
