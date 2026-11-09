from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
RELEASE = Path(__file__).resolve().parents[4]
if 'HPSAFEMOE_STAGE1_RUN_DIR' not in os.environ:
    raise RuntimeError('Configure HPSAFEMOE_STAGE1_RUN_DIR as a new directory under runs/')
ROOT = Path(os.environ['HPSAFEMOE_STAGE1_RUN_DIR']).resolve()
if not ROOT.is_relative_to(RELEASE / 'runs'):
    raise ValueError('Stage1 output must be a new directory under runs/')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)

def sha256(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            d.update(b)
    return d.hexdigest()

def protocol_hash():
    return json.loads((RELEASE / 'checkpoint/jarvis/stage1/protocol_identity.json').read_text())['source_protocol_canonical_sha256']

def fold(jid):
    return int.from_bytes(hashlib.blake2b(('jarvis34-oof-v1:' + jid).encode(), digest_size=8).digest(), 'little') % 5

def registry():
    return json.loads((ROOT / 'registry.json').read_text())

def manifest(task):
    return json.loads((ROOT / 'data' / 'tasks' / task / 'manifest.json').read_text())

def training_labels(task, heldout):
    assert heldout in [-1, 0, 1, 2, 3, 4]
    labels = json.loads((ROOT / 'data' / 'tasks' / task / 'train.json').read_text())
    m = manifest(task)['role_ids']
    assert set(labels) == set(m['train'])
    return labels if heldout == -1 else {j: y for j, y in labels.items() if fold(j) != heldout}

def validation_labels(task):
    labels = json.loads((ROOT / 'data' / 'tasks' / task / 'val.json').read_text())
    m = manifest(task)['role_ids']
    assert set(labels) == set(m['val'])
    return labels

def record_event(event, **kwargs):
    with (ROOT / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'event': event, **kwargs}) + '\n')

def output_dir(task, heldout):
    assert heldout in [-1, 0, 1, 2, 3, 4]
    return ROOT / 'artifacts' / task / ('full' if heldout < 0 else f'fold{heldout}')

def validate_expert_protocol(task, config):
    """Stage1 artifacts must match the raw-official-split protocol exactly."""
    assert config.get('protocol_sha256') == protocol_hash()
    assert config.get('retrained_from_scratch') is True
    assert set(config['trained_ids']) == set(training_labels(task, config['fold']))
    return 'v6_1'
