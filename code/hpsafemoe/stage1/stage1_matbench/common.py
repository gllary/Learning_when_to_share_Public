import hashlib
import json
import os
import sys
from pathlib import Path
RELEASE = Path(__file__).resolve().parents[4]
if 'HPSAFEMOE_STAGE1_RUN_DIR' not in os.environ:
    raise RuntimeError('Configure HPSAFEMOE_STAGE1_RUN_DIR as a new directory under runs/')
ROOT = Path(os.environ['HPSAFEMOE_STAGE1_RUN_DIR']).resolve()
if not ROOT.is_relative_to(RELEASE / 'runs'):
    raise ValueError('Stage1 writes must stay under runs/')
PROJECT = RELEASE / 'results/matbench'
sys.path.insert(0, str(RELEASE / 'code'))

def read(p):
    return json.loads(Path(p).read_text())

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def update(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(q, p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def array_sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class RestrictedTargets:
    """Only explicitly allowed label positions can be indexed by the estimator."""

    def __init__(self, values, positions):
        import numpy as np
        self.values = values
        self.allowed = np.zeros(len(values), dtype=bool)
        self.allowed[positions] = True

    def __getitem__(self, key):
        import numpy as np
        if not np.all(self.allowed[key]):
            raise PermissionError('Held-out label access forbidden')
        return self.values[key]
