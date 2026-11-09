from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
import numpy as np
import pandas as pd
from .progress import ProgressHeartbeat
MATERIAL_COLUMNS = ('sample_id', 'composition', 'structure_path', 'structure_json')
TARGET_ALIASES = frozenset({'target', 'targets', 'label', 'labels', 'y', 'y_true', 'property', 'value'})
MATTERVIAL_COMMIT = '0d17f9d24a7e0d982ab2845caf3205d520cccb08'
MODNET_COMMIT = 'fca6bda0146e0fd3696b4df5fd14a0b71fcbe8cd'
PRODUCTION_SCHEMA_VERSION = 1
MODULE_NAME = 'hpsafe_features'
MODULE_API_VERSION = '1.1'
CACHE_CONTRACT = 'hpsafe.material-feature-cache/v1'

def module_contract() -> dict[str, str]:
    """Return the stable producer identity embedded in every new artifact."""
    return {'producer_module': MODULE_NAME, 'producer_api_version': MODULE_API_VERSION, 'cache_contract': CACHE_CONTRACT}

def _require_current_contract(value: Mapping[str, Any], *, artifact: str) -> None:
    expected = module_contract()
    observed = {key: value.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f'{artifact} contract mismatch: expected {expected}, found {observed}. Use a new standalone data-processing output root.')
BLOCK_SPECS: dict[str, dict[str, Any]] = {'matminer': {'modality': 'composition_or_structure', 'runtime': 'primary', 'gpu': False, 'availability': 'any_finite'}, 'mvl16': {'modality': 'structure', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'mvl32': {'modality': 'structure', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'megnet_mm': {'modality': 'structure', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'megnet_ofm': {'modality': 'structure', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'roost_mpgap': {'modality': 'composition', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'roost_oqmd_eform': {'modality': 'composition', 'runtime': 'primary', 'gpu': True, 'availability': 'all_finite'}, 'orb_v3': {'modality': 'structure', 'runtime': 'orb', 'gpu': True, 'availability': 'all_finite'}}
UNIT_BLOCKS: dict[str, tuple[str, ...]] = {'matminer': ('matminer',), 'mvl_all': ('mvl16', 'mvl32'), 'mvl16': ('mvl16',), 'mvl32': ('mvl32',), 'megnet_mm': ('megnet_mm',), 'megnet_ofm': ('megnet_ofm',), 'roost_mpgap': ('roost_mpgap',), 'roost_oqmd_eform': ('roost_oqmd_eform',), 'orb_v3': ('orb_v3',)}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as handle:
            temporary = handle.name
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value

def _canonical_composition(value: str) -> str:
    from pymatgen.core import Composition
    composition = Composition(value)
    if not composition.valid or composition.num_atoms <= 0:
        raise ValueError(f'Invalid material composition: {value!r}')
    return composition.reduced_composition.alphabetical_formula

def _material_identity_structure(structure: Any) -> Any:
    """Drop arbitrary object/site properties while preserving physical identity."""
    from pymatgen.core import Structure
    return Structure(lattice=structure.lattice, species=list(structure.species), coords=np.asarray(structure.frac_coords, dtype=float), charge=structure.charge, coords_are_cartesian=False, to_unit_cell=False, validate_proximity=False)

def _structure_from_record(value: Mapping[str, Any]) -> Any:
    from pymatgen.core import Structure
    kind = str(value['kind'])
    if kind == 'inline_pymatgen_json':
        payload = value.get('payload')
        if not isinstance(payload, dict):
            raise ValueError('Inline structure payload is not a JSON object')
        structure = Structure.from_dict(payload)
    elif kind == 'file':
        path = Path(str(value['path']))
        if not path.is_file():
            raise FileNotFoundError(f'Structure file disappeared: {path}')
        if sha256_file(path) != value.get('file_sha256'):
            raise ValueError(f'Structure file changed after input lock: {path}')
        structure = Structure.from_file(path)
    else:
        raise ValueError(f'Unknown normalized structure kind: {kind!r}')
    structure = _material_identity_structure(structure)
    semantic_digest = json_sha256(structure.as_dict())
    if semantic_digest != value.get('semantic_sha256'):
        raise ValueError('Normalized structure semantic hash changed')
    return structure

def _normalize_structure(*, structure_path: str, structure_json: str, structure_root: Path) -> tuple[dict[str, Any] | None, Any | None]:
    if structure_path and structure_json:
        raise ValueError('Each row may use structure_path or structure_json, not both')
    if not structure_path and (not structure_json):
        return (None, None)
    from pymatgen.core import Structure
    if structure_path:
        raw = Path(structure_path).expanduser()
        path = (structure_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            path.relative_to(structure_root)
        except ValueError as exc:
            raise ValueError(f'Structure path escapes --structure-root: {path}') from exc
        if not path.is_file():
            raise FileNotFoundError(f'Structure file does not exist: {path}')
        structure = _material_identity_structure(Structure.from_file(path))
        record = {'kind': 'file', 'path': str(path), 'file_sha256': sha256_file(path), 'semantic_sha256': json_sha256(structure.as_dict())}
        return (record, structure)
    if structure_json:
        payload = json.loads(structure_json)
        if not isinstance(payload, dict):
            raise ValueError('structure_json must encode one pymatgen Structure object')
        structure = _material_identity_structure(Structure.from_dict(payload))
        canonical = structure.as_dict()
        return ({'kind': 'inline_pymatgen_json', 'payload': canonical, 'semantic_sha256': json_sha256(canonical)}, structure)
    raise AssertionError('Unreachable structure normalization branch')

def inspect_material_csv(source_csv: Path) -> dict[str, Any]:
    """Inspect IDs and modalities without materializing non-material columns."""
    source_csv = source_csv.expanduser().resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    header = pd.read_csv(source_csv, nrows=0).columns.astype(str).tolist()
    if 'sample_id' not in header:
        raise ValueError('Input CSV requires a sample_id column')
    material_columns = [name for name in MATERIAL_COLUMNS if name in header]
    if not any((name in material_columns for name in ('composition', 'structure_path', 'structure_json'))):
        raise ValueError('Input CSV requires composition, structure_path or structure_json')
    sample_ids: list[str] = []
    seen: set[str] = set()
    duplicate_examples: list[str] = []
    all_have_composition = True
    all_have_structure = True
    material_projection_digest = hashlib.sha256()
    material_projection_digest.update(json.dumps(material_columns, separators=(',', ':')).encode('utf-8') + b'\n')
    for frame in pd.read_csv(source_csv, usecols=material_columns, dtype=str, keep_default_na=False, na_filter=False, chunksize=10000):
        ids = frame['sample_id'].astype(str).str.strip().tolist()
        if any((not value for value in ids)):
            raise ValueError('sample_id values must be non-empty')
        for value in ids:
            if value in seen and len(duplicate_examples) < 5:
                duplicate_examples.append(value)
            seen.add(value)
        sample_ids.extend(ids)
        for values in frame.loc[:, material_columns].itertuples(index=False, name=None):
            material_projection_digest.update(json.dumps([str(value) for value in values], ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
        has_composition = frame['composition'].astype(str).str.strip().ne('') if 'composition' in frame else pd.Series(False, index=frame.index)
        path_present = frame['structure_path'].astype(str).str.strip().ne('') if 'structure_path' in frame else pd.Series(False, index=frame.index)
        json_present = frame['structure_json'].astype(str).str.strip().ne('') if 'structure_json' in frame else pd.Series(False, index=frame.index)
        if (path_present & json_present).any():
            raise ValueError('A row cannot provide both structure_path and structure_json')
        has_structure = path_present | json_present
        if (~(has_composition | has_structure)).any():
            raise ValueError('Every row requires a composition or structure')
        all_have_composition = all_have_composition and bool(has_composition.all())
        all_have_structure = all_have_structure and bool(has_structure.all())
    if not sample_ids:
        raise ValueError('Input CSV contains no material rows')
    if duplicate_examples:
        raise ValueError(f'Duplicate sample_id values: {duplicate_examples}')
    excluded = [name for name in header if name not in material_columns]
    return {'source_csv': str(source_csv), 'source_sha256': material_projection_digest.hexdigest(), 'source_hash_scope': 'canonical_material_column_projection_only', 'n_samples': len(sample_ids), 'sample_ids': sample_ids, 'sample_ids_sha256': json_sha256(sample_ids), 'material_columns_used': material_columns, 'excluded_columns': excluded, 'excluded_target_aliases': [name for name in excluded if name.casefold() in TARGET_ALIASES], 'all_have_composition': all_have_composition, 'all_have_structure': all_have_structure}

def required_modalities(blocks: Sequence[str]) -> set[str]:
    unknown = sorted(set(blocks).difference(BLOCK_SPECS))
    if unknown:
        raise ValueError(f'Unknown MatterVial feature blocks: {unknown}')
    return {str(BLOCK_SPECS[name]['modality']) for name in blocks}

def validate_modality_coverage(inspection: Mapping[str, Any], blocks: Sequence[str]) -> None:
    modalities = required_modalities(blocks)
    if 'structure' in modalities and (not bool(inspection['all_have_structure'])):
        raise ValueError('Requested structure blocks, but one or more rows have composition only')
    if 'composition' in modalities and (not (bool(inspection['all_have_composition']) or bool(inspection['all_have_structure']))):
        raise ValueError('Requested composition blocks, but one or more rows lack composition/structure')

def write_target_free_subset(*, source_csv: Path, output_csv: Path, limit: int) -> dict[str, Any]:
    """Write a small smoke-test input while never reading non-material columns."""
    if limit <= 0:
        raise ValueError('subset limit must be positive')
    inspection = inspect_material_csv(source_csv)
    source_csv = Path(str(inspection['source_csv']))
    output_csv = output_csv.expanduser().resolve()
    if output_csv == source_csv:
        raise ValueError('Smoke subset must not overwrite its source CSV')
    frame = pd.read_csv(source_csv, usecols=inspection['material_columns_used'], dtype=str, keep_default_na=False, na_filter=False, nrows=min(limit, int(inspection['n_samples'])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_csv.parent, suffix='.csv.tmp', delete=False) as handle:
            temporary = handle.name
            frame.to_csv(handle, index=False)
        os.replace(temporary, output_csv)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    receipt = {'schema_version': PRODUCTION_SCHEMA_VERSION, **module_contract(), 'status': 'complete', 'source_csv': str(source_csv), 'source_sha256': inspection['source_sha256'], 'source_hash_scope': inspection['source_hash_scope'], 'output_csv': str(output_csv), 'output_sha256': sha256_file(output_csv), 'n_samples': len(frame), 'material_columns_used': inspection['material_columns_used'], 'excluded_columns_not_read': inspection['excluded_columns'], 'target_values_used': False, 'target_values_stored': False}
    _atomic_json(output_csv.with_suffix(output_csv.suffix + '.receipt.json'), receipt)
    return receipt

def _split_bounds(size: int, shards: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError('Cannot shard an empty material table')
    if shards <= 0:
        raise ValueError('shards must be positive')
    shards = min(shards, size)
    base, remainder = divmod(size, shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        stop = start + base + (1 if index < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return bounds

def prepare_material_input(*, source_csv: Path, output_dir: Path, shards: int, structure_root: Path | None=None, progress_path: Path | None=None) -> dict[str, Any]:
    """Create target-free normalized shards from arbitrary material rows."""
    inspection = inspect_material_csv(source_csv)
    source_csv = Path(str(inspection['source_csv']))
    output_dir = output_dir.resolve()
    structure_root = structure_root.expanduser().resolve() if structure_root is not None else source_csv.parent.resolve()
    manifest_path = output_dir / 'input_manifest.json'
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get('status') == 'complete' and existing.get('producer_module') == MODULE_NAME and (existing.get('producer_api_version') == MODULE_API_VERSION) and (existing.get('cache_contract') == CACHE_CONTRACT) and (existing.get('source_sha256') == inspection['source_sha256']) and (existing.get('sample_ids_sha256') == inspection['sample_ids_sha256']) and (int(existing.get('requested_shards', -1)) == min(shards, inspection['n_samples'])):
            valid = True
            for shard in existing.get('shards', []):
                path = output_dir / str(shard['path'])
                valid = valid and path.is_file() and (sha256_file(path) == shard.get('sha256'))
                if valid:
                    with path.open('r', encoding='utf-8') as handle:
                        for line in handle:
                            structure = json.loads(line).get('structure')
                            if not isinstance(structure, dict) or structure.get('kind') != 'file':
                                continue
                            structure_path = Path(str(structure.get('path', '')))
                            if not structure_path.is_file() or sha256_file(structure_path) != structure.get('file_sha256'):
                                valid = False
                                break
            if valid:
                return existing
        raise ValueError(f'Incompatible generated input exists at {output_dir}; use a new output root instead of mixing material identities')
    frame = pd.read_csv(source_csv, usecols=inspection['material_columns_used'], dtype=str, keep_default_na=False, na_filter=False)
    bounds = _split_bounds(len(frame), shards)
    shard_root = output_dir / 'normalized_shards'
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_receipts: list[dict[str, Any]] = []
    global_record_hashes: list[str] = []
    heartbeat = ProgressHeartbeat((progress_path or output_dir / 'progress.json').resolve(), interval_seconds=15)
    with heartbeat:
        heartbeat.update('material_input_validation', detail='target-free CSV columns locked', completed=1, total=len(frame), progress_unit_label='materials')
        normalized_count = 0
        for shard_index, (start, stop) in enumerate(bounds):
            path = shard_root / f'shard_{shard_index:05d}.jsonl'
            temporary: str | None = None
            record_hashes: list[str] = []
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=shard_root, suffix='.tmp', delete=False) as handle:
                    temporary = handle.name
                    for source_row in range(start, stop):
                        row = frame.iloc[source_row]
                        sample_id = str(row['sample_id']).strip()
                        explicit = str(row.get('composition', '')).strip()
                        structure_record, structure = _normalize_structure(structure_path=str(row.get('structure_path', '')).strip(), structure_json=str(row.get('structure_json', '')).strip(), structure_root=structure_root)
                        derived = None if structure is None else structure.composition.reduced_composition.alphabetical_formula
                        canonical = _canonical_composition(explicit) if explicit else derived
                        if canonical is None:
                            raise ValueError(f'{sample_id}: missing both composition and structure')
                        if derived is not None and canonical != derived:
                            raise ValueError(f'{sample_id}: composition {canonical!r} disagrees with structure {derived!r}')
                        normalized = {'sample_id': sample_id, 'source_row': source_row, 'composition': canonical, 'structure': structure_record}
                        normalized['record_sha256'] = json_sha256(normalized)
                        record_hashes.append(str(normalized['record_sha256']))
                        global_record_hashes.append(str(normalized['record_sha256']))
                        handle.write(json.dumps(normalized, sort_keys=True, separators=(',', ':')) + '\n')
                        normalized_count += 1
                        if normalized_count == len(frame) or normalized_count % 25 == 0:
                            heartbeat.update('material_normalization', detail=f'shard {shard_index + 1}/{len(bounds)}', completed=normalized_count, total=len(frame), progress_unit_label='materials')
                os.replace(temporary, path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            shard_receipts.append({'shard_index': shard_index, 'start': start, 'stop': stop, 'n_samples': stop - start, 'path': str(path.relative_to(output_dir)), 'sha256': sha256_file(path), 'record_hashes_sha256': json_sha256(record_hashes)})
    manifest = {'schema_version': PRODUCTION_SCHEMA_VERSION, **module_contract(), 'status': 'complete', 'source_csv': str(source_csv), 'source_sha256': inspection['source_sha256'], 'source_hash_scope': inspection['source_hash_scope'], 'n_samples': inspection['n_samples'], 'sample_ids': inspection['sample_ids'], 'sample_ids_sha256': inspection['sample_ids_sha256'], 'material_columns_used': inspection['material_columns_used'], 'excluded_columns': inspection['excluded_columns'], 'excluded_target_aliases': inspection['excluded_target_aliases'], 'non_material_columns_materialized': False, 'target_values_used': False, 'target_values_stored': False, 'target_exclusion_policy': 'CSV_usecols_material_whitelist_before_value_materialization', 'structure_root': str(structure_root), 'all_have_composition': inspection['all_have_composition'], 'all_have_structure': inspection['all_have_structure'], 'requested_shards': len(bounds), 'material_records_sha256': json_sha256(global_record_hashes), 'shards': shard_receipts}
    manifest['input_identity_sha256'] = json_sha256({key: manifest[key] for key in ('source_sha256', 'sample_ids_sha256', 'material_records_sha256', 'requested_shards', 'cache_contract')})
    _atomic_json(manifest_path, manifest)
    return manifest

def _load_normalized_shard(manifest_path: Path, shard_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    if manifest.get('status') != 'complete' or manifest.get('target_values_used') is not False:
        raise ValueError('Material input manifest is incomplete or target-tainted')
    _require_current_contract(manifest, artifact='material input manifest')
    records = manifest.get('shards', [])
    if not 0 <= shard_index < len(records):
        raise ValueError(f'Invalid shard index {shard_index}')
    metadata = records[shard_index]
    path = manifest_path.parent / str(metadata['path'])
    if sha256_file(path) != metadata.get('sha256'):
        raise ValueError(f'Normalized input shard checksum mismatch: {path}')
    values: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict) or value.get('record_sha256') != json_sha256({key: item for key, item in value.items() if key != 'record_sha256'}):
                raise ValueError(f'Normalized input record checksum mismatch: {path}')
            values.append(value)
    if len(values) != int(metadata['n_samples']):
        raise ValueError(f'Normalized input shard row count mismatch: {path}')
    return (manifest, values)

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def _source_commit(package_file: str, expected_commit: str) -> tuple[str | None, str]:
    root = Path(package_file).resolve().parent.parent
    receipt_path = root / '.hpsafe_vendor_receipt.json'
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        return (str(receipt.get('commit')), 'verified_offline_vendor_receipt')
    git_root = root / '.git'
    if git_root.exists():
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return (result.stdout.strip(), 'local_git_checkout')
    return (None, f'unverified_expected_{expected_commit}')

def runtime_identity(*, require_verified: bool=True, require_modnet: bool=False) -> dict[str, Any]:
    import mattervial
    commit, source_mode = _source_commit(mattervial.__file__, MATTERVIAL_COMMIT)
    if require_verified and commit != MATTERVIAL_COMMIT:
        raise RuntimeError('MatterVial source is not proven to be the pinned commit. Install the verified offline vendor snapshot before production featurization.')
    modnet_commit: str | None = None
    modnet_mode: str | None = None
    if require_modnet:
        import modnet
        modnet_commit, modnet_mode = _source_commit(modnet.__file__, MODNET_COMMIT)
        if require_verified and modnet_commit != MODNET_COMMIT:
            raise RuntimeError('MODNet source is not proven to be the pinned 0.4.5 commit')
    return {'mattervial_version': str(getattr(mattervial, '__version__', 'unknown')), 'mattervial_commit': commit, 'mattervial_source_mode': source_mode, 'modnet_version': _package_version('modnet') if require_modnet else None, 'modnet_commit': modnet_commit, 'modnet_source_mode': modnet_mode, 'tensorflow_version': _package_version('tensorflow'), 'torch_version': _package_version('torch'), 'pymatgen_version': _package_version('pymatgen'), 'matminer_version': _package_version('matminer'), 'orb_models_version': _package_version('orb-models'), 'python': sys.version.split()[0], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'verified': commit == MATTERVIAL_COMMIT and (not require_modnet or modnet_commit == MODNET_COMMIT)}

class _ProgressTee:
    """Forward official output while turning its counters into a durable heartbeat."""
    _fraction = re.compile('(?<!\\d)(\\d{1,9})/(\\d{1,9})(?!\\d)')

    def __init__(self, stream: TextIO, heartbeat: ProgressHeartbeat, unit: str) -> None:
        self.stream = stream
        self.heartbeat = heartbeat
        self.unit = unit
        self.buffer = ''
        self.last_update = 0.0
        self.model_index = -1

    def flush(self) -> None:
        self.stream.flush()

    def write(self, value: str) -> int:
        written = self.stream.write(value)
        self.stream.flush()
        self.buffer = (self.buffer + value)[-4096:]
        now = time.monotonic()
        if 'Processing model:' in value:
            self.model_index += 1
        matches = list(self._fraction.finditer(self.buffer))
        if matches and now - self.last_update >= 1.5:
            match = matches[-1]
            completed, total = (int(match.group(1)), int(match.group(2)))
            if 0 <= completed <= total and total > 0:
                fraction = completed / total
                if self.unit == 'mvl_all' and self.model_index >= 0:
                    fraction = (min(self.model_index, 5) + fraction) / 6.0
                scaled = min(920, max(30, int(30 + 890 * fraction)))
                self.heartbeat.update('feature_extraction', detail=f'{self.unit}: official backend {completed}/{total}', completed=scaled, total=1000, progress_unit_label='feature-progress')
                self.last_update = now
                self.buffer = self.buffer[match.end():]
        return written

def _as_feature_frame(value: Any, *, sample_ids: Sequence[str]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=False)
    else:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(f'Feature provider returned shape {array.shape}')
        frame = pd.DataFrame(array, columns=[f'feature_{index + 1}' for index in range(array.shape[1])])
    if len(frame) != len(sample_ids) or frame.shape[1] == 0:
        raise ValueError(f'Feature provider returned {frame.shape}; expected {len(sample_ids)} non-empty rows')
    columns = frame.columns.astype(str).tolist()
    if len(set(columns)) != len(columns):
        raise ValueError('Feature provider returned duplicate feature names')
    frame.columns = columns
    frame.index = list(sample_ids)
    return frame.apply(pd.to_numeric, errors='coerce')

def _official_features(unit: str, *, sample_ids: list[str], compositions: list[str], structures: list[Any] | None, n_jobs: int, batch_size: int) -> dict[str, pd.DataFrame]:
    series_index = pd.Index(sample_ids, dtype=str)
    composition_series = pd.Series(compositions, index=series_index)
    structure_series = None if structures is None else pd.Series(structures, index=series_index)
    if unit == 'matminer':
        from modnet.preprocessing import MODData
        from pymatgen.core import Composition
        materials = structures if structures is not None else [Composition(value) for value in compositions]
        data = MODData(materials=materials, structure_ids=sample_ids)
        data.featurize(fast=False, n_jobs=n_jobs, drop_allnan=False)
        return {'matminer': _as_feature_frame(data.df_featurized, sample_ids=sample_ids)}
    if unit in {'mvl_all', 'mvl16', 'mvl32'}:
        if structure_series is None:
            raise ValueError(f'{unit} requires crystal structures')
        from mattervial import MVLFeaturizer
        layers = ['layer32', 'layer16'] if unit == 'mvl_all' else unit.replace('mvl', 'layer')
        frame = _as_feature_frame(MVLFeaturizer(layers=layers).get_features(structure_series), sample_ids=sample_ids)
        result: dict[str, pd.DataFrame] = {}
        if unit in {'mvl_all', 'mvl16'}:
            result['mvl16'] = frame.loc[:, [name for name in frame if name.startswith('MVL16_')]]
        if unit in {'mvl_all', 'mvl32'}:
            result['mvl32'] = frame.loc[:, [name for name in frame if name.startswith('MVL32_')]]
        return {name: _as_feature_frame(value, sample_ids=sample_ids) for name, value in result.items()}
    if unit in {'megnet_mm', 'megnet_ofm'}:
        if structure_series is None:
            raise ValueError(f'{unit} requires crystal structures')
        from mattervial import DescriptorMEGNetFeaturizer
        descriptor = 'l-MM_v1' if unit == 'megnet_mm' else 'l-OFM_v1'
        frame = DescriptorMEGNetFeaturizer(base_descriptor=descriptor).get_features(structure_series)
        return {unit: _as_feature_frame(frame, sample_ids=sample_ids)}
    if unit in {'roost_mpgap', 'roost_oqmd_eform'}:
        from mattervial import RoostModelFeaturizer
        model_type = 'mpgap' if unit == 'roost_mpgap' else 'oqmd_eform'
        frame = RoostModelFeaturizer(model_type=model_type, batch_size=batch_size).get_features(composition_series)
        return {unit: _as_feature_frame(frame, sample_ids=sample_ids)}
    if unit == 'orb_v3':
        if structure_series is None:
            raise ValueError('orb_v3 requires crystal structures')
        from mattervial import ORBFeaturizer
        frame = ORBFeaturizer(model_name='ORB_v3', device='cuda').get_features(structure_series)
        return {unit: _as_feature_frame(frame, sample_ids=sample_ids)}
    raise ValueError(f'Unsupported feature unit: {unit}')

def _valid_shard_receipt(path: Path, signature: str) -> bool:
    try:
        receipt = _read_json(path)
        if receipt.get('status') != 'complete' or receipt.get('request_signature') != signature:
            return False
        root = path.parent
        return all(((root / str(receipt[key])).is_file() and sha256_file(root / str(receipt[key])) == receipt[f'{key}_sha256'] for key in ('features_path', 'sample_ids_path', 'feature_names_path', 'availability_path')))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False

def generate_feature_shard(request_path: Path) -> dict[str, Any]:
    request_path = request_path.resolve()
    request = _read_json(request_path)
    _require_current_contract(request, artifact='feature shard request')
    unit = str(request['unit'])
    if unit not in UNIT_BLOCKS:
        raise ValueError(f'Unknown feature unit {unit!r}')
    provider = str(request.get('provider', 'mattervial_official'))
    if provider != 'mattervial_official':
        raise ValueError('Expected the MatterVial feature provider')
    manifest_path = Path(str(request['input_manifest'])).resolve()
    shard_index = int(request['shard_index'])
    output_dir = Path(str(request['output_dir'])).resolve()
    progress_path = Path(str(request.get('progress_path', output_dir / 'progress.json'))).resolve()
    manifest, records = _load_normalized_shard(manifest_path, shard_index)
    expected_blocks = list(UNIT_BLOCKS[unit])
    if [str(value) for value in request.get('blocks', expected_blocks)] != expected_blocks:
        raise ValueError(f'Feature shard request blocks do not match unit {unit!r}')
    shard_metadata = manifest['shards'][shard_index]
    if int(request.get('start', shard_metadata['start'])) != int(shard_metadata['start']) or int(request.get('stop', shard_metadata['stop'])) != int(shard_metadata['stop']):
        raise ValueError('Feature shard request row bounds differ from the immutable input manifest')
    input_identity = str(manifest['input_identity_sha256'])
    request_signature = json_sha256({'schema_version': PRODUCTION_SCHEMA_VERSION, **module_contract(), 'unit': unit, 'provider': provider, 'input_identity': input_identity, 'shard_index': shard_index, 'block_names': UNIT_BLOCKS[unit], 'n_jobs': int(request.get('n_jobs', 1)), 'batch_size': int(request.get('batch_size', 128)), 'mattervial_commit': MATTERVIAL_COMMIT, 'modnet_commit': MODNET_COMMIT if unit == 'matminer' else None})
    expected_receipts = [output_dir / block / 'receipt.json' for block in UNIT_BLOCKS[unit]]
    if expected_receipts and all((_valid_shard_receipt(path, request_signature) for path in expected_receipts)):
        return {**module_contract(), 'status': 'cache_hit', 'unit': unit, 'shard_index': shard_index, 'receipts': [str(path) for path in expected_receipts]}
    sample_ids = [str(value['sample_id']) for value in records]
    compositions = [str(value['composition']) for value in records]
    needs_structure = any((BLOCK_SPECS[name]['modality'] == 'structure' for name in UNIT_BLOCKS[unit]))
    structures = [_structure_from_record(value['structure']) for value in records] if needs_structure else None
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = ProgressHeartbeat(progress_path, interval_seconds=15)
    with heartbeat:
        heartbeat.update('input_validation', detail=f'{unit}: normalized shard {shard_index}', completed=10, total=1000, progress_unit_label='feature-progress')
        identity = runtime_identity(require_verified=not bool(request.get('allow_unverified_provider', False)), require_modnet=unit == 'matminer')
        tee = _ProgressTee(sys.stdout, heartbeat, unit)
        error_tee = _ProgressTee(sys.stderr, heartbeat, unit)
        heartbeat.update('provider_initialization', detail=f'{unit}: pinned MatterVial backend', completed=20, total=1000, progress_unit_label='feature-progress')
        with redirect_stdout(tee), redirect_stderr(error_tee):
            features = _official_features(unit, sample_ids=sample_ids, compositions=compositions, structures=structures, n_jobs=int(request.get('n_jobs', 16)), batch_size=int(request.get('batch_size', 128)))
        if set(features) != set(UNIT_BLOCKS[unit]):
            raise ValueError(f'Provider block mismatch: {sorted(features)} != {sorted(UNIT_BLOCKS[unit])}')
        receipts: list[str] = []
        total_nonfinite = 0
        total_failed_rows = 0
        for block_name, frame in features.items():
            frame = _as_feature_frame(frame, sample_ids=sample_ids)
            array = frame.to_numpy(dtype=np.float32)
            finite = np.isfinite(array)
            availability_policy = str(BLOCK_SPECS[block_name]['availability'])
            if availability_policy == 'all_finite':
                available = np.all(finite, axis=1)
            elif availability_policy == 'any_finite':
                available = np.any(finite, axis=1)
            else:
                raise ValueError(f'Unsupported availability policy for {block_name}: {availability_policy}')
            nonfinite = int((~np.isfinite(array)).sum())
            failed_rows = int((~available).sum())
            total_nonfinite += nonfinite
            total_failed_rows += failed_rows
            block_dir = output_dir / block_name
            features_path = block_dir / 'features.npy'
            sample_path = block_dir / 'sample_ids.npy'
            names_path = block_dir / 'feature_names.json'
            availability_path = block_dir / 'availability.npy'
            _atomic_npy(features_path, array)
            _atomic_npy(sample_path, np.asarray(sample_ids, dtype=str))
            _atomic_json(names_path, frame.columns.astype(str).tolist())
            _atomic_npy(availability_path, available.astype(np.bool_))
            receipt = {'schema_version': PRODUCTION_SCHEMA_VERSION, **module_contract(), 'status': 'complete', 'provider': provider, 'provider_identity': identity, 'request_signature': request_signature, 'input_identity_sha256': input_identity, 'unit': unit, 'block': block_name, 'shard_index': shard_index, 'start': int(manifest['shards'][shard_index]['start']), 'stop': int(manifest['shards'][shard_index]['stop']), 'n_samples': len(sample_ids), 'n_features': int(array.shape[1]), 'dtype': str(array.dtype), 'features_path': features_path.name, 'features_path_sha256': sha256_file(features_path), 'sample_ids_path': sample_path.name, 'sample_ids_path_sha256': sha256_file(sample_path), 'feature_names_path': names_path.name, 'feature_names_path_sha256': sha256_file(names_path), 'availability_path': availability_path.name, 'availability_path_sha256': sha256_file(availability_path), 'nonfinite_count': nonfinite, 'availability_policy': availability_policy, 'all_finite_rows': int(np.all(finite, axis=1).sum()), 'any_finite_rows': int(np.any(finite, axis=1).sum()), 'unavailable_rows': failed_rows, 'all_nonfinite_rows': int((~np.any(finite, axis=1)).sum()), 'unavailable_sample_ids_preview': [sample_ids[index] for index in np.flatnonzero(~available)[:20]], 'target_values_used': False, 'target_values_stored': False}
            _atomic_json(block_dir / 'receipt.json', receipt)
            receipts.append(str(block_dir / 'receipt.json'))
        heartbeat.update('artifact_validation', detail=f'{unit}: checksums committed', completed=1000, total=1000, progress_unit_label='feature-progress', nonfinite_count=total_nonfinite, failed_material_count=total_failed_rows)
    result = {**module_contract(), 'status': 'complete', 'unit': unit, 'shard_index': shard_index, 'request_signature': request_signature, 'receipts': receipts, 'target_values_used': False}
    _atomic_json(output_dir / 'job_receipt.json', result)
    return result

def _copy_shards_to_memmap(*, block: str, shard_receipts: Sequence[Path], expected_ids: Sequence[str], output_path: Path) -> dict[str, Any]:
    pairs = sorted(((_read_json(path), path) for path in shard_receipts), key=lambda value: int(value[0]['start']))
    ordered = [receipt for receipt, _ in pairs]
    if not ordered or int(ordered[0]['start']) != 0:
        raise ValueError(f'{block}: shard coverage does not begin at zero')
    feature_names: list[str] | None = None
    cursor = 0
    width: int | None = None
    for receipt, receipt_path in pairs:
        if receipt.get('block') != block or int(receipt['start']) != cursor:
            raise ValueError(f'{block}: non-contiguous or wrong shard receipt')
        root = receipt_path.parent
        ids = np.load(root / str(receipt['sample_ids_path']), allow_pickle=False).astype(str)
        names = json.loads((root / str(receipt['feature_names_path'])).read_text(encoding='utf-8'))
        if not isinstance(names, list) or not all((isinstance(name, str) for name in names)):
            raise ValueError(f'{block}: invalid feature-name schema')
        current_width = int(receipt['n_features'])
        if feature_names is None:
            feature_names = names
            width = current_width
        elif feature_names != names or width != current_width:
            raise ValueError(f'{block}: feature schema changed between shards')
        stop = int(receipt['stop'])
        if ids.tolist() != list(expected_ids[cursor:stop]):
            raise ValueError(f'{block}: sample order differs from normalized input')
        cursor = stop
    if cursor != len(expected_ids) or width is None or feature_names is None:
        raise ValueError(f'{block}: incomplete shard coverage {cursor}/{len(expected_ids)}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f'.{output_path.name}.{os.getpid()}.tmp.npy'
    if temporary.exists():
        temporary.unlink()
    destination = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(expected_ids), width))
    availability = np.zeros(len(expected_ids), dtype=np.bool_)
    nonfinite = 0
    cursor = 0
    try:
        for receipt, receipt_path in pairs:
            root = receipt_path.parent
            values = np.load(root / str(receipt['features_path']), mmap_mode='r', allow_pickle=False)
            available = np.load(root / str(receipt['availability_path']), allow_pickle=False)
            stop = cursor + len(values)
            destination[cursor:stop] = values
            availability[cursor:stop] = available
            nonfinite += int((~np.isfinite(values)).sum())
            cursor = stop
        destination.flush()
        del destination
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {'path': output_path.name, 'shape': [len(expected_ids), width], 'dtype': 'float32', 'sha256': sha256_file(output_path), 'feature_names': feature_names, 'nonfinite_count': nonfinite, 'available_rows': int(availability.sum()), 'all_nonfinite_rows': int((~availability).sum()), 'availability': availability}

def merge_feature_cache(plan_path: Path) -> dict[str, Any]:
    plan = _read_json(plan_path.resolve())
    _require_current_contract(plan, artifact='feature production plan')
    input_manifest_path = Path(str(plan['input_manifest'])).resolve()
    manifest = _read_json(input_manifest_path)
    _require_current_contract(manifest, artifact='material input manifest')
    expected_ids = [str(value) for value in manifest['sample_ids']]
    cache_root = Path(str(plan['cache_root'])).resolve()
    blocks = [str(value) for value in plan['blocks']]
    cache_root.mkdir(parents=True, exist_ok=True)
    heartbeat = ProgressHeartbeat(cache_root / 'merge_progress.json', interval_seconds=15)
    block_receipts: dict[str, Any] = {}
    availability_columns: list[np.ndarray] = []
    with heartbeat:
        for index, block in enumerate(blocks):
            heartbeat.update('merge_feature_blocks', detail=block, completed=index, total=len(blocks) + 1, progress_unit_label='feature-blocks')
            shard_paths = [Path(str(value)).resolve() for value in plan['block_shard_receipts'][block]]
            metadata = _copy_shards_to_memmap(block=block, shard_receipts=shard_paths, expected_ids=expected_ids, output_path=cache_root / f'{block}.npy')
            availability_columns.append(metadata.pop('availability'))
            block_receipts[block] = metadata
        sample_path = cache_root / 'sample_ids.npy'
        _atomic_npy(sample_path, np.asarray(expected_ids, dtype=str))
        availability = np.column_stack(availability_columns).astype(np.bool_)
        availability_path = cache_root / 'availability.npy'
        _atomic_npy(availability_path, availability)
        schema = {'schema_version': PRODUCTION_SCHEMA_VERSION, **module_contract(), 'source_kind': 'arbitrary_material_csv', 'sample_count': len(expected_ids), 'block_order': blocks, 'blocks': {name: value['feature_names'] for name, value in block_receipts.items()}}
        schema_path = cache_root / 'feature_schema.json'
        _atomic_json(schema_path, schema)
        cache_key = json_sha256({'input_identity_sha256': manifest['input_identity_sha256'], 'blocks': {name: value['sha256'] for name, value in block_receipts.items()}, 'schema_version': PRODUCTION_SCHEMA_VERSION, 'cache_contract': CACHE_CONTRACT})
        receipt = {'schema_version': 3, **module_contract(), 'status': 'complete', 'cache_kind': 'production_material_features', 'cache_key': cache_key, 'provider': 'MatterVial', 'provider_commit': MATTERVIAL_COMMIT, 'input_manifest': str(input_manifest_path), 'input_manifest_sha256': sha256_file(input_manifest_path), 'input_identity_sha256': manifest['input_identity_sha256'], 'sample_ids_path': sample_path.name, 'sample_ids_sha256': sha256_file(sample_path), 'feature_schema_sha256': sha256_file(schema_path), 'availability_path': availability_path.name, 'availability_sha256': sha256_file(availability_path), 'block_order': blocks, 'blocks': block_receipts, 'target_values_stored': False, 'source_target_columns_read': False, 'source_target_values_accessed_by_cache_builder': False, 'target_exclusion_policy': manifest['target_exclusion_policy'], 'sample_rows_with_all_requested_blocks': int(np.all(availability, axis=1).sum()), 'sample_rows_with_any_requested_block': int(np.any(availability, axis=1).sum()), 'production_readiness': 'ready' if bool(np.all(availability)) else 'partial_features_require_model_level_abstention_or_repair'}
        _atomic_json(cache_root / 'receipt.json', receipt)
        heartbeat.update('complete', detail=receipt['production_readiness'], completed=len(blocks) + 1, total=len(blocks) + 1, progress_unit_label='feature-blocks', failed_material_count=int((~np.all(availability, axis=1)).sum()))
    return receipt

def _open_cache_arrays(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    root = root.expanduser().resolve()
    receipt = _read_json(root / 'receipt.json')
    if receipt.get('status') != 'complete' or receipt.get('target_values_stored') is not False:
        raise ValueError(f'Feature cache is incomplete or target-tainted: {root}')
    sample_path = root / str(receipt['sample_ids_path'])
    if not sample_path.is_file() or sha256_file(sample_path) != receipt.get('sample_ids_sha256'):
        raise ValueError(f'Feature cache sample IDs failed checksum validation: {root}')
    sample_ids = np.load(sample_path, allow_pickle=False).astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f'Feature cache has duplicate sample IDs: {root}')
    return (receipt, sample_ids)

def _feature_catalog(root: Path, receipt: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    catalog: dict[str, tuple[str, int]] = {}
    for block_name, metadata in receipt.get('blocks', {}).items():
        path = root / str(metadata['path'])
        if not path.is_file() or sha256_file(path) != metadata.get('sha256'):
            raise ValueError(f'Feature block failed checksum validation: {path}')
        names = [str(value) for value in metadata.get('feature_names', [])]
        shape = [int(value) for value in metadata.get('shape', [])]
        if len(shape) != 2 or len(names) != shape[1]:
            raise ValueError(f'Feature schema/shape mismatch for {block_name!r}')
        for index, name in enumerate(names):
            if name in catalog:
                raise ValueError(f'Duplicate feature name across cache blocks: {name!r}')
            catalog[name] = (str(block_name), index)
    return catalog

def _select_named_features(root: Path, receipt: Mapping[str, Any], catalog: Mapping[str, tuple[str, int]], names: Sequence[str], rows: np.ndarray) -> np.ndarray:
    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(f'Cache is missing required features: {missing[:10]}')
    grouped: dict[str, list[tuple[int, int]]] = {}
    for output_column, name in enumerate(names):
        block, source_column = catalog[name]
        grouped.setdefault(block, []).append((output_column, source_column))
    result = np.empty((len(rows), len(names)), dtype=np.float32)
    for block, columns in grouped.items():
        metadata = receipt['blocks'][block]
        values = np.load(root / str(metadata['path']), mmap_mode='r', allow_pickle=False)
        output_columns = [value[0] for value in columns]
        source_columns = [value[1] for value in columns]
        result[:, output_columns] = values[np.ix_(rows, source_columns)]
    return result

@dataclass(frozen=True)
class ProductionFeatureCache:
    root: Path
    receipt: dict[str, Any]
    sample_ids: np.ndarray

    @classmethod
    def open(cls, root: Path, *, verify_hashes: bool=True) -> ProductionFeatureCache:
        root = root.resolve()
        receipt = _read_json(root / 'receipt.json')
        if receipt.get('status') != 'complete' or receipt.get('cache_kind') != 'production_material_features':
            raise ValueError(f'Not a complete production feature cache: {root}')
        if receipt.get('cache_contract') not in {None, CACHE_CONTRACT}:
            raise ValueError(f'Unsupported production feature cache contract: {root}')
        if receipt.get('target_values_stored') is not False:
            raise ValueError('Production feature cache contains targets')
        sample_path = root / str(receipt['sample_ids_path'])
        sample_ids = np.load(sample_path, allow_pickle=False).astype(str)
        if verify_hashes:
            expected = {sample_path: receipt['sample_ids_sha256'], root / 'feature_schema.json': receipt['feature_schema_sha256'], root / str(receipt['availability_path']): receipt['availability_sha256'], **{root / str(block['path']): block['sha256'] for block in receipt['blocks'].values()}}
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise ValueError(f'Production feature cache checksum mismatch: {path}')
        return cls(root=root, receipt=receipt, sample_ids=sample_ids)

    def block(self, name: str) -> np.ndarray:
        metadata = self.receipt['blocks'].get(name)
        if metadata is None:
            raise KeyError(name)
        return np.load(self.root / str(metadata['path']), mmap_mode='r', allow_pickle=False)

    def row_indices(self, sample_ids: Sequence[str]) -> np.ndarray:
        """Return cache rows in consumer order, failing on missing or duplicate IDs."""
        requested = [str(value) for value in sample_ids]
        if len(set(requested)) != len(requested):
            raise ValueError('Consumer requested duplicate sample IDs')
        lookup = {value: index for index, value in enumerate(self.sample_ids.tolist())}
        missing = [value for value in requested if value not in lookup]
        if missing:
            raise KeyError(f'Production feature cache is missing sample IDs: {missing[:10]}')
        return np.asarray([lookup[value] for value in requested], dtype=np.int64)

    def aligned_blocks(self, sample_ids: Sequence[str], blocks: Sequence[str]) -> dict[str, np.ndarray]:
        """Read only requested rows and blocks in the exact downstream order."""
        rows = self.row_indices(sample_ids)
        result: dict[str, np.ndarray] = {}
        for name in blocks:
            values = self.block(str(name))
            result[str(name)] = np.asarray(values[rows], dtype=np.float32)
        return result

    def aligned_availability(self, sample_ids: Sequence[str]) -> np.ndarray:
        rows = self.row_indices(sample_ids)
        path = self.root / str(self.receipt['availability_path'])
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        return np.asarray(values[rows], dtype=np.bool_)
