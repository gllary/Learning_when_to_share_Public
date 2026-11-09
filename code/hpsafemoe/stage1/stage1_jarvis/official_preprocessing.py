"""Literal structural roundtrips from the pinned public contributions."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime/tf_deps'))

def kgcnn_structure(raw):
    from jarvis.core.atoms import Atoms
    from jarvis.io.vasp.inputs import Poscar
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure
    # The benchmark populate command writes POSCAR; run.py rereads it, writes
    # CIF; CrystalDataset calls CifParser.get_structures() (primitive=True).
    poscar = Poscar(Atoms.from_dict(raw)).to_string()
    pmg = Poscar.from_string(poscar).atoms.pymatgen_converter()
    st = CifParser.from_string(pmg.to(fmt='cif')).get_structures(primitive=True)[0]
    return Structure.from_dict(json.loads(json.dumps(st.as_dict())))

def tree_structure(raw):
    from jarvis.core.atoms import Atoms
    st = Atoms.from_dict(raw).pymatgen_converter()
    for site in st.sites:
        site.to_unit_cell(in_place=True)
    return st

def alignn_atoms(raw):
    from jarvis.core.atoms import Atoms
    from jarvis.io.vasp.inputs import Poscar
    return Poscar.from_string(Poscar(Atoms.from_dict(raw)).to_string()).atoms
