"""Read-only, fingerprinted reuse of Type0B's five-point boundary machinery.

This reference is intentionally separate from the older genus-one recursion
pin. No Type0B structure constants, pictures, or amplitudes are used here.
Set SO7E8_TYPE0B_ROOT to relocate the sibling checkout. Updating these source
fingerprints is a reviewed dependency change, not an automatic git operation.
"""
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace


SOURCE_REVISION = "0a4c344553af0f10c1e19c29827796bba9883de8"
SOURCE_HASHES = {
    "fivepoint_elliptic": "46a4c93b160082d1a81bf781c57ddc0a9515766e7e880b169534438acf844d2c",
    "fivepoint_elliptic_jets": "a83e871f07fffebea2f164e1213059878e4a579f976af3a673a70dade733aed4",
    "fivepoint_ope_integrator": "ca0eb307acda3c1fde45d1a94467c0f66a350472ac0da7896c7f8f9a0de199da",
    "fivepoint_collar_forest": "6295751b0153f22fd73f963b053f88ce2c42d355331bac158373a29b1b24ebef",
}


def boundary_root():
    default = Path(__file__).resolve().parents[2] / "Type0B-Matrix"
    return Path(os.environ.get("SO7E8_TYPE0B_ROOT", default)).expanduser().resolve()


def source_manifest(root=None):
    root = boundary_root() if root is None else Path(root).resolve()
    directory = root / "Code/sphere_five_point/type0b_ns_five_tachyon"
    hashes = {}
    for name, expected in SOURCE_HASHES.items():
        path = directory / (name + ".py")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; set SO7E8_TYPE0B_ROOT")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Type0B boundary source changed: {path}; review before updating its pin")
        hashes[name + ".py"] = actual
    return dict(root=str(root), reference_revision=SOURCE_REVISION, sha256=hashes)


@lru_cache(maxsize=4)
def _load(root):
    manifest = source_manifest(root)
    directory = Path(root) / "Code/sphere_five_point/type0b_ns_five_tachyon"
    # The upstream jet module imports its Type0B application for OTHER
    # classes. We use only Jet and pillow_jet_geometry, never that assembly.
    # Restore search paths even if an upstream import fails.
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        modules = {}
        for name in SOURCE_HASHES:
            existing = sys.modules.get(name)
            if existing is not None and Path(existing.__file__).resolve() != directory / (name + ".py"):
                raise RuntimeError(f"Conflicting Type0B module already loaded: {name}")
            modules[name] = importlib.import_module(name)
    finally:
        sys.path[:] = original_path
    return SimpleNamespace(
        elliptic=modules["fivepoint_elliptic"],
        jets=modules["fivepoint_elliptic_jets"],
        integration=modules["fivepoint_ope_integrator"],
        forest=modules["fivepoint_collar_forest"],
        manifest=manifest,
    )


def boundary_layers():
    return _load(str(boundary_root()))
