"""Read-only numerical dependency for the all-NS HetSO(23) five-point task.

This pin is independent of the older genus-one pin and of every SO(7)
adapter. Only coefficient/coordinate/jet/atlas/integration APIs are reused;
Type0B's physical sixteen-term PCO sum and normalization are not imported
as heterotic data. Working-copy content is identified by hashes, not HEAD.
"""
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace


DIRECTORIES = {
    "recursion": "Code/c_Recursion",
    "sphere": "Code/sphere_five_point/type0b_ns_five_tachyon",
}
PINS = {
    "ns_multipoint_c_recursion": ("recursion", "9387baa58c25a24d155716a48fb70ec14b9a18e52c24a6d5bc0f190cedb92b75"),
    "ns_global_osp_block": ("recursion", "0903de11a2e12b7d70517514baffb43a7b244f92f086a7cbfc2b3599763564d9"),
    "ns_human_convention": ("recursion", "8f1d38b550333c6ff0ba577317110b1457bdf01a3429327c32acd8a5cad570cc"),
    "ns_recursion_recipe": ("recursion", "fe5e0e4497ddf6edf81e87cc2eff1d2ecd763c7b0d3b6812bfc6c74fd9dcfd76"),
    "fivepoint_runtime": ("sphere", "48078113eeb1c09da440c4d6240952c4e11cba8b83e9c677ad94142699b22c13"),
    "fivepoint_elliptic": ("sphere", "46a4c93b160082d1a81bf781c57ddc0a9515766e7e880b169534438acf844d2c"),
    "fivepoint_qw": ("sphere", "5dc7bcd073b6136691ff62b008bbea077ac0ef7adb2adc25f2bc76ca3739247e"),
    "fivepoint_elliptic_jets": ("sphere", "a83e871f07fffebea2f164e1213059878e4a579f976af3a673a70dade733aed4"),
    "fivepoint_elliptic_atlas": ("sphere", "d8501a47b3b7cc78a3c66ca2f18878997d0000ad5929ba67c607abd4753ea7fd"),
    "fivepoint_ope_integrator": ("sphere", "ca0eb307acda3c1fde45d1a94467c0f66a350472ac0da7896c7f8f9a0de199da"),
    "fivepoint_collar_forest": ("sphere", "6295751b0153f22fd73f963b053f88ce2c42d355331bac158373a29b1b24ebef"),
}
CONVENTION_PINS = {
    "Code/sphere_five_point/type0b_ns_five_tachyon/fivepoint_bry_qw.py":
        "f69bd4ea475b5be910e9d8ded1cbda8422e9d06b5f1e1c4ca002cc7ca42fff10",
    "Data Set/type0b_fivepoint_bry_native_20260914/DERIVATION.md":
        "97c52b53435adc78ffdfe1fe72ba023f0fcd0fb4acd598bbbffab16137b974fd",
}


def reference_root():
    default = Path(__file__).resolve().parents[2] / "Type0B-Matrix"
    return Path(os.environ.get("SPIN23_FIVEPOINT_TYPE0B_ROOT", default)).expanduser().resolve()


def source_manifest():
    root = reference_root()
    hashes = {}
    for name, (directory, expected) in PINS.items():
        relative = DIRECTORIES[directory] + "/" + name + ".py"
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Reviewed Type0B source changed: {path}")
        hashes[relative] = actual
    for relative, expected in CONVENTION_PINS.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Reviewed Type0B convention changed: {root / relative}")
        hashes[relative] = actual
    return {"root": str(root), "sha256": hashes, "physical_Type0B_assembly_used": False}


@lru_cache(maxsize=1)
def layers():
    manifest = source_manifest()
    root = Path(manifest["root"])
    for name, (directory, _) in PINS.items():
        old = sys.modules.get(name)
        if old is not None and Path(old.__file__).resolve() != root / DIRECTORIES[directory] / (name + ".py"):
            raise RuntimeError(f"Conflicting recursion dependency already loaded: {name}")
    previous = sys.path[:]
    try:
        sys.path[:0] = [str(root / d) for d in DIRECTORIES.values()]
        modules = {name: importlib.import_module(name) for name in PINS}
    finally:
        sys.path[:] = previous
    return SimpleNamespace(
        CompactCBlock=modules["fivepoint_runtime"].CompactCBlock,
        CoefficientStore=modules["fivepoint_runtime"].CoefficientStore,
        elliptic=modules["fivepoint_elliptic"], qw=modules["fivepoint_qw"],
        jets=modules["fivepoint_elliptic_jets"], atlas=modules["fivepoint_elliptic_atlas"],
        integration=modules["fivepoint_ope_integrator"], forest=modules["fivepoint_collar_forest"],
        manifest=manifest)
