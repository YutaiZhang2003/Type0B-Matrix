"""Verified access to the frozen Het super-Liouville coefficient banks.

Only the CFT layer is imported. Heterotic amplitudes and GSO tensors are
not input data for the Type 0B calculation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "Data Set/type0b_rr_sphere_20260921/het_inputs"
RELEASE = "706bef6ef7b6306210b29c7c05a2ef5d9817e7bc55da06127918a9bb116d867a"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def initialize(inputs=DEFAULT_INPUTS):
    inputs = Path(inputs).resolve()
    marker = json.loads((inputs / ".standalone_workspace.json").read_text())
    if marker["bundle_release_id"] != RELEASE:
        raise ValueError("unrecognized frozen release")
    manifest = json.loads((inputs / "native/manifest.json").read_text())
    claimed = manifest.pop("manifest_sha256")
    if digest(manifest) != claimed:
        raise ValueError("native manifest checksum mismatch")
    manifest["manifest_sha256"] = claimed
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((inputs / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen scientific source changed: " + relative)
    sys.path.insert(0, str(inputs / "Codes"))
    return inputs, manifest


INPUTS, MANIFEST = initialize()

from literature_component_blocks import ramond_state, ramond_vertex, crossing_phase, SECTORS
from literature_self_dual_correlator import SelfDualConstants
from so7e8_literature_campaign import FrozenBlocks, chiral_components, sign_pairs, unpairs, pairs
from so7e8_literature_atlas import BlockGeometry, block_grid_values


def generic_integrator():
    """Load frozen polynomial algebra/sewing, never its heterotic driver."""
    path = INPUTS / "reference/moduli/source/integrate_so7e8_literature_banks.py"
    spec = importlib.util.spec_from_file_location("_frozen_moduli_algebra", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALGEBRA = generic_integrator()


def records(energy, family):
    result = []
    for index, task in enumerate(MANIFEST["bank_tasks"]):
        if (task["energy"], task["family"]) != (energy, family):
            continue
        path = INPUTS / "native/banks" / f"{index:04d}.json"
        record = json.loads(path.read_text())
        if (record["manifest_sha256"] != MANIFEST["manifest_sha256"]
                or record["task"] != task
                or record["payload_sha256"] != digest(record["payload"])):
            raise ValueError(f"bank integrity failure: {path}")
        payload = record["payload"]
        if payload["order"] < 5 or payload["arithmetic"]["decimal_digits"] < 60:
            raise ValueError("insufficient bank order or assembly precision")
        result.append((task, payload))
    if len(result) != 32 or {t["momentum_index"] for t, _ in result} != set(range(32)):
        raise ValueError("exactly 32 distinct momentum nodes are required")
    return result


def pair(task, payload, order):
    if order < 0 or order > payload["order"]:
        raise ValueError("requested order is not stored")
    p = tuple(unpairs(task["momenta"]))
    table = unpairs(payload["coefficients"])[..., :2*order+1]
    return (FrozenBlocks(task["family"], p, task["P"], table[0]),
            FrozenBlocks(task["family"], tuple(x.conjugate() for x in p),
                         task["P"], table[1]))


def correlators(rows, points, external, order=5, constants=None):
    """Full physical Liouville components with the stored external factor stripped."""
    constants = constants or SelfDualConstants(70)
    geometry = BlockGeometry.build(points)
    result = {e: np.zeros(len(geometry.points), complex) for e in external}
    for task, payload in rows:
        hol, dual = pair(task, payload, order)
        hv = block_grid_values(hol, geometry)
        av = block_grid_values(dual, geometry).conjugate()
        densities = ALGEBRA.density_map(constants, hol)
        for e in external:
            for hi, ai, dp, sl, sr, c in ALGEBRA.sewing_terms(hol.family, e):
                result[e] += task["dP_weight"]/np.pi*c*densities[dp, sl, sr]*hv[hi]*av[ai]
    return result


def source_identity():
    return dict(release=RELEASE, manifest_sha256=MANIFEST["manifest_sha256"],
                native_banks=864, quadrature="dP/pi, applied once",
                frozen_source_files=len(MANIFEST["source_sha256"]))
