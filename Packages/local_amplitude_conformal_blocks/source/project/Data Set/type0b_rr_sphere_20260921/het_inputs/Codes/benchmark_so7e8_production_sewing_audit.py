"""Level-zero component diagnostic of the retained internal-R sewing ansatz.

This is NOT a correlator crossing test or a physical counterexample: the
component-to-physical map has not been established. It uses only ground-level
PBW blocks, the same oriented signs as
build_fixed_p_two_ramond_kernel, and the unmodified BRY constants. In
particular it does not refit a sign or a normalization to achieve crossing.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

from so7e8_ramond_fourpoint import (
    ramond_sld_chiral_branch,
    antiholomorphic_ramond_sld_chiral_branch,
)
from so7e8_ramond_sphere_integrand import internal_ramond_state_projectors
from spin23_super_liouville_data import rr_ns_chiral_structure_constant


MOMENTA = (.02 + .22j, .03 + .23j, .04 + .24j, .09 + .69j)


def primary_ground_counterexample(internal_momentum=.37, momenta=MOMENTA):
    """Compare a primary-reduction fixture, before PCO or physical U_eta.

    All four external chiral R ground bits are zero and both NS words are
    empty. The published normalized even representative has leading
    coefficient one for every structure-sign pair (Suchanek eq. 49).
    Applying the legacy routing to this raw fixture gives zero for each
    sign pair separately. The separately recorded published primary ground
    product is NOT the same observable without an established physical-state
    and dual-frame map. Their difference is not evidence of crossing failure
    and does not identify a physical sewing error.
    """
    rows = []
    legacy = 0j
    even_representative = 0j
    for sl, sr in product((-1, 1), repeat=2):
        options = dict(
            external_momenta=momenta, maximum_twice_level=0,
            left_structure_sign=sl, right_structure_sign=-sr,
        )
        pieces = []
        for forms in product((0, 1), repeat=2):
            hol = ramond_sld_chiral_branch(
                internal_momentum, form_parities=forms, **options).coefficients[0]
            for projection in internal_ramond_state_projectors(
                sl, sr, forms, (0, 0), (0, 0), (0, 0), (0, 0),
            ):
                anti = antiholomorphic_ramond_sld_chiral_branch(
                    internal_momentum,
                    form_parities=projection.antiholomorphic_form_parities,
                    **options).coefficients[0]
                pieces.append(dict(
                    label=projection.internal_local_field,
                    holomorphic_forms=forms,
                    antiholomorphic_forms=projection.antiholomorphic_form_parities,
                    holomorphic_coefficient=hol, antiholomorphic_coefficient=anti,
                    projector_phase=projection.phase,
                    contribution=projection.phase * hol * anti,
                ))
        coefficient = (
            rr_ns_chiral_structure_constant(
                momenta[3], internal_momentum, momenta[2], structure_sign=sl)
            * rr_ns_chiral_structure_constant(
                -internal_momentum, momenta[0], momenta[1], structure_sign=sr) / 4
        )
        reduced = sum(piece["contribution"] for piece in pieces)
        rows.append(dict(
            oriented_signs=(sl, sr), block_signs=(sl, -sr),
            structure_product=coefficient,
            legacy_leading_coefficient=reduced,
            normalized_even_representative_leading_coefficient=1., pieces=pieces,
        ))
        legacy += coefficient * reduced
        even_representative += coefficient
    return dict(
        internal_momentum=internal_momentum, momenta=momenta, rows=rows,
        legacy_primary_fixture=legacy,
        normalized_even_primary_anchor=even_representative,
        relative_anchor_defect=abs(legacy-even_representative)/abs(even_representative),
        physical_heterotic_amplitude=False,
    )


def serialize(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


def run():
    root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__), root / "Codes/so7e8_ramond_sphere_integrand.py",
        root / "Codes/so7e8_ramond_fourpoint.py",
        root / "Codes/so7e8_human_conventions.py",
        root / "Codes/spin23_super_liouville_data.py",
    )
    return dict(
        status="raw component cancellation only; NOT a correlator comparison or physical counterexample",
        reference="https://arxiv.org/pdf/1012.2974, equations (37), (49)",
        maximum_ramond_level=0,
        quadrature_used=False, elliptic_reexpansion_used=False,
        production_modified=False,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources},
        cases=[primary_ground_counterexample(p) for p in (.37, .70)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialize(report), indent=2) + "\n")
    for case in report["cases"]:
        print(f"P={case['internal_momentum']}: raw fixture={case['legacy_primary_fixture']}, "
              f"separate primary product={case['normalized_even_primary_anchor']}; "
              "these are not established as the same physical observable")
