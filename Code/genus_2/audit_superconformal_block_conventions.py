#!/usr/bin/env python3
"""Compare literature block phases with the saved Human-Note conventions.

Read-only imports of the existing kernels. The BRY reference independently
implements its sphere c-recursion; the Ramond check is an exact ground-state
calculation of the outgoing-bra conversion, not a proposed genus-two kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
for relative in ("Code", "Code/c_Recursion", "Code/double_virasoro/nsrr"):
    sys.path.insert(0, str(ROOT / relative))

from superconformal_blocks import HighPrecisionNSSphereFourPointBlock
from ramond_pbw_generalized_ward import RamondPBWModule

REFERENCE = ROOT / (
    "Data Set/type0b_fivepoint_imaginary_elliptic_20260913/"
    "bry_human_dictionary/audit.py"
)
spec = importlib.util.spec_from_file_location("literal_bry_convention_reference", REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def pair(value):
    return [mp.nstr(mp.re(value), 35), mp.nstr(mp.im(value), 35)]


def ns_comparison():
    """All four star patterns, real/complex weights, through level four."""
    c, h = mp.mpf("14.19870372000744"), mp.mpf(".73")
    fixtures = (
        tuple(map(mp.mpc, (".37", ".61", ".48", ".29"))),
        (mp.mpc(".37", ".08"), mp.mpc(".61", "-.04"),
         mp.mpc(".48", ".03"), mp.mpc(".29", "-.02")),
    )
    z = mp.mpc(".23", ".17")
    rows, coefficient_count, maximum = [], 0, mp.mpf(0)
    for fixture, weights in enumerate(fixtures):
        for star2, star3 in itertools.product((0, 1), repeat=2):
            bry = reference.LiteralBRY(c, weights, h, star2, star3)
            saved = HighPrecisionNSSphereFourPointBlock(
                c=c, h1=weights[0], h2=weights[1], h3=weights[2], h4=weights[3],
                internal_weight=h, star2=star2, star3=star3, working_precision=60,
            )
            coefficients = []
            for k in range(9):
                # Derived from BRY (3.14),(3.18) and the saved signed seed.
                phase = (-1) ** ((k % 2) * (1 + star3 + star2 * star3))
                actual, expected = saved.coefficient(k), bry.coefficient(k)
                error = abs(phase * actual - expected) / max(1, abs(actual), abs(expected))
                maximum = max(maximum, error)
                coefficients.append((actual, expected))
                coefficient_count += 1
            for parity in (0, 1):
                phase = (-1) ** (parity * (1 + star3 + star2 * star3))
                primary = mp.exp((h - weights[0] - weights[1] - mp.mpf(star2)/2) * mp.log(z))
                actual = primary * mp.fsum(coefficients[k][0] * z**(mp.mpf(k)/2)
                                          for k in range(parity, 9, 2))
                expected = primary * mp.fsum(coefficients[k][1] * z**(mp.mpf(k)/2)
                                            for k in range(parity, 9, 2))
                ratio = actual / expected
                rows.append({
                    "fixture": fixture, "external_weights_origin_to_infinity": list(map(pair, weights)),
                    "star2": star2, "star3": star3, "internal_parity": parity,
                    "multiply_saved_block_to_get_BRY": phase,
                    "saved_block": pair(actual), "BRY_block": pair(expected),
                    "raw_norm_ratio": mp.nstr(abs(ratio), 35),
                    "raw_phase_degrees": mp.nstr(mp.arg(ratio)*180/mp.pi, 35),
                    "converted_complex_relative_error": mp.nstr(abs(phase*ratio-1), 12),
                })
    assert maximum < mp.mpf("1e-45"), maximum
    return {
        "central_charge": pair(c), "internal_weight": pair(h), "z": pair(z),
        "maximum_level": 4, "working_precision": 60,
        "coefficient_comparisons": coefficient_count,
        "maximum_scaled_coefficient_error": mp.nstr(maximum, 12),
        "phase_formula": "F_BRY^p=(-1)^(p*(1+star3+star2*star3))*H_saved^p",
        "scope": "Sphere four-point, bottom external states at zero and infinity; common primary powers and log branch.",
        "reference_endpoint": "Includes rs=2m: the first null pole contributes at its own level; BRY's printed strict inequality omits this endpoint.",
        "rows": rows,
    }


def ramond_outgoing_bra():
    """Suchanek section 3.1 ground coefficient versus a mixed pairing."""
    momentum = sp.Symbol("P", positive=True)
    beta = sp.I*momentum/sp.sqrt(2)
    a = sp.I*beta*(1-sp.I)/sp.sqrt(2)
    norm_u = -beta**2  # u=G_0 w^+; positive on this physical momentum line.
    bilinear_odd_metric = sp.simplify(norm_u/a**2)
    literature = sp.simplify(sp.conjugate(a)*a/norm_u)
    mixed = sp.simplify(a*a/norm_u)
    outgoing_conversion = sp.simplify(sp.conjugate(a)/a)
    saved_metric = RamondPBWModule.ground_pairing(1, 1)
    corrected = sp.simplify(outgoing_conversion/bilinear_odd_metric)
    assert bilinear_odd_metric == saved_metric == sp.I
    assert literature == corrected == 1
    assert mixed == -sp.I
    return {
        "momentum_domain": "P>0, beta=i*P/sqrt(2); analytically continue only after fixing the state/dual convention.",
        "ground_state": "u=G_0 w^+=a w^-; a=i*beta*exp(-i*pi/4)",
        "norm_u": str(sp.simplify(norm_u)),
        "bilinear_metric_w_minus": str(saved_metric),
        "literature_odd_ground_block": str(literature),
        "unit_three_forms_with_bilinear_metric": str(mixed),
        "mixed_over_literature_norm": 1,
        "mixed_over_literature_phase_degrees": -90,
        "outgoing_odd_ground_bra_conversion": str(outgoing_conversion),
        "consistent_bilinear_result": str(corrected),
        "scope": "NR/RN sphere ground sewing only. No all-descendant or genus-two adapter is inferred.",
    }


def ns_full_vertex_phase():
    h, t = sp.symbols("h C_tilde_BRY", real=True)
    raw_norm = -(2*h)**2
    phased_norm = sp.expand(sp.I**2 * raw_norm)
    physical_coefficient = sp.expand(-1 * (sp.I*t)**2)
    assert phased_norm == (2*h)**2
    assert physical_coefficient == t**2
    return {
        "raw_graded_odd_tensor_bilinear_norm": str(raw_norm),
        "phase_i_tensor_bilinear_norm": str(phased_norm),
        "saved_pants_coefficient": "C_HN^(1)=i*C_tilde_BRY (the opposite sign gives the same two-pants product)",
        "explicit_odd_sewing_sign_times_two_vertices": str(physical_coefficient),
        "scope": "All-NS genus-two bilinear convention. This does not supply the NSRR contraction.",
    }


def free_superfield_complex_comparison():
    """Compare a chiral free SC block with one scalar times one Majorana."""
    import numpy as np
    from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
    from physical_free_plumbing_resummation import theta_charged_boson_resummation
    from fixed_spin_free_plumbing import charged_frame, fixed_spin_chiral_partition

    p0, p1 = sp.Rational(31, 100), sp.Rational(47, 100)
    c, h = sp.Rational(3, 2), (p1-p0)**2/2
    components = {}
    for f in (0, 1):
        oracle = HumanNSRRThetaOracle(
            central_charge=c, h_ns=h, beta_r1=sp.I*p1/sp.sqrt(2),
            beta_r2=sp.I*p0/sp.sqrt(2), form_parity=f, primary_parity=0, etas=(1, 1),
        )
        components[f] = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                         for e in level_triples(6)}
    config = json.loads((ROOT/"Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json").read_text())
    original = tuple(complex(z) for z in next(p for p in config["points"]
                     if p["point_id"] == "generic_04")["source"]["q_values"])
    rows = []
    for scale in (.02, .1):
        q = tuple(scale*z for z in original)
        qs = q[::-1]
        primary_weights = (h, c/24+p1**2/2, c/24+p0**2/2)
        primary = math.prod(qs[k]**float(weight) for k, weight in enumerate(primary_weights))
        scalar = theta_charged_boson_resummation(
            q, alpha_zero=float(p0), alpha_one=-float(p1), max_mode=32)
        frame = charged_frame(q, max_mode=32)
        # Use the charge chart throughout, so this call has B=0. Its branch
        # is fixed by radial continuation, independently of the SC block.
        fermion = fixed_spin_chiral_partition(
            q, frame.omega_charge, ((1, 1), (0, 0)),
            period_branch=np.zeros((2, 2), dtype=int), max_mode=32,
        )
        expected = scalar.chiral_value*fermion["majorana_chiral"]
        for order in (2, 3):
            blocks = {}
            for f in (0, 1):
                blocks[f] = primary*sum(
                    math.sqrt(2)*complex(coefficient)*math.prod(qs[k]**(e[k]/2) for k in range(3))
                    for e, vector in components[f].items() if sum(e) <= 2*order
                    for p, coefficient in enumerate(vector) if not ((p >> 1) & 1)
                )
            row = {"q_scale": scale, "total_PBW_order": order, "free_chiral": pair(expected),
                   "primary_weights_human_slots": list(map(str, primary_weights)),
                   "primary_prefactor": pair(primary),
                   "definition": "chiral_value=P*F; F is the descendant-only Human-Note block",
                   "descendant_blocks": {str(f): pair(blocks[f]/primary) for f in (0, 1)}}
            for name, value in (("even_projected", blocks[0]),
                                ("legacy_combination", (blocks[0]+1j*blocks[1])/2)):
                ratio = value/expected
                row[name] = {
                    "chiral_value": pair(value), "norm_ratio": abs(ratio),
                    "phase_degrees": float(mp.arg(ratio)*180/mp.pi),
                    "complex_relative_error": abs(ratio-1),
                }
            rows.append(row)
    return {
        "theory": "c=3/2, one charged free scalar and one Majorana",
        "charges_geometry": [float(p0), -float(p1), float(p1-p0)],
        "comparison_frame": "charge period, characteristic [11|00], q^L0 propagation",
        "branch": "independently continued Majorana with positive Ramond degeneration coefficient",
        "scope": "This free-field calibration does not determine an interacting replacement kernel.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Data Set/superconformal_block_conventions_20260915/conventions.json")
    args = parser.parse_args()
    with mp.workdps(60):
        report = {
            "schema": "superconformal-block-literature-conventions-v1",
            "status": "NS_phase_dictionary_verified_Ramond_bra_mismatch_exhibited",
            "sources": {
                "BRY": "https://arxiv.org/html/2201.05621",
                "Suchanek": "https://arxiv.org/pdf/1012.2974",
                "Hadasz_Jaskolski_Suchanek": "https://arxiv.org/pdf/0810.1203",
                "Belavin_Geiko": "https://arxiv.org/pdf/1806.09563",
            },
            "ns": ns_comparison(),
            "ramond_ground_bra": ramond_outgoing_bra(),
            "all_NS_full_vertex": ns_full_vertex_phase(),
            "free_superfield_complex": free_superfield_complex_comparison(),
            "production_kernels_modified": False,
            "interacting_Ramond_sewing_certified": False,
        }
    paths = (REFERENCE, Path(__file__), ROOT/"Code/c_Recursion/superconformal_blocks.py",
             ROOT/"Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py",
             ROOT/"Code/double_virasoro/nsrr/nsrr_genus2_block.py",
             ROOT/"Code/genus_2/physical_nsrr_sewing.py",
             ROOT/"Code/genus_2/fixed_spin_free_plumbing.py",
             ROOT/"Code/genus_2/physical_free_plumbing_resummation.py",
             ROOT/"Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json")
    report["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({
        "output": str(args.output),
        "NS_coefficient_comparisons": report["ns"]["coefficient_comparisons"],
        "NS_maximum_error": report["ns"]["maximum_scaled_coefficient_error"],
        "Ramond_ground": report["ramond_ground_bra"],
        "free_complex": report["free_superfield_complex"]["rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
