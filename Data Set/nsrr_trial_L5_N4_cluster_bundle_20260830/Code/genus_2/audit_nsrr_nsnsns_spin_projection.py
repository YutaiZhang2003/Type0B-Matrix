#!/usr/bin/env python3
"""Compare the literal Human-Note spin sum with the runtime star character.

This verifies the corrected projection at ground level and records the old
star-character result separately. It does not recompute the partition sum.
"""
import argparse
from pathlib import Path

import sympy as sp

import nsrr_nsnsns_theta_omega_scan as scan
from nsrr_genus2_block import HumanNSRRThetaOracle
from theta_star_algebra import fwht, star_spectrum


def encode(value):
    value = complex(value)
    return [value.real, value.imag]


def audit():
    b = sp.Rational(7, 5)
    background = b + 1/b
    c = sp.Rational(3, 2) + 3*background**2
    momenta = (sp.Rational(21, 100), sp.Rational(37, 100), sp.Rational(52, 100))
    lifts = (1, 1, -1)
    character = scan.spin_character_index(lifts)
    runtime = scan.NSRRDoubleVirasoroTheta(b=float(b),
        physical_momenta=tuple(float(p) for p in momenta), cutoff=1,
        completion="pbw_diagnostic")
    rows = []
    for f in (0, 1):
        for left, right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            oracle = HumanNSRRThetaOracle(central_charge=c,
                h_ns=background**2/8+momenta[0]**2/2,
                beta_r1=sp.I*momenta[1]/sp.sqrt(2), beta_r2=sp.I*momenta[2]/sp.sqrt(2),
                form_parity=f, primary_parity=0, etas=(left, right))
            # These components ALREADY contain the theta quadratic sign,
            # as in Human Notes/SCblock.tex, equation labelled Rblock.
            components = oracle.coefficient_components(0, 0, 0)
            literal = fwht(components)[character]
            spectral = star_spectrum(components)[character]
            actual = runtime.physical_series(f, left, right, character).get((0, 0, 0), 0)
            if abs(actual-literal) > 1e-10:
                raise ArithmeticError("runtime disagrees with the literal Human-Note spin sum")
            rows.append({"form_parity": f, "eta_left": left, "eta_right": right,
                         "components_including_theta_sign": [encode(x) for x in components],
                         "literal_fixed_lift_sum": encode(literal),
                         "star_character": encode(spectral), "runtime_ground": encode(actual),
                         "runtime_minus_literal_absolute": float(abs(actual-literal)),
                         "runtime_minus_star_absolute": float(abs(actual-spectral))})
    return {"schema": "nsrr-nsnsns-spin-projection-ground-audit-v2",
            "numerical_kernel_fingerprint": scan.fingerprint(), "b": float(b),
            "physical_momenta": [float(x) for x in momenta], "lifts_in_runtime_slot_order": list(lifts),
            "star_character_index": character, "rows": rows,
            "literal_projection": "sum_p B_p product_e eta_e^p_e, with the theta quadratic sign already included in B_p",
            "star_projection": "sum_p (-1)^(p0*p1+p0*p2+p1*p2) B_p product_e eta_e^p_e",
            "conclusion": "The runtime now agrees with the literal Human-Note fixed-lift PBW block. The old star-character quotient is a different observable.",
            "scope": "Opposite-HJS-sign nullspace information is supplied explicitly by the independent PBW oracle. This chiral check alone does not certify nonchiral Ramond sewing or modular equality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    scan.write_json(args.output, result)
    for row in result["rows"]:
        print({k: v for k, v in row.items() if k != "components_including_theta_sign"})


if __name__ == "__main__":
    main()
