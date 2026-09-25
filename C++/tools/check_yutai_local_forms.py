#!/usr/bin/env python3
"""Compare ordered NS--R--R forms with Yutai's 0B sphere and torus code.

The sphere helper is checked only in the combinations used by its four-point
blocks: an internal NS descendant with an external Ramond ground state, or
an external NS primary/superpartner with an internal Ramond descendant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Packages/local_amplitude_conformal_blocks/source/project/Code"
SPHERE = (ROOT / "Packages/local_amplitude_conformal_blocks/source/project/"
          "Data Set/type0b_rr_sphere_20260921/het_inputs/Codes")
sys.path[:0] = [str(CODE / "double_virasoro/nsrr"),
                str(CODE / "type0b_rr_genus1"), str(SPHERE)]
from ramond_pbw_generalized_ward import GeneralizedNRRWard  # noqa: E402
from numeric_nrr import NRR  # noqa: E402
from literature_component_blocks import ThreePoint, sign_form, _ns  # noqa: E402


def ward(*, p: int, f: int, eta: int, h: tuple[sp.Expr, ...],
         beta: tuple[sp.Expr, sp.Expr], c: sp.Expr) -> GeneralizedNRRWard:
    return GeneralizedNRRWard(
        p_phi=p, form_parity=f, eta=eta, h_ns=h[0],
        h_second=h[1], h_third=h[2],
        beta_second=beta[0], beta_third=beta[1], central_charge=c,
    )


def torus_check() -> tuple[int, float]:
    h = (sp.Rational(7, 11), sp.Rational(13, 17), sp.Rational(19, 23))
    beta = (sp.I * sp.Rational(2, 7), sp.I * sp.Rational(3, 11))
    c = sp.Rational(27, 2)
    words = (
        ((), (), ()),
        ((("G", -sp.Rational(1, 2)),), (), ()),
        ((), (("G", -1),), ()),
        ((), (), (("G", -1),)),
        ((("L", -1),), (), ()),
        ((), (("L", -1),), ()),
        ((), (), (("L", -1),)),
    )
    count = 0
    maximum = 0.0
    for p in (0, 1):
        for f in (0, 1):
            for eta in (-1, 1):
                production = NRR(
                    *(float(x) for x in h), complex(beta[0]), complex(beta[1]),
                    float(c), eta=eta, p_phi=p, form_parity=f,
                )
                reference = ward(p=p, f=f, eta=eta, h=h, beta=beta, c=c)
                for w1, w2, w3 in words:
                    for g2 in (0, 1):
                        for g3 in (0, 1):
                            got = production.value(w1, w2, g2, w3, g3)
                            want = complex(sp.N(reference.value(
                                w1, w2, g2, w3, g3), 25))
                            maximum = max(maximum, float(abs(got - want)))
                            count += 1
    return count, maximum


def sphere_check() -> tuple[int, float]:
    pn, pm, pk = (sp.Rational(3, 10), sp.Rational(2, 7),
                  sp.Rational(5, 13))
    h_ns = lambda x: sp.Rational(1, 16) + x * x / 2
    h = (h_ns(pn), h_ns(pm) + sp.Rational(1, 16),
         h_ns(pk) + sp.Rational(1, 16))
    beta = (-sp.I * pm / sp.sqrt(2), -sp.I * pk / sp.sqrt(2))
    production = ThreePoint(
        "NS", "R", "R", float(pn), float(pm), float(pk))
    ns_words = ((), (("G", -1),), (("L", -2),))
    r_words = ((), (("G", -2),), (("L", -2),))
    count = 0
    maximum = 0.0
    for eta in (-1, 1):
        for ns in ns_words:
            for r in r_words:
                if ns == (("L", -2),) and r:
                    continue  # This combination is not used in the sphere blocks.
                for g2 in (0, 1):
                    for g3 in (0, 1):
                        f = (sum(k == "G" for k, _ in ns) + g2
                             + sum(k == "G" for k, _ in r) + g3) % 2
                        reference = ward(p=0, f=f, eta=eta, h=h,
                                         beta=beta, c=sp.Integer(3))
                        translated_ns = tuple(
                            (k, sp.Rational(n, 2)) for k, n in ns)
                        translated_r = tuple(
                            (k, sp.Rational(n, 2)) for k, n in r)
                        got = sign_form(
                            production.nr((ns, 0), g2, (r, g3)), "NR", f, eta)
                        want = complex(sp.N(reference.value(
                            translated_ns, (), g2, translated_r, g3), 25))
                        maximum = max(maximum, float(abs(got - want)))
                        count += 1
    return count, maximum


def sphere_reversed_check() -> tuple[int, float]:
    """Check the R--R--NS vertex via the ordered NS--R--R Ward form."""
    pr, pm, pn = (sp.Rational(5, 13), sp.Rational(2, 7),
                  sp.Rational(3, 10))
    h_ns = lambda x: sp.Rational(1, 16) + x * x / 2
    h = (h_ns(pn), h_ns(pm) + sp.Rational(1, 16),
         h_ns(pr) + sp.Rational(1, 16))
    beta = (-sp.I * pm / sp.sqrt(2), -sp.I * pr / sp.sqrt(2))
    production = ThreePoint(
        "R", "R", "NS", float(pr), float(pm), float(pn))
    words = (
        (), (("G", -2),), (("L", -2),), (("G", -4),),
        (("L", -4),), (("L", -2), ("G", -2)),
        (("L", -2), ("L", -2)),
    )
    count = 0
    maximum = 0.0
    for word in words:
        odd = sum(kind == "G" for kind, _ in word) % 2
        translated = tuple((kind, sp.Rational(mode, 2))
                           for kind, mode in word)
        for ns_star in (0, 1):
            ns_word = ((("G", -sp.Rational(1, 2)),)
                       if ns_star else ())
            for ground in (0, 1):
                for middle_ground in (0, 1):
                    f = (ns_star + ground + middle_ground + odd) % 2
                    for eta in (-1, 1):
                        reference = ward(
                            p=0, f=f, eta=eta, h=h,
                            beta=beta, c=sp.Integer(3))
                        got = sign_form(
                            production.rn(
                                (word, ground), middle_ground, _ns(ns_star)),
                            "RN", f, eta)
                        ordered = complex(sp.N(reference.value(
                            ns_word, (), middle_ground, translated, ground),
                            25))
                        # Inversion exchanges the NS and outer Ramond slots.
                        # The phases of the NS superpartner and Ramond G
                        # descendant, and their graded exchange, give this.
                        phase = (1j ** (odd - ns_star)
                                 * (-1) ** (
                                     ground * (ns_star + odd)
                                     + ns_star * odd))
                        maximum = max(maximum, float(abs(got - phase * ordered)))
                        count += 1
    return count, maximum


def main() -> None:
    torus_count, torus_error = torus_check()
    sphere_count, sphere_error = sphere_check()
    reversed_count, reversed_error = sphere_reversed_check()
    result = {
        "torus_production_ward_components": torus_count,
        "torus_maximum_absolute_error": torus_error,
        "sphere_production_support_components": sphere_count,
        "sphere_maximum_absolute_error": sphere_error,
        "sphere_reversed_support_components": reversed_count,
        "sphere_reversed_maximum_absolute_error": reversed_error,
    }
    print(json.dumps(result, indent=2))
    if max(torus_error, sphere_error, reversed_error) >= 1e-12:
        raise AssertionError("Yutai local form convention mismatch")


if __name__ == "__main__":
    main()
