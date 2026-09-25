"""Bounded level-five INTERMEDIATE-R benchmark in the human bilinear frame.

The reference uses exact rational-complex momenta, exact Ramond PBW Gram
inversion, and literal human ground tensors. It does not use the HJS-to-human
rotation or the double-Virasoro generator. The finite branch and full PBW
calculations still share the polynomial Ward algebra, so this is not an
independent proof of those Ward identities or of nonchiral sewing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import G
from ramond_algebra.nrr_three_point_tensor import rr_three_point_from_ground_tensor
from ramond_algebra.ramond_sca import gram_matrix, ground_state
from ramond_sphere_uniformization import elliptically_prefactor_internal_ramond_series
from so7e8_human_conventions import ramond_form_rotation
from so7e8_internal_ramond_double_virasoro import (
    ramond_branch_numbers, self_dual_internal_ramond_double_virasoro_coefficients,
)


SIGNS = (1, -1)
SIGN_PAIRS = tuple(product(SIGNS, repeat=2))
CHIRALITIES = ("holomorphic", "antiholomorphic")
MOMENTA = tuple(sp.Rational(r, 100) + sp.I*sp.Rational(i, 100)
               for r, i in ((2, 22), (3, 23), (4, 24), (9, 69)))
MODULI = (.35+.1j, .5+.1j, .65+.1j)
BASE_CONTOUR = dict(radius=.12, check_radius=.15, samples=32, tolerance=1e-8)
REFINED_CONTOUR = dict(radius=.10, check_radius=.13, samples=40, tolerance=1e-8)
REPO = Path(__file__).resolve().parents[1]


def phase(chirality):
    if chirality not in CHIRALITIES:
        raise ValueError("invalid chirality")
    return (1 + (sp.I if chirality == "holomorphic" else -sp.I))/sp.sqrt(2)


def literal_tensor(p_infinity, p_zero, sign, chirality):
    """SCblock.tex: unit component seeds; rho_odd=rho_+-+i*s*rho_-+."""
    if sign not in SIGNS:
        raise ValueError("invalid structure sign")
    unit = sp.I if chirality == "holomorphic" else -sp.I
    alpha = phase(chirality)
    left = sp.diag(1, alpha*sp.I*p_infinity/sp.sqrt(2))
    right = sp.diag(1, alpha*sp.I*p_zero/sp.sqrt(2))
    return (left*sp.Matrix([[1, 1], [unit*sign, sign]])*right).applyfunc(sp.expand)


def form_parities(stars, grounds, internal_parity):
    return (internal_parity ^ stars[1] ^ grounds[1],
            internal_parity ^ stars[0] ^ grounds[0])


class ExactRamondReference:
    """Complete homogeneous parity subspaces, not a subset of descendants.

    In the polynomial basis G0|+>=|->. The human basis is S times this
    basis, S_A=(exp(+-i*pi/4)*beta)^(-ground_parity(A)). Thus G_H=S G_P S,
    with transpose/bilinear sewing, never adjoint or momentum conjugation.
    All algebra and inversion here is exact; evalf is only for comparison.
    """

    def __init__(self, internal_momentum=sp.Rational(37, 100), momenta=MOMENTA):
        self.p = sp.sympify(internal_momentum)
        self.momenta = tuple(map(sp.sympify, momenta))
        if self.p == 0 or self.momenta[0] == 0 or self.momenta[3] == 0:
            raise ValueError("zero Ramond momentum requires a separate limit")
        if any(x.has(sp.Float) for x in (self.p, *self.momenta)):
            raise ValueError("the exact reference requires exact input momenta")
        self.c = sp.Rational(27, 2)
        self.h = self.c/24+self.p**2/2
        self.weights = tuple((self.c/24 if i in (0, 3) else sp.Rational(1, 2))+p**2/2
                             for i, p in enumerate(self.momenta))

    @lru_cache(None)
    def gram(self, level, internal_parity, chirality):
        if not isinstance(level, int) or level < 0 or internal_parity not in (0, 1):
            raise ValueError("integer Ramond level and parity bit required")
        basis, polynomial = gram_matrix(2*level, h=self.h, c=self.c, parity=internal_parity)
        scale = sp.diag(*[(phase(chirality)*sp.I*self.p/sp.sqrt(2))**(-s.ground_parity)
                         for s in basis])
        human = (scale*polynomial*scale).applyfunc(sp.expand)
        inverse = human.inv(method="DM")
        assert not human.has(sp.Float) and not inverse.has(sp.Float)
        return tuple(basis), scale, human, inverse

    @lru_cache(None)
    def vector(self, level, internal_parity, chirality, side, star, ground, sign):
        basis, scale, _, _ = self.gram(level, internal_parity, chirality)
        word = (G(Fraction(-1, 2)),) if star else ()
        if side == "left":
            external_index, middle_index = 3, 2
            tensor = literal_tensor(self.momenta[3], self.p, sign, chirality)
        elif side == "right":
            external_index, middle_index = 0, 1
            tensor = literal_tensor(self.p, self.momenta[0], sign, chirality)
        else:
            raise ValueError("side must be left or right")
        divisor = (phase(chirality)*sp.I*self.momenta[external_index]/sp.sqrt(2))**ground
        values = []
        for i, state in enumerate(basis):
            infinity, zero = ((ground_state(ground), state) if side == "left"
                              else (state, ground_state(ground)))
            hi, hz = ((self.weights[3], self.h) if side == "left"
                      else (self.h, self.weights[0]))
            value = rr_three_point_from_ground_tensor(
                infinity, word, zero, h_infinity=hi, h_ns=self.weights[middle_index],
                h_zero=hz, c=self.c, ground_tensor=tensor)
            values.append(sp.expand(value*scale[i, i]/divisor))
        return sp.ImmutableMatrix(values)

    def coefficients(self, *, level=5, stars=(0, 0), grounds=(0, 0),
                     internal_parity=0, signs=(1, 1), chirality="holomorphic"):
        out = {}
        for n in range(level+1):
            _, _, _, inverse = self.gram(n, internal_parity, chirality)
            left = self.vector(n, internal_parity, chirality, "left", stars[1], grounds[1], signs[0])
            right = self.vector(n, internal_parity, chirality, "right", stars[0], grounds[0], signs[1])
            value = sp.expand((left.T*inverse*right)[0])
            assert not value.has(sp.Float)
            out[2*n] = value
        return out

    def gram_audit(self, level=5):
        rows = []
        for n in range(level+1):
            all_basis, full = gram_matrix(2*n, h=self.h, c=self.c)
            off_diagonal_zero = all(full[i, j] == 0 for i, a in enumerate(all_basis)
                                    for j, b in enumerate(all_basis) if a.parity != b.parity)
            for parity, chirality in product((0, 1), CHIRALITIES):
                basis, _, human, inverse = self.gram(n, parity, chirality)
                residual = (human*inverse-sp.eye(len(basis))).applyfunc(sp.expand)
                rows.append(dict(level=n, internal_parity=parity, chirality=chirality,
                    parity_dimension=len(basis), full_dimension=len(all_basis),
                    inverse_residual_exact_zero=residual == sp.zeros(len(basis)),
                    off_parity_gram_exact_zero=off_diagonal_zero,
                    human_gram_condition_number=float(np.linalg.cond(np.array(human, complex)))))
        return rows


def double_virasoro_human_bank(reference, *, level, stars, grounds, internal_parity,
                              chirality, contour):
    """Evaluate four old sign forms once; transport BOTH endpoint spaces.

    This is the same finite basis map as human_internal_ramond_coefficients;
    the separate regression compares this batching to that public entry point.
    No PBW results enter this side of the comparison.
    """
    results = [self_dual_internal_ramond_double_virasoro_coefficients(
        internal_momentum=complex(reference.p), external_momenta=tuple(map(complex, reference.momenta)),
        maximum_twice_level=2*level, ns_stars=stars, ramond_ground_parities=grounds,
        form_parities=form_parities(stars, grounds, internal_parity), chirality=chirality,
        left_structure_sign=l, right_structure_sign=r, **contour) for l, r in SIGN_PAIRS]
    matrix = ramond_form_rotation(chirality)
    unit = 1j if chirality == "holomorphic" else -1j
    bank = {}
    for l, r in SIGN_PAIRS:
        weights = [unit**grounds[1]*matrix[SIGNS.index(l), SIGNS.index(a)]
                   * matrix[SIGNS.index(r), SIGNS.index(b)] for a, b in SIGN_PAIRS]
        coefficients = {k: sum(w*result.series.coefficients[k] for w, result in zip(weights, results))
                        for k in range(0, 2*level+1, 2)}
        left = literal_tensor(reference.momenta[3], reference.p, l, chirality)
        right = literal_tensor(reference.p, reference.momenta[0], r, chirality)
        bank[l, r] = replace(results[0].series, coefficients=coefficients,
            left_ground_tensor=tuple(map(tuple, left.tolist())),
            right_ground_tensor=tuple(map(tuple, right.tolist())), external_ground_basis="human_bilinear")
    diagnostics = dict(
        largest_old_basis_radius_discrepancy=max(d.absolute_error for x in results for d in x.diagnostics.values()),
        largest_rotated_radius_bound=max(sum(abs(w)*result.diagnostics[k].absolute_error
            for w, result in zip(weights, results)) for k in range(0, 2*level+1, 2)),
        generic_evaluations=sum(x.generic_evaluations for x in results),
        max_eigen_residual=max(x.max_eigen_residual for x in results),
        min_spectral_gap=min(x.min_spectral_gap for x in results),
        full_level_pbw_gram_used_by_generator=any(x.series.gram_condition_numbers for x in results))
    return bank, diagnostics


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def comparison_row(reference, source, *, stars, grounds, parity, signs, chirality, level):
    exact = reference.coefficients(level=level, stars=stars, grounds=grounds,
                                   internal_parity=parity, signs=signs, chirality=chirality)
    numeric = {k: complex(v.evalf(60)) for k, v in exact.items()}
    rows = [dict(level=k//2, exact_pbw=str(exact[k]), pbw=pair(v),
                 double_virasoro=pair(source.coefficients[k]),
                 absolute_error=float(abs(source.coefficients[k]-v)),
                 relative_error=float(abs(source.coefficients[k]-v)/max(abs(v), 1e-30)))
            for k, v in numeric.items()]
    # Propagation of coefficient agreement through the same production H
    # transform, not a separate proof of that algebraic transform.
    h_dv = elliptically_prefactor_internal_ramond_series(source)
    h_pbw = elliptically_prefactor_internal_ramond_series(replace(source, coefficients=numeric))
    values = [dict(z=pair(z), pbw_H_truncated_value=pair(h_pbw.value(z)),
                   double_virasoro_H_truncated_value=pair(h_dv.value(z)),
                   relative_error=abs(h_dv.value(z)-h_pbw.value(z))/max(abs(h_pbw.value(z)), 1e-30))
              for z in MODULI]
    return dict(internal_momentum=str(reference.p), stars=stars, grounds=grounds,
        internal_parity=parity, signs=signs, chirality=chirality,
        form_parities=form_parities(stars, grounds, parity), coefficients=rows,
        elliptic_H_coefficients_pbw=[pair(x) for x in h_pbw.h_coefficients],
        elliptic_H_coefficients_double_virasoro=[pair(x) for x in h_dv.h_coefficients],
        elliptic_values=values,
        passed=all(x["absolute_error"] <= 2e-10*max(1, abs(numeric[2*x["level"]])) for x in rows))


def run(*, level=5, progress=print):
    if level != 5:
        raise ValueError("this benchmark is deliberately bounded to intermediate-R level 5")
    started = time.perf_counter()
    reference = ExactRamondReference()
    rows, banks, diagnostics = [], {}, []
    # Every star pattern x internal parity x chirality x sign pair; all four
    # external ground patterns are covered, not their full Cartesian product.
    grounds_by_stars = {(0, 0): (0, 0), (1, 0): (1, 0), (0, 1): (0, 1), (1, 1): (1, 1)}
    for stars, parity, chirality in product(grounds_by_stars, (0, 1), CHIRALITIES):
        grounds = grounds_by_stars[stars]
        bank, diagnostic = double_virasoro_human_bank(reference, level=level, stars=stars,
            grounds=grounds, internal_parity=parity, chirality=chirality, contour=BASE_CONTOUR)
        banks[stars, parity, chirality] = bank
        diagnostics.append(dict(stars=stars, grounds=grounds, internal_parity=parity,
                                chirality=chirality, **diagnostic))
        batch = [comparison_row(reference, source, stars=stars, grounds=grounds,
                 parity=parity, signs=signs, chirality=chirality, level=level)
                 for signs, source in bank.items()]
        rows.extend(batch)
        progress(f"R level 5: stars={stars}, grounds={grounds}, parity={parity}, {chirality}: "
                 f"max coefficient error={max(x['absolute_error'] for row in batch for x in row['coefficients']):.3e}")

    # An extra momentum and a one-star/unequal-form-parity case: the main
    # balanced star/ground patterns above have f_L=f_R.
    additional = ExactRamondReference(sp.Rational(7, 10))
    bank, diagnostic = double_virasoro_human_bank(additional, level=level, stars=(1, 0),
        grounds=(0, 0), internal_parity=1, chirality="holomorphic", contour=BASE_CONTOUR)
    diagnostics.append(dict(internal_momentum="7/10", stars=(1, 0), grounds=(0, 0),
                            internal_parity=1, chirality="holomorphic", **diagnostic))
    rows.extend(comparison_row(additional, source, stars=(1, 0), grounds=(0, 0), parity=1,
                signs=signs, chirality="holomorphic", level=level) for signs, source in bank.items())
    progress("Second momentum and unequal endpoint form parities checked.")

    refinement = []
    for parity in (0, 1):
        refined, diagnostic = double_virasoro_human_bank(reference, level=level, stars=(1, 1),
            grounds=(1, 1), internal_parity=parity, chirality="holomorphic", contour=REFINED_CONTOUR)
        base = banks[(1, 1), parity, "holomorphic"]
        differences = [abs(source.coefficients[k]-base[signs].coefficients[k])
                       for signs, source in refined.items() for k in source.coefficients]
        refinement.append(dict(internal_parity=parity, stars=(1, 1), grounds=(1, 1),
            maximum_coefficient_change=float(max(differences)), **diagnostic))
        progress(f"Changed both contour radii and node count, parity={parity}: {max(differences):.3e}")

    gram_rows = reference.gram_audit(level)+[dict(internal_momentum="7/10", **r)
                                          for r in additional.gram_audit(level)]
    coefficient_rows = [c for row in rows for c in row["coefficients"]]
    artifacts = [Path(__file__), REPO/"Codes/so7e8_human_conventions.py",
        REPO/"Codes/so7e8_internal_ramond_double_virasoro.py",
        REPO/"Codes/so7e8_mixed_double_virasoro.py", REPO/"Codes/virasoro_sphere_c_recursion.py",
        REPO/"Codes/ramond_algebra/nrr_three_point_tensor.py", REPO/"Codes/ramond_algebra/ramond_sca.py",
        REPO/"Codes/ramond_sphere_uniformization.py",
        REPO.parent/"Type0B-matrix/Human Notes/SCblock.tex"]
    return dict(intermediate_sector="Ramond", maximum_descendant_level=level,
        maximum_twice_level=2*level, b=1, c="27/2", external_order=["R0", "NSz", "NS1", "R_infinity"],
        external_momenta=[str(x) for x in MOMENTA], internal_momenta=["37/100", "7/10"],
        branch_numbers=[str(n) for n in ramond_branch_numbers(2*level)],
        reference_arithmetic="exact rational-complex Gram inverse and exact Ward vectors; 60-digit display evaluation",
        human_ground_gram={"holomorphic": "diag(1,i)", "antiholomorphic": "diag(1,-i)"},
        NS_stars="unrescaled G_-1/2; not an all-primary block with shifted weights",
        base_contour=BASE_CONTOUR, refined_contour=REFINED_CONTOUR,
        case_count=len(rows), coefficient_comparisons=len(coefficient_rows),
        maximum_absolute_coefficient_error=max(x["absolute_error"] for x in coefficient_rows),
        maximum_relative_coefficient_error=max(x["relative_error"] for x in coefficient_rows),
        maximum_relative_H_value_error=max(x["relative_error"] for row in rows for x in row["elliptic_values"]),
        passed=all(row["passed"] for row in rows)
            and all(row["inverse_residual_exact_zero"] and row["off_parity_gram_exact_zero"] for row in gram_rows)
            and all(row["maximum_coefficient_change"] < 2e-10 for row in refinement),
        exact_gram_audit=gram_rows, contour_diagnostics=diagnostics, refinement=refinement, cases=rows,
        common_infrastructure="finite branch forms and PBW oracle share the polynomial Ward/mode algebra",
        nonchiral_projector_certified=False, cross_channel_agreement_certified=False,
        integrated_amplitude_certified=False, production_algorithm_changed=False,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in artifacts if path.is_file()}, elapsed_seconds=time.perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(progress=lambda message: print(message, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in (
        "passed", "case_count", "coefficient_comparisons", "maximum_absolute_coefficient_error",
        "maximum_relative_coefficient_error", "maximum_relative_H_value_error", "elapsed_seconds")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
