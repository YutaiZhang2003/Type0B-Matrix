#!/usr/bin/env python3
"""Exact, scoped checks for the SO(d) contraction spin-Calogero candidate.

This is NOT a string-amplitude calculation.  The scattering calculation is
for two particles with no external potential, on the ordered half-line
r = x_1 - x_2 > 0, with relative Hamiltonian

    h = -d^2/dr^2 + c/r^2.

At infinity e^{-ipr} is incoming and e^{+ipr} is outgoing, for p > 0.
For the scale-invariant endpoint branch u(r) ~ r^s, c=s(s-1), the regular
Bessel solution sqrt(r) J_{s-1/2}(pr) has reflection R_s=exp(-i*pi*s).
The Jastrow calculation at the end is a formal differential identity in an
ordered chamber with a confining oscillator, not a complete domain/spectrum
theorem.  Neither calculation is extrapolated to the inverted-oscillator
Fermi sea or to scattering of its collective bosons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


def require_zero_matrix(matrix: sp.MatrixBase, label: str) -> None:
    residual = matrix.applyfunc(sp.simplify)
    if residual != sp.zeros(*matrix.shape):
        raise AssertionError(label)


def pair_checks(d: int) -> dict:
    if d <= 6:
        raise ValueError("Use d > 6 so 0 < lambda=2/(d-4) < 1.")
    lam = sp.Rational(2, d - 4)
    size = d * d
    ident = sp.eye(size, cls=sp.SparseMatrix)
    permutation = sp.SparseMatrix(
        size, size, {(b * d + a, a * d + b): 1 for a in range(d) for b in range(d)}
    )
    contraction = sp.SparseMatrix(
        size, size, {(a * d + a, b * d + b): 1 for a in range(d) for b in range(d)}
    )
    trace = contraction / d
    antisymmetric = (ident - permutation) / 2
    symmetric_traceless = (ident + permutation) / 2 - trace
    sectors = {
        "symmetric_traceless": symmetric_traceless,
        "antisymmetric": antisymmetric,
        "trace": trace,
    }
    require_zero_matrix(permutation**2 - ident, "P^2 != I")
    require_zero_matrix(contraction**2 - d * contraction, "Q^2 != d Q")
    require_zero_matrix(permutation * contraction - contraction, "P Q != Q")
    require_zero_matrix(sum(sectors.values(), sp.zeros(size, cls=sp.SparseMatrix)) - ident,
                        "Projectors do not sum to I")
    for name, projector in sectors.items():
        require_zero_matrix(projector**2 - projector, f"{name} is not a projector")
        for other_name, other in sectors.items():
            if other_name != name:
                require_zero_matrix(projector * other, f"{name}/{other_name} not orthogonal")

    spin_coupling = lam**2 * ident - lam * permutation + lam * contraction
    couplings = {
        "symmetric_traceless": lam * (lam - 1),
        "antisymmetric": lam * (lam + 1),
        "trace": (lam + 1) * (lam + 2),
    }
    for name, projector in sectors.items():
        require_zero_matrix(spin_coupling * projector - couplings[name] * projector,
                            f"Wrong coupling in {name}")

    powers = {
        "symmetric_traceless": lam,
        "antisymmetric": lam + 1,
        "trace": lam + 2,
    }
    for name, exponent in powers.items():
        assert sp.simplify(exponent * (exponent - 1) - couplings[name]) == 0
        assert couplings[name] >= -sp.Rational(1, 4)

    # Remove a common phase before comparing matrices.  Integer changes in
    # the Bessel index give (+1,-1,+1) in the three spin channels.
    natural_reduced = symmetric_traceless - antisymmetric + trace
    require_zero_matrix(natural_reduced - permutation, "Natural pair S is not phase times P")
    natural_conversion = natural_reduced[d + 1, 0]
    assert natural_conversion == 0

    # Friedrichs selects s=1-lambda in the symmetric traceless channel only
    # when lambda < 1/2.  For larger lambda it already selects s=lambda.
    friedrichs_powers = {
        name: max(exponent, 1 - exponent) for name, exponent in powers.items()
    }
    if lam < sp.Rational(1, 2):
        # S_F = -cos(pi lambda) I - i sin(pi lambda) P
        #       + (2 cos(pi lambda)/d) Q.
        phase = sp.exp(-sp.I * sp.pi * lam)
        sf = (-sp.exp(sp.I * sp.pi * lam) * symmetric_traceless
              - phase * antisymmetric + phase * trace)
        expected_sf = (-sp.cos(sp.pi * lam) * ident
                       - sp.I * sp.sin(sp.pi * lam) * permutation
                       + 2 * sp.cos(sp.pi * lam) / d * contraction)
        require_zero_matrix((sf - expected_sf).applyfunc(sp.expand_complex),
                            "Wrong Friedrichs S matrix")
        conversion = 2 * sp.cos(sp.pi * lam) / d
        assert sp.simplify(sp.expand_complex(sf[d + 1, 0] - conversion)) == 0
    else:
        conversion = sp.Integer(0)

    return {
        "dimension": d,
        "lambda": str(lam),
        "relative_hamiltonian": "-d^2/dr^2 + c/r^2, r>0; no external potential",
        "incoming": "exp(-i p r)",
        "outgoing": "exp(+i p r)",
        "bessel_branch_reflection": "R_s = exp(-i*pi*s)",
        "projector_dimensions": {name: int(sp.trace(projector)) for name, projector in sectors.items()},
        "couplings": {name: str(value) for name, value in couplings.items()},
        "natural_scale_invariant_powers": {name: str(value) for name, value in powers.items()},
        "friedrichs_powers": {name: str(value) for name, value in friedrichs_powers.items()},
        "natural_pair_S": "exp(-i*pi*lambda) P",
        "natural_00_to_aa": str(natural_conversion),
        "friedrichs_00_to_aa": str(conversion),
        "friedrichs_00_to_aa_numeric": float(sp.N(conversion)),
        "scope": "Two-body scattering only; not an inverted-trap or collective string S matrix.",
    }


def bessel_equation_check() -> dict:
    r, p, exponent = sp.symbols("r p s", positive=True)
    u = sp.sqrt(r) * sp.besselj(exponent - sp.Rational(1, 2), p * r)
    residual = -sp.diff(u, r, 2) + exponent * (exponent - 1) * u / r**2 - p**2 * u
    residual = sp.simplify(sp.expand_func(residual))
    assert residual == 0
    return {
        "differential_equation_residual": str(residual),
        "asymptotic": "sqrt(r) J_(s-1/2)(p r) ~ sqrt(2/(pi p)) cos(p r - pi s/2)",
        "reflection_derivation": "outgoing coefficient exp(-i*pi*s/2) divided by incoming exp(+i*pi*s/2)",
    }


def jastrow_check(particle_count: int = 3) -> dict:
    coordinates = sp.symbols(f"x0:{particle_count}", real=True)
    lam, frequency = sp.symbols("lambda Omega", positive=True)
    logarithmic_gradients = [
        lam * sum(1 / (coordinate - other)
                  for index2, other in enumerate(coordinates) if index2 != index)
        - frequency * coordinate
        for index, coordinate in enumerate(coordinates)
    ]
    local_energy = -sum(
        sp.diff(gradient, coordinate) + gradient**2
        for coordinate, gradient in zip(coordinates, logarithmic_gradients)
    ) / 2
    local_energy += frequency**2 * sum(coordinate**2 for coordinate in coordinates) / 2
    local_energy += lam * (lam - 1) * sum(
        1 / (coordinates[i] - coordinates[j])**2
        for i in range(particle_count) for j in range(i + 1, particle_count)
    )
    expected = frequency * (
        sp.Rational(particle_count, 2) + lam * particle_count * (particle_count - 1) / 2
    )
    residual = sp.factor(local_energy - expected)
    assert residual == 0

    # A nonzero symmetric-traceless tensor exists for any rank: v^tensor n
    # with v=(1,i,0,...), so v.v=0.  Every permutation fixes it and every
    # pair contraction annihilates it, making 1-P+Q zero pair by pair.
    null_polarization_norm = sp.Integer(1) + sp.I**2
    assert null_polarization_norm == 0
    return {
        "particle_count_symbolically_checked": particle_count,
        "wavefunction": "product_(i<j)|xi-xj|^lambda exp(-Omega sum xi^2/2) T, T in ST^N(C^d)",
        "formal_energy": "Omega [N/2 + lambda N(N-1)/2]",
        "scalar_schrodinger_residual": str(residual),
        "spin_identity": "Pij T=T and Qij T=0, hence lambda(1-Pij+Qij)T=0",
        "explicit_tensor_witness": "T=v^tensor N, v=(1,i,0,...), v.v=0",
        "scope": "Formal identity in a chamber; requires lambda endpoint domains; not the full many-body spectrum or a singlet vacuum.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--json", type=Path, help="Also write the result to this JSON file.")
    args = parser.parse_args()
    result = {"pair": pair_checks(args.dimension),
              "bessel": bessel_equation_check(),
              "jastrow": jastrow_check()}
    serialized = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
