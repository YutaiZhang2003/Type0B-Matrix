#!/usr/bin/env python3
"""Full marked-plumbing factorization test for the genus-two c=1 integrand.

This test compares differential-form coefficients in the same local
coordinates ``(tau_left, tau_right, q_bridge)``.  The left side uses:

* the actual glasses plumbing period map;
* a numerical complex Jacobian of that full map;
* the production free-boson primitive-word product;
* the production CCY Liouville glasses block; and
* the production genus-two compact/Mumford integrand.

The right side is reconstructed from the tested genus-one density and two
fixed-puncture Liouville torus one-point blocks.  It is the leading neutral
compact-vacuum/Liouville-primary term.  Bridge descendants and finite-q
period-map corrections are present only on the production side, so their
effect must vanish as ``q_bridge -> 0``.

The comparison is on the marked plumbing cover, with no genus-two stack
factor.  At ``alpha'=1`` the expected leading coefficient is

    K_sep^q = 1/(2*pi*r*|q|^2)
              int_0^inf dP/pi |q|^(2 P^2) B_left(P) B_right(P).

No matrix-model result or fitted normalization is used.
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        glasses_free_boson_product,
        noncompact_scalar_zero_mode_factor,
    )
    from genus2_c1_string_integrand import (
        SameFrameMatterPartitions,
        genus2_c1_string_integrand_density,
    )
    from integrate_tau_compact_liouville import (
        string_note_genus1_integrand_per_liouville_volume,
    )
    from liouville_genus2 import liouville_genus2_pair_of_tori
    from liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing_algorithms import glasses_collocation_period_matrix
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        glasses_free_boson_product,
        noncompact_scalar_zero_mode_factor,
    )
    from plumbing.genus2_c1_string_integrand import (
        SameFrameMatterPartitions,
        genus2_c1_string_integrand_density,
    )
    from plumbing.integrate_tau_compact_liouville import (
        string_note_genus1_integrand_per_liouville_volume,
    )
    from plumbing.liouville_genus2 import liouville_genus2_pair_of_tori
    from plumbing.liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing.plumbing_algorithms import glasses_collocation_period_matrix


ALPHA_PRIME = 1.0


@dataclass(frozen=True)
class PeriodJacobian:
    omega: np.ndarray
    determinant: complex
    abs_squared: float
    cauchy_riemann_error: float
    seam_residual: float
    symmetry_error: float


def _q_from_tau(tau: complex) -> complex:
    return cmath.exp(2j * math.pi * complex(tau))


def _tau_from_q(q_value: complex) -> complex:
    q_value = complex(q_value)
    if not 0.0 < abs(q_value) < 1.0:
        raise ValueError("handle nomes must have modulus strictly between zero and one")
    return cmath.log(q_value) / (2j * math.pi)


def _period_vector(omega: np.ndarray) -> np.ndarray:
    omega = 0.5 * (
        np.asarray(omega, dtype=np.complex128)
        + np.asarray(omega, dtype=np.complex128).T
    )
    return np.asarray([omega[0, 0], omega[1, 1], omega[0, 1]], dtype=np.complex128)


def _period_matrix(
    tau_left: complex,
    tau_right: complex,
    q_bridge: complex,
    *,
    basis_order: int,
    samples_per_seam: int,
) -> tuple[np.ndarray, float, float]:
    omega, seam_residual, symmetry_error = glasses_collocation_period_matrix(
        _q_from_tau(tau_left),
        _q_from_tau(tau_right),
        complex(q_bridge),
        basis_order=int(basis_order),
        samples_per_seam=int(samples_per_seam),
    )
    omega = 0.5 * (
        np.asarray(omega, dtype=np.complex128)
        + np.asarray(omega, dtype=np.complex128).T
    )
    return omega, float(seam_residual), float(symmetry_error)


def _complex_derivative(
    coordinates: tuple[complex, complex, complex],
    index: int,
    step: float,
    *,
    basis_order: int,
    samples_per_seam: int,
) -> tuple[np.ndarray, float]:
    """Differentiate the holomorphic period map in real and imaginary directions."""

    def value(displacement: complex) -> np.ndarray:
        shifted = list(coordinates)
        shifted[index] += displacement
        omega, _, _ = _period_matrix(
            shifted[0],
            shifted[1],
            shifted[2],
            basis_order=basis_order,
            samples_per_seam=samples_per_seam,
        )
        return _period_vector(omega)

    real_derivative = (
        -value(2.0 * step)
        + 8.0 * value(step)
        - 8.0 * value(-step)
        + value(-2.0 * step)
    ) / (12.0 * step)
    imaginary_derivative = (
        -value(2.0j * step)
        + 8.0 * value(1.0j * step)
        - 8.0 * value(-1.0j * step)
        + value(-2.0j * step)
    ) / (12.0j * step)
    scale = max(float(np.max(np.abs(real_derivative))), 1.0e-300)
    cauchy_error = float(np.max(np.abs(real_derivative - imaginary_derivative)) / scale)
    return 0.5 * (real_derivative + imaginary_derivative), cauchy_error


def period_jacobian(
    tau_left: complex,
    tau_right: complex,
    q_bridge: complex,
    *,
    basis_order: int,
    samples_per_seam: int,
    tau_step: float,
    bridge_relative_step: float,
) -> PeriodJacobian:
    coordinates = (complex(tau_left), complex(tau_right), complex(q_bridge))
    bridge_step = max(1.0e-7, bridge_relative_step * abs(complex(q_bridge)))
    columns = []
    cauchy_errors = []
    for index, step in enumerate((tau_step, tau_step, bridge_step)):
        derivative, error = _complex_derivative(
            coordinates,
            index,
            float(step),
            basis_order=basis_order,
            samples_per_seam=samples_per_seam,
        )
        columns.append(derivative)
        cauchy_errors.append(error)
    matrix = np.column_stack(columns)
    determinant = complex(np.linalg.det(matrix))
    omega, seam_residual, symmetry_error = _period_matrix(
        tau_left,
        tau_right,
        q_bridge,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
    )
    return PeriodJacobian(
        omega=omega,
        determinant=determinant,
        abs_squared=float(abs(determinant) ** 2),
        cauchy_riemann_error=max(cauchy_errors),
        seam_residual=seam_residual,
        symmetry_error=symmetry_error,
    )


def _bry_liouville_vacuum_trace(tau: complex) -> float:
    q_value = _q_from_tau(tau)
    eta_abs = dedekind_eta_abs_from_q(q_value)
    return float(1.0 / (4.0 * math.pi * math.sqrt(tau.imag) * eta_abs**2))


def fixed_block_bridge_integral(
    *,
    tau_left: complex,
    tau_right: complex,
    q_bridge: complex,
    radius: float,
    p_max: float,
    quadrature_order: int,
    theta_cutoff: int,
    dps: int,
) -> tuple[float, float]:
    """Return the leading bridge integral and the I1/direct-block check."""

    q_left = _q_from_tau(tau_left)
    q_right = _q_from_tau(tau_right)
    liouville = liouville_genus2_pair_of_tori(
        b=1.0,
        q1=q_left,
        q2=q_right,
        q_bridge=q_bridge,
        block_order=1,
        bridge_p_max=p_max,
        handle_p_max=p_max,
        bridge_quadrature_order=quadrature_order,
        handle_quadrature_order=quadrature_order,
        dps=dps,
        include_bridge_vacuum_energy=True,
        include_cosmological_prefactor=False,
    )
    i1_left = string_note_genus1_integrand_per_liouville_volume(
        tau_left,
        radius=radius,
        theta_cutoff=theta_cutoff,
        alpha_prime=ALPHA_PRIME,
    )
    i1_right = string_note_genus1_integrand_per_liouville_volume(
        tau_right,
        radius=radius,
        theta_cutoff=theta_cutoff,
        alpha_prime=ALPHA_PRIME,
    )
    zl0_left = _bry_liouville_vacuum_trace(tau_left)
    zl0_right = _bry_liouville_vacuum_trace(tau_right)
    eta_left = dedekind_eta_abs_from_q(q_left)
    eta_right = dedekind_eta_abs_from_q(q_right)
    theta_left = 4.0 * math.pi * tau_left.imag**2 * i1_left / radius
    theta_right = 4.0 * math.pi * tau_right.imag**2 * i1_right / radius

    from_i1 = 0.0
    direct = 0.0
    for sample in liouville.samples:
        bridge_factor = abs(q_bridge) ** (
            2.0 * sample.bridge_momentum * sample.bridge_momentum
        )
        torus_left = float(sample.left_torus_one_point.real)
        torus_right = float(sample.right_torus_one_point.real)
        b_left = tau_left.imag * i1_left * torus_left / zl0_left
        b_right = tau_right.imag * i1_right * torus_right / zl0_right
        b_left_direct = (
            radius * theta_left * eta_left**2 * torus_left / math.sqrt(tau_left.imag)
        )
        b_right_direct = (
            radius * theta_right * eta_right**2 * torus_right / math.sqrt(tau_right.imag)
        )
        weight = sample.bridge_measure_weight * bridge_factor
        from_i1 += weight * b_left * b_right
        direct += weight * b_left_direct * b_right_direct
    return float(from_i1), float(from_i1 / direct)


def evaluate_point(
    *,
    tau_left: complex,
    tau_right: complex,
    q_bridge: float,
    radius: float,
    basis_order: int,
    stability_basis_increment: int,
    samples_per_basis: int,
    tau_step: float,
    bridge_relative_step: float,
    scalar_word_length: int,
    scalar_max_mode: int,
    p_max: float,
    quadrature_order: int,
    theta_cutoff: int,
    theta_nmax: int,
    dps: int,
) -> dict[str, float]:
    q_bridge_complex = complex(float(q_bridge), 0.0)
    samples = samples_per_basis * basis_order
    jacobian = period_jacobian(
        tau_left,
        tau_right,
        q_bridge_complex,
        basis_order=basis_order,
        samples_per_seam=samples,
        tau_step=tau_step,
        bridge_relative_step=bridge_relative_step,
    )
    stability_basis = basis_order + stability_basis_increment
    stable_jacobian = period_jacobian(
        tau_left,
        tau_right,
        q_bridge_complex,
        basis_order=stability_basis,
        samples_per_seam=samples_per_basis * stability_basis,
        tau_step=0.5 * tau_step,
        bridge_relative_step=0.5 * bridge_relative_step,
    )
    omega = stable_jacobian.omega
    q_values = (_q_from_tau(tau_left), _q_from_tau(tau_right), q_bridge_complex)

    scalar_product = glasses_free_boson_product(
        *q_values,
        max_word_length=scalar_word_length,
        max_mode=scalar_max_mode,
        tolerance=1.0e-14,
    )
    scalar_partition = (
        scalar_product.nonchiral_value * noncompact_scalar_zero_mode_factor(omega)
    )
    liouville = liouville_genus2_glasses_partition(
        b=1.0,
        q_left=q_values[0],
        q_right=q_values[1],
        q_bridge=q_values[2],
        block_order=1,
        p_max=p_max,
        quadrature_order=quadrature_order,
        dps=dps,
        propagator_shift=0.0,
        include_vacuum_seed=True,
        vacuum_word_len=scalar_word_length,
        vacuum_oscillator_level_max=scalar_max_mode,
        include_cosmological_prefactor=False,
        store_samples=False,
    )
    liouville_partition = float(liouville.value.real)
    if liouville_partition <= 0.0 or abs(liouville.value.imag) > 1.0e-10 * liouville_partition:
        raise ValueError("production Liouville partition is not positive real")

    production = genus2_c1_string_integrand_density(
        omega,
        radius,
        alpha_prime=ALPHA_PRIME,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame="plumbing:glasses",
            liouville_partition=liouville_partition,
            noncompact_scalar_partition=scalar_partition,
        ),
        lattice_tolerance=1.0e-14,
        theta_nmax=theta_nmax,
        theta_tolerance=1.0e-14,
        chi10_normalization="product",
    )
    bridge_integral, i1_rewrite_ratio = fixed_block_bridge_integral(
        tau_left=tau_left,
        tau_right=tau_right,
        q_bridge=q_bridge_complex,
        radius=radius,
        p_max=p_max,
        quadrature_order=quadrature_order,
        theta_cutoff=theta_cutoff,
        dps=dps,
    )

    # Marked-cover CFT factorization.  This is twice the stack/physical
    # alpha'/(4*pi) tube expression at alpha'=1; no stack factor is present.
    sewing_q_density = bridge_integral / (
        2.0 * math.pi * radius * abs(q_bridge_complex) ** 2
    )
    mumford_pullback = (
        production.factorization_normalized_density * stable_jacobian.abs_squared
    )
    note_pullback = production.string_note_kernel_density * stable_jacobian.abs_squared
    leading_jacobian = 1.0 / (4.0 * math.pi**2)

    return {
        "q_bridge": float(q_bridge),
        "omega12_abs": float(abs(omega[0, 1])),
        "q_over_2pi_omega12_abs": float(
            abs(q_bridge_complex) / (2.0 * math.pi * abs(omega[0, 1]))
        ),
        "period_seam_residual": float(stable_jacobian.seam_residual),
        "period_symmetry_error": float(stable_jacobian.symmetry_error),
        "jacobian_abs_squared": float(stable_jacobian.abs_squared),
        "jacobian_over_leading": float(stable_jacobian.abs_squared / leading_jacobian),
        "jacobian_basis_relative_change": float(
            stable_jacobian.abs_squared / jacobian.abs_squared - 1.0
        ),
        "jacobian_cauchy_riemann_error": float(
            stable_jacobian.cauchy_riemann_error
        ),
        "scalar_partition": float(scalar_partition),
        "scalar_primitive_count": float(scalar_product.primitive_count),
        "scalar_omitted_chiral_tail": float(
            scalar_product.omitted_chiral_tail_estimate
        ),
        "liouville_partition": float(liouville_partition),
        "mumford_omega_density": float(production.factorization_normalized_density),
        "mumford_pullback_q_density": float(mumford_pullback),
        "string_note_pullback_q_density": float(note_pullback),
        "sewing_q_density": float(sewing_q_density),
        "marked_mumford_ratio": float(mumford_pullback / sewing_q_density),
        "current_note_ratio": float(note_pullback / sewing_q_density),
        "current_note_ratio_over_1_over_pi": float(
            note_pullback / sewing_q_density * math.pi
        ),
        "i1_fixed_block_rewrite_ratio": float(i1_rewrite_ratio),
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Full marked-plumbing factorization test of the production genus-two integrand."
    )
    parser.add_argument("--q-left", type=complex, default=0.003 + 0.0j)
    parser.add_argument("--q-right", type=complex, default=0.0025 + 0.0j)
    parser.add_argument(
        "--bridge-values",
        default="0.02,0.01,0.005,0.0025,0.00125",
    )
    parser.add_argument("--radius", type=float, default=1.31)
    parser.add_argument("--basis-order", type=int, default=18)
    parser.add_argument("--stability-basis-increment", type=int, default=4)
    parser.add_argument("--samples-per-basis", type=int, default=6)
    parser.add_argument("--tau-step", type=float, default=1.0e-5)
    parser.add_argument("--bridge-relative-step", type=float, default=5.0e-4)
    parser.add_argument("--scalar-word-length", type=int, default=8)
    parser.add_argument("--scalar-max-mode", type=int, default=60)
    parser.add_argument("--p-max", type=float, default=1.5)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--theta-cutoff", type=int, default=16)
    parser.add_argument("--theta-nmax", type=int, default=10)
    parser.add_argument("--dps", type=int, default=24)
    args = parser.parse_args(list(argv) if argv is not None else None)

    tau_left = _tau_from_q(args.q_left)
    tau_right = _tau_from_q(args.q_right)
    bridge_values = tuple(
        sorted(
            (
                float(piece.strip())
                for piece in args.bridge_values.split(",")
                if piece.strip()
            ),
            reverse=True,
        )
    )
    rows = [
        evaluate_point(
            tau_left=tau_left,
            tau_right=tau_right,
            q_bridge=q_bridge,
            radius=args.radius,
            basis_order=args.basis_order,
            stability_basis_increment=args.stability_basis_increment,
            samples_per_basis=args.samples_per_basis,
            tau_step=args.tau_step,
            bridge_relative_step=args.bridge_relative_step,
            scalar_word_length=args.scalar_word_length,
            scalar_max_mode=args.scalar_max_mode,
            p_max=args.p_max,
            quadrature_order=args.quadrature_order,
            theta_cutoff=args.theta_cutoff,
            theta_nmax=args.theta_nmax,
            dps=args.dps,
        )
        for q_bridge in bridge_values
    ]

    print("Full marked-plumbing factorization test")
    print(f"  alpha'=1, radius r={args.radius:.12g}")
    print(
        "  columns: q, |J|^2/(1/4pi^2), Mumford/sewn, "
        "Knote/sewn, pi*Knote/sewn, I1->B"
    )
    for row in rows:
        print(
            f"  {row['q_bridge']:.8g}  "
            f"{row['jacobian_over_leading']:.12e}  "
            f"{row['marked_mumford_ratio']:.12e}  "
            f"{row['current_note_ratio']:.12e}  "
            f"{row['current_note_ratio_over_1_over_pi']:.12e}  "
            f"{row['i1_fixed_block_rewrite_ratio']:.12e}"
        )

    final = rows[-1]
    fit_count = min(4, len(rows))
    fit_rows = sorted(rows, key=lambda row: row["q_bridge"])[:fit_count]
    fit_q = np.asarray([row["q_bridge"] for row in fit_rows], dtype=float)
    fit_ratio = np.asarray(
        [row["marked_mumford_ratio"] for row in fit_rows], dtype=float
    )
    fit_degree = min(2, fit_count - 1)
    boundary_ratio = float(np.polyfit(fit_q, fit_ratio, fit_degree)[-1])
    print(
        f"  q->0 marked-ratio extrapolation (degree {fit_degree})="
        f"{boundary_ratio:.12e}"
    )
    marked_errors = [abs(row["marked_mumford_ratio"] - 1.0) for row in rows]
    checks = {
        "period Jacobian stability": abs(final["jacobian_basis_relative_change"]) < 2.0e-5,
        "period-map holomorphy": final["jacobian_cauchy_riemann_error"] < 2.0e-5,
        "I1 fixed-block rewriting": abs(final["i1_fixed_block_rewrite_ratio"] - 1.0) < 2.0e-12,
        "marked ratio improves toward node": marked_errors[-1] < marked_errors[0],
        "marked factorization limit": marked_errors[-1] < 2.0e-3,
        "marked boundary extrapolation": abs(boundary_ratio - 1.0) < 5.0e-4,
        "current note multiplier is 1/pi": abs(
            final["current_note_ratio_over_1_over_pi"] - 1.0
        )
        < 2.0e-3,
    }
    print(f"  all checks pass={all(checks.values())}")
    for name, passed in checks.items():
        print(f"    {name}: {passed}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    run()
