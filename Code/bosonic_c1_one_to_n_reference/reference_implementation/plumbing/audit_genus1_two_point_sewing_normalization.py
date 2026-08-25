#!/usr/bin/env python3
"""Target-free sewing audit for the c=1 torus two-point normalization.

The audit deliberately does not import a torus amplitude from the string
notes, the matrix model, or the literature.  It starts from the same local
data used in the v5 genus-two normalization note:

* normalized three-punctured-sphere tensors;
* inverse BPZ metrics on every sewn edge;
* Xi's edgewise plumbing two-form;
* the local necklace-to-period map; and
* the quotient by the double-edge graph symmetry.

For the torus with two labelled external punctures the maximal stable graph
has two trivalent sphere vertices and two parallel internal edges.  In the
necklace chart

    q1 = exp(i z),            q2 = exp(i (2*pi*tau - z)),

so q1*q2 = exp(2*pi*i*tau).  The result of the audit is

    A_1,2 = 8*pi^2*i*g_s^2 * I_1,2.

The double-Gamma/Upsilon implementation is not involved in this audit; its
normalization remains a separate local-CFT input.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus1_two_point_normalization/sewing_audit.json"
)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _determinant(matrix: Sequence[Sequence[complex]]) -> complex:
    """Small exact-by-construction determinant, avoiding a NumPy dependency."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    total = 0j
    for permutation in itertools.permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        term = complex(-1 if inversions % 2 else 1)
        for row, column in enumerate(permutation):
            term *= complex(matrix[row][column])
        total += term
    return total


def _parallel_edge_automorphisms() -> list[tuple[int, int]]:
    """Enumerate graph automorphisms preserving both labelled external legs.

    The external labels fix the two trivalent vertices.  The only remaining
    freedom is a permutation of the two indistinguishable parallel edges.
    """

    edges = (0, 1)
    return [permutation for permutation in itertools.permutations(edges)]


def _polyakov_phases_from_sewing() -> dict[str, complex]:
    """Derive N_04 and N_12 from the one-particle sewing recurrences.

    With N_03=1, separating sewing of two three-punctured spheres gives

        N_03*N_03 = -i*N_04,

    and nonseparating sewing gives

        N_04 = -i*N_12.

    These are the phase recurrences fixed by the inverse physical BPZ metric
    and the oriented tube residue in the v5 normalization strategy.
    """

    n_03 = 1.0 + 0.0j
    n_04 = 1j * n_03 * n_03
    n_12 = 1j * n_04
    return {"N_03": n_03, "N_04": n_04, "N_12": n_12}


def build_audit(
    *,
    alpha_prime: float = 1.0,
    g_s: float = 1.0,
) -> dict[str, object]:
    """Return the complete target-free factor ledger."""

    alpha_prime = _positive_finite("alpha_prime", alpha_prime)
    g_s = _positive_finite("g_s", g_s)

    vertices = 2
    internal_edges = 2
    independent_loops = internal_edges - vertices + 1
    if independent_loops != 1:
        raise AssertionError("the double-edge graph must have one loop")

    # Residual c=1 sphere topology metric.  The alpha' dependence cancels
    # before the maximal graph is evaluated.
    k_tilde_sphere = 2.0 / math.sqrt(alpha_prime)
    k_residual_sphere = math.sqrt(alpha_prime) * k_tilde_sphere
    sphere_tensor_factor = k_residual_sphere**vertices
    inverse_metric_factor = k_residual_sphere ** (-internal_edges)
    topology_factor = sphere_tensor_factor * inverse_metric_factor

    # Holomorphic local map in coordinates (tau,z):
    #   d log q1 = i dz,
    #   d log q2 = 2*pi*i d tau - i dz.
    holomorphic_matrix = (
        (0.0 + 0.0j, 0.0 + 1.0j),
        (2.0 * math.pi * 1j, 0.0 - 1.0j),
    )
    holomorphic_jacobian = _determinant(holomorphic_matrix)
    nonchiral_jacobian = abs(holomorphic_jacobian) ** 2

    # The natural tube order is (q1,bar q1,q2,bar q2).  Grouping the
    # holomorphic and antiholomorphic forms takes one transposition.
    edgewise_to_grouped_orientation = -1

    # Direct determinant of
    #   d tau ^ d z ^ d bar(tau) ^ d bar(z)
    # relative to d tau_1 ^ d tau_2 ^ d z_1 ^ d z_2.
    complex_to_real_matrix = (
        (1.0, 1j, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1j),
        (1.0, -1j, 0.0, 0.0),
        (0.0, 0.0, 1.0, -1j),
    )
    complex_to_real = _determinant(complex_to_real_matrix)

    phases = _polyakov_phases_from_sewing()
    polyakov_phase = phases["N_12"]
    form_orientation_phase = (
        polyakov_phase * edgewise_to_grouped_orientation
    )

    automorphisms = _parallel_edge_automorphisms()
    graph_quotient = 1.0 / len(automorphisms)

    geometric_coefficient = (
        form_orientation_phase.real
        * nonchiral_jacobian
        * complex_to_real.real
        * graph_quotient
    )

    # Independent gauge-fixed-coordinate ledger.  Integrating the common
    # insertion over the torus and dividing by translations produces 4*pi^2.
    # The tau and relative-z forms then contribute 2*2, followed by the same
    # double-edge/reflection quotient.
    sample_tau_2 = 1.37
    center_of_mass_volume = 8.0 * math.pi**2 * sample_tau_2
    translation_ckv_volume = 2.0 * sample_tau_2
    ckv_reduction = center_of_mass_volume / translation_ckv_volume
    coordinate_form_conversion = 2.0**2
    ckv_geometric_coefficient = (
        ckv_reduction * coordinate_form_conversion * graph_quotient
    )

    # Lorentzian phase of the connected target-time contraction.  There are
    # two sphere Fourier tensors, two inverse timelike BPZ metrics, and one
    # independent loop-energy contour rotation.  The rotation orientation is
    # the one already fixed by the positive-i tree-level Fourier convention.
    sphere_fourier_phase = 1j**vertices
    inverse_bpz_phase = (1.0 / 1j) ** internal_edges
    loop_rotation_phase = 1j**independent_loops
    lorentzian_phase = (
        sphere_fourier_phase * inverse_bpz_phase * loop_rotation_phase
    )

    coupling_factor = g_s ** (2 * 1 - 2 + 2)
    coefficient = (
        geometric_coefficient
        * topology_factor
        * coupling_factor
        * lorentzian_phase
    )
    expected_geometric = 8.0 * math.pi**2

    tolerance = 2.0e-13
    checks = {
        "residual_sphere_metric_is_2": abs(k_residual_sphere - 2.0) < tolerance,
        "sphere_metrics_cancel": abs(topology_factor - 1.0) < tolerance,
        "holomorphic_jacobian_is_2pi": (
            abs(holomorphic_jacobian - 2.0 * math.pi) < tolerance
        ),
        "nonchiral_jacobian_is_4pi2": (
            abs(nonchiral_jacobian - 4.0 * math.pi**2) < tolerance
        ),
        "complex_to_real_factor_is_4": abs(complex_to_real - 4.0) < tolerance,
        "N_12_is_minus_one": abs(polyakov_phase + 1.0) < tolerance,
        "N_12_cancels_edgewise_orientation": (
            abs(form_orientation_phase - 1.0) < tolerance
        ),
        "double_edge_quotient_is_one_half": graph_quotient == 0.5,
        "plumbing_and_ckv_ledgers_agree": (
            abs(geometric_coefficient - ckv_geometric_coefficient) < tolerance
        ),
        "geometric_coefficient_is_8pi2": (
            abs(geometric_coefficient - expected_geometric) < tolerance
        ),
        "lorentzian_phase_is_plus_i": abs(lorentzian_phase - 1j) < tolerance,
        "coupling_power_is_gs2": abs(coupling_factor - g_s**2) < tolerance,
        "final_coefficient_is_8pi2_i_gs2": (
            abs(coefficient - 8.0 * math.pi**2 * 1j * g_s**2) < tolerance
        ),
    }

    return {
        "scope": "absolute normalization of the reduced c=1 torus two-point amplitude",
        "result": {
            "formula": "A_1,2 = 8*pi^2*i*g_s^2*I_1,2",
            "coefficient_real": coefficient.real,
            "coefficient_imag": coefficient.imag,
            "g_s": g_s,
            "geometric_coefficient": geometric_coefficient,
            "lorentzian_phase": "+i",
        },
        "external_target_controls": {
            "string_note_torus_prefactor_used": False,
            "matrix_model_amplitude_used": False,
            "literature_torus_amplitude_used": False,
            "numerical_fit_used": False,
        },
        "maximal_graph": {
            "vertices": vertices,
            "internal_edges": internal_edges,
            "independent_loops": independent_loops,
            "automorphisms_preserving_external_labels": [
                list(permutation) for permutation in automorphisms
            ],
            "automorphism_order": len(automorphisms),
            "quotient": graph_quotient,
        },
        "sphere_and_state_metrics": {
            "alpha_prime": alpha_prime,
            "K_tilde_S2": k_tilde_sphere,
            "K_residual_S2": k_residual_sphere,
            "two_sphere_tensors": sphere_tensor_factor,
            "two_inverse_BPZ_metrics": inverse_metric_factor,
            "net_topology_factor": topology_factor,
        },
        "plumbing_map": {
            "q1": "exp(i*z)",
            "q2": "exp(i*(2*pi*tau-z))",
            "q1_q2": "exp(2*pi*i*tau)",
            "det_dlogq_over_dtaudz_real": holomorphic_jacobian.real,
            "det_dlogq_over_dtaudz_imag": holomorphic_jacobian.imag,
            "absolute_square": nonchiral_jacobian,
            "edgewise_to_grouped_orientation": edgewise_to_grouped_orientation,
            "complex_top_form_to_positive_real": complex_to_real.real,
        },
        "phase_ledger": {
            "N_03": str(phases["N_03"]),
            "N_04_from_separating_sewing": str(phases["N_04"]),
            "N_12_from_nonseparating_sewing": str(phases["N_12"]),
            "N_12_times_edgewise_orientation": str(form_orientation_phase),
            "sphere_Fourier_tensors": str(sphere_fourier_phase),
            "inverse_timelike_BPZ_metrics": str(inverse_bpz_phase),
            "one_loop_energy_rotation": str(loop_rotation_phase),
            "net_Lorentzian_phase": str(lorentzian_phase),
        },
        "independent_ckv_ledger": {
            "sample_tau_2": sample_tau_2,
            "center_of_mass_volume": center_of_mass_volume,
            "translation_CKV_volume": translation_ckv_volume,
            "ratio": ckv_reduction,
            "two_complex_form_conversion": coordinate_form_conversion,
            "graph_quotient": graph_quotient,
            "geometric_coefficient": ckv_geometric_coefficient,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the c=1 torus two-point prefactor from maximal sewing, "
            "without an external target amplitude."
        )
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--g-s", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_audit(alpha_prime=args.alpha_prime, g_s=args.g_s)
    if not payload["passed"]:
        failed = [name for name, passed in payload["checks"].items() if not passed]
        raise RuntimeError(f"torus two-point sewing audit failed: {failed}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("Genus-one two-point sewing normalization audit")
    print("  external torus target used: no")
    print("  topology factor K_res^2*(K_res^-1)^2 = 1: passed")
    print("  det[d(log q1,q2)/d(tau,z)] = 2*pi: passed")
    print("  N_12=-1 cancels the edgewise-form orientation: passed")
    print("  double-edge quotient = 1/2: passed")
    print("  Lorentzian loop phase = +i: passed")
    print("  A_1,2 = 8*pi^2*i*g_s^2*I_1,2: passed")
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
