#!/usr/bin/env python3
"""Audit all-NS bosonization in the saved, explicitly marked theta charts.

The charge basis is fixed by the A0/A1 seams and the infinity reference
edge. The integer B-period shift is read from the saved geometry, validated
against direct normalized one-forms, and applied to the spin characteristic.
No production parameters or saved denominators are changed.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "Code", ROOT / "Code/genus_2_cross_channel"):
    sys.path.insert(0, str(directory))

from audit_fixed_spin_free_q_expansion import LEVELS, evaluate, ns_majorana_coefficients, serializable
from fixed_spin_free_plumbing import charge_lattice_sum, direct_charged_fock_sum, fixed_spin_partition
from free_boson_plumbing import riemann_theta_constant_genus2
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from plumbing_algorithms import solve_theta_collocation
from spin_structure import SpinCharacteristic


def matrix(value):
    return np.asarray([[complex(z) for z in row] for row in value])


def action(m, omega):
    return (m[:2, :2] @ omega + m[:2, 2:]) @ np.linalg.inv(m[2:, :2] @ omega + m[2:, 2:])


def run(config_path, output):
    started = time.monotonic()
    payload = config_path.read_bytes()
    config = json.loads(payload)
    m = np.asarray(config["source_to_target"], dtype=int)
    source_spin = SpinCharacteristic((1, 1), (0, 0))
    target_spin = source_spin.transport(m)  # Also verifies that M is symplectic.
    assert target_spin.pairs == ((0, 0), (0, 0))
    coefficients = ns_majorana_coefficients(max(LEVELS))
    integer_charges = np.asarray(list(itertools.product(range(-5, 6), repeat=2)))
    rows = []
    for point in config["points"]:
        chart = point["target"]
        q = tuple(map(complex, chart["q_values"]))
        omega = matrix(chart["omega"])
        assert SpinCharacteristic.from_pairs(chart["characteristic"]) == target_spin
        branch = np.asarray(chart["period_branch"], dtype=int)
        collocation = solve_theta_collocation(
            *q, basis_order=chart["collocation_basis_order"], samples_per_seam=112)
        exact = fixed_spin_partition(q, omega, target_spin.pairs,
                                     period_branch=branch, max_mode=32, lattice_cutoff=5)
        charge_omega = np.asarray(exact["omega_charge"])
        oscillator = exact["boson_chiral"]
        charge_spin = SpinCharacteristic.from_pairs(exact["characteristic_charge"])
        modular_error = float(np.max(abs(action(m, matrix(point["source"]["omega"])) - omega)))
        a_error = float(np.max(abs(collocation.a_period_matrix - np.eye(2))))
        marked_error = float(np.max(abs(collocation.omega - omega)))
        charge_error = float(np.max(abs(charge_omega - collocation.omega - branch)))
        if modular_error > 1e-12 or a_error > 1e-12 or marked_error > 1e-8 or charge_error > 1e-10:
            raise ArithmeticError("the charge and geometric homology frames do not agree")

        # Check all four B-cycle spin choices, including their complex phase.
        spin_checks = []
        for beta in itertools.product((0, 1), repeat=2):
            marked_spin = SpinCharacteristic((0, 0), beta)
            sewing_spin = marked_spin.charge_frame(branch)
            parity = (np.einsum("ni,ij,nj->n", integer_charges, branch, integer_charges)
                      + integer_charges @ (np.asarray(sewing_spin.beta) - np.asarray(beta))) % 2
            assert not np.any(parity)
            theta_marked = complex(riemann_theta_constant_genus2(omega, marked_spin.pairs, tol=1e-15))
            # Exact T_B identity separates basis transport from inverse-map error.
            translated = charge_lattice_sum(omega + branch, sewing_spin.pairs, cutoff=5)
            theta_charge = charge_lattice_sum(charge_omega, sewing_spin.pairs, cutoff=5)
            direct = theta_physical_fermion_fredholm(
                q, sewing_spin.all_ns_determinant_lifts(), max_mode=32).determinant_values[0]
            spin_checks.append({
                "marked_spin": marked_spin.pairs, "charge_spin": sewing_spin.pairs,
                "termwise_integer_charge_sign_identity": True,
                "exact_translation_complex_relative_error": abs(translated / theta_marked - 1),
                "charge_vs_saved_marked_theta_complex_relative_error": abs(theta_charge / theta_marked - 1),
                "direct_majorana_squared_vs_bosonized_complex_relative_error": abs(direct**2 / (oscillator * theta_charge) - 1),
            })

        roots = tuple(eta * cmath.sqrt(z) for eta, z in zip(charge_spin.all_ns_determinant_lifts(), q))
        expected = oscillator * exact["dirac_charge_sum"]
        sweeps = []
        for level in LEVELS:
            bosonized = direct_charged_fock_sum(q, charge_spin.pairs,
                                              total_level=level, lattice_cutoff=5)["dirac_chiral"]
            majorana = complex(evaluate(coefficients, roots, 2 * level))
            sweeps.append({
                "oscillator_cutoff": level,
                "bosonized_dirac_fock": bosonized,
                "majorana_fock_squared": majorana**2,
                "bosonized_vs_formula_complex_relative_error": abs(bosonized / expected - 1),
                "bosonized_vs_direct_majorana_squared_complex_relative_error": abs(bosonized / majorana**2 - 1),
            })
        wrong = charge_lattice_sum(charge_omega, target_spin.pairs, cutoff=5)
        rows.append({
            "point_id": point["point_id"], "q_values": q,
            "omega_target_marked": omega, "omega_charge": charge_omega,
            "spin_frame": exact["spin_frame"],
            "source_to_target_period_residual": modular_error,
            "A_period_identity_residual": a_error,
            "geometric_period_vs_saved_marked_residual": marked_error,
            "charge_period_vs_geometric_period_plus_B_residual": charge_error,
            "same_q_marked_period_from_charge": charge_omega - branch,
            "all_four_spin_checks": spin_checks,
            "sweep": sweeps,
            "bosonized_Z_free": exact["Z_free"],
            "saved_Z_free_relative_error": abs(exact["Z_free"] / chart["Z_free"] - 1),
            "wrong_untransported_spin_majorana_norm_relative_change": abs(wrong) / abs(exact["dirac_charge_sum"]) - 1,
        })
        print(point["point_id"], "charge beta", charge_spin.beta, "homology checked", flush=True)

    spin_rows = [s for r in rows for s in r["all_four_spin_checks"]]
    maxima = {key: max(r[key] for r in rows) for key in (
        "source_to_target_period_residual", "A_period_identity_residual",
        "geometric_period_vs_saved_marked_residual",
        "charge_period_vs_geometric_period_plus_B_residual", "saved_Z_free_relative_error")}
    maxima.update({key: max(s[key] for s in spin_rows) for key in (
        "exact_translation_complex_relative_error",
        "charge_vs_saved_marked_theta_complex_relative_error",
        "direct_majorana_squared_vs_bosonized_complex_relative_error")})
    if maxima["exact_translation_complex_relative_error"] > 1e-12 or maxima["direct_majorana_squared_vs_bosonized_complex_relative_error"] > 1e-12:
        raise ArithmeticError("the complex bosonization or homology-translation identity failed")
    convergence = [{"oscillator_cutoff": level,
                    **{key: max(s[key] for r in rows for s in r["sweep"] if s["oscillator_cutoff"] == level)
                       for key in ("bosonized_vs_formula_complex_relative_error",
                                   "bosonized_vs_direct_majorana_squared_complex_relative_error")}}
                   for level in LEVELS]
    dependencies = [Path(__file__), Path(__file__).with_name("audit_fixed_spin_free_q_expansion.py"),
                    Path(__file__).with_name("fixed_spin_free_plumbing.py"),
                    Path(__file__).with_name("spin_structure.py"),
                    Path(__file__).with_name("physical_free_plumbing_resummation.py"),
                    ROOT / "Code/genus_2_cross_channel/plumbing_algorithms.py",
                    ROOT / "Code/genus_2_cross_channel/free_boson_plumbing.py",
                    ROOT / "Code/genus_2_cross_channel/free_majorana_pair_of_pants.py"]
    report = {
        "schema": "all-NS-bosonization-homology-v1", "status": "passed",
        "config_path": str(config_path.resolve()), "config_sha256": hashlib.sha256(payload).hexdigest(),
        "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        "cycle_ledger": {
            "A0_A1": "positively oriented loops around the zero and one seams on sphere 0",
            "B0_B1": "infinity seam to zero/one seam on sphere 0, cross that seam, return to infinity on sphere 1, close through the infinity seam",
            "normalized_forms": "integral over A_j of omega_i = delta_ij; Omega_ij = integral over B_j of omega_i",
            "charge_labels": "a0=n0, a1=n1, a_infinity=-(n0+n1) with all charges outgoing",
            "primary": "exp((n0^2 Log(q0)+n1^2 Log(q1)+(n0+n1)^2 Log(q_infinity))/2)",
            "principal_logs": "Log(q0)+Log(q_infinity) is kept separately; never silently replaced by Log(q0*q_infinity)",
            "basis_translation": "A_charge=A_marked; B_charge_i=B_marked_i+sum_j B_ij A_marked_j",
            "period_translation": "Omega_charge=Omega_target_marked+B",
            "all_NS_spin_translation": "alpha_charge=0; beta_charge=beta_marked+diag(B) modulo 2",
            "theta_identity": "theta[0,beta_charge/2](Omega_marked+B)=theta[0,beta_marked/2](Omega_marked), including complex phase",
            "source_to_target_symplectic_matrix": m.tolist(),
            "source_spin": source_spin.pairs, "target_spin": target_spin.pairs,
        },
        "maxima": maxima, "convergence": convergence, "points": rows,
        "production_values_or_parameters_changed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n")
    lines = ["# All-NS bosonization with explicit homology", "",
             "The saved all-NS free denominator already uses integer-charge bosonization. This audit freshly compares the charged Fock sum, direct Majorana sewing, and theta constant, with the marked homology tracked explicitly.", "",
             "A0 and A1 encircle the zero and one seams on sphere 0. The B0 and B1 paths use the infinity seam as their common return edge. Normalized forms have identity A-periods. Charges are (n0,n1,-n0-n1), with n0,n1 integers; their primary powers are q0^(n0²/2) q1^(n1²/2) q∞^((n0+n1)²/2), evaluated using the separate principal logs.", "",
             "The source-to-target symplectic map is checked against both saved period matrices and transports [11|00] to [00|00]. The bosonized sum uses Omega_charge=Omega_target_marked+B, with beta_charge=beta_target+diag(B) mod 2. This is a B-cycle shear with the A-cycles unchanged. No modular reduction or unrecorded cycle permutation is applied.", "",
             "| Saved targets | Integer B | Marked spin | Charge spin | Charge sign in sum |", "|---|---|---|---|---|",
             r"| generic_01, generic_05 | [[-1,-1],[-1,1]] | [00\|00] | [00\|11] | (-1)^(n0+n1) |",
             r"| generic_02–04, generic_06–10 | [[0,0],[0,1]] | [00\|00] | [00\|01] | (-1)^n1 |", "",
             "For integer n, nᵀBn = diag(B)·n mod 2. The additional spin sign therefore cancels the phase from the period translation term by term. The resulting theta constants agree as complex numbers, not only in modulus. All four all-NS B-cycle spin choices were checked at every surface.", "",
             "| Oscillator cutoff | Bosonized Dirac Fock / resummed P theta error | Bosonized Dirac Fock / direct Majorana Fock squared error |", "|---:|---:|---:|"]
    for row in convergence:
        lines.append(f"| {row['oscillator_cutoff']} | {row['bosonized_vs_formula_complex_relative_error']:.6e} | {row['bosonized_vs_direct_majorana_squared_complex_relative_error']:.6e} |")
    lines += ["", "These are maximum absolute relative errors over ten surfaces. Each direct sum has the stated oscillator cutoff, and charged primary powers are retained. The determinant comparison uses the existing mode cutoff 32; the integer lattice cutoff remains 5.", "", "Additional checks:", ""]
    lines += [f"- {key}: {value:.6e}" for key, value in maxima.items()]
    lines += ["", "The residual against the saved marked period comes from the original finite-accuracy inverse plumbing map. The direct charged and geometric periods at the same q agree more closely after the recorded B shift. The target denominators are unchanged.", "",
              "Run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /private/tmp/type0b-nsrr-smoke-venv/bin/python Code/genus_2/audit_all_ns_bosonization_homology.py`", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(serializable({"maxima": maxima, "convergence": convergence}), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Data Set/all_ns_bosonization_homology_20260912")
    args = parser.parse_args()
    run(args.config, args.output)
