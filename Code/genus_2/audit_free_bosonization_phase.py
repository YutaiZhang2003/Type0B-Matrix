#!/usr/bin/env python3
r"""Compare bosonization and independent Fock sewing before taking norms.

All coordinates are theta-pants coordinates in (0,1,infinity) order,
with q^L0 propagators. Each square-root branch is fixed at degeneration
and continued without consulting the Fock answer. This diagnostic writes
new results only; it does not modify production formulas or conventions.
"""
from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from flint import fmpq

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "Code", ROOT / "Code/genus_2_cross_channel"):
    sys.path.insert(0, str(directory))

from audit_fixed_spin_free_q_expansion import (
    charged_boson_coefficients, evaluate, ns_majorana_coefficients, serializable,
)
from fixed_spin_free_plumbing import (
    charged_frame, charge_grid, charge_lattice_sum,
    fixed_spin_chiral_partition, theta_translation_phase,
)
from free_boson_plumbing import riemann_theta_constant_genus2
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from plumbing_algorithms import solve_theta_collocation
from spin_structure import SpinCharacteristic, ThetaLogBranch
from theta_fermion_ccy.direct_fermion import (
    RationalAuxiliaryThreePoint, _strict_partitions,
)

LEVELS = (6, 10, 14, 18)
CONTINUOUS_CHARGES = ((0., 0.), (.27, -.63), (.5, -.5), (.5, 1.5))


def comparison(actual, reference):
    if abs(reference) < 1e-100:
        raise ValueError("a vanishing partition function has no phase")
    ratio = complex(actual / reference)
    return {"value": complex(actual), "reference": complex(reference),
            "norm_ratio": abs(ratio), "norm_relative_error": abs(abs(ratio) - 1),
            "phase_radians": cmath.phase(actual),
            "reference_phase_radians": cmath.phase(reference),
            "phase_residual_radians": cmath.phase(ratio),
            "phase_absolute_error_radians": abs(cmath.phase(ratio)),
            "complex_relative_error": abs(ratio - 1)}


def translation_identity_check():
    """Check the characteristic-reduction phase outside the saved spin choices."""
    omega = np.array([[.13 + 1.2j, .17 + .21j], [.17 + .21j, -.11 + 1.1j]])
    maximum = 0.
    count = 0
    for branch in (((0, 0), (0, 1)), ((-1, -1), (-1, 1)), ((2, -1), (-1, 3))):
        for alpha, beta in itertools.product(itertools.product((0, 1), repeat=2), repeat=2):
            spin = SpinCharacteristic(alpha, beta)
            if spin.arf:
                continue
            before = complex(riemann_theta_constant_genus2(omega, spin.pairs, tol=1e-15))
            after = complex(riemann_theta_constant_genus2(
                omega + np.asarray(branch), spin.charge_frame(branch).pairs, tol=1e-15))
            maximum = max(maximum, abs(after / (theta_translation_phase(spin, branch) * before) - 1))
            count += 1
    if maximum > 1e-12:
        raise ArithmeticError("the theta-translation phase failed an even-spin check")
    return {"even_spin_and_branch_combinations": count, "maximum_complex_relative_error": maximum}


def ramond_majorana_coefficients(max_level):
    r"""Unbosonized R,R,NS fermion-mode sewing, with the primary stripped.

    The reused Ward oracle contains only free-fermion Clifford-mode algebra;
    none of its auxiliary-field insertion or four-edge assembly is used.
    It defines rho=i^p_one r/2^((g_one+g_zero)/2). We express the two
    pants in mutually dual spin frames, so their product is
    r^2/2^(g_one+g_zero). The two ground labels are summed explicitly.

    The standard chiral Ramond normalization is sqrt(2), rather than an
    integer-dimensional standalone graded trace. Accordingly the overall
    factor multiplying this sum is 1/sqrt(2); factoring out sqrt(2) leaves
    the 1/2 below. This is fixed before evaluating the theta formula.
    Keys are (R_zero oscillator level, R_one level, twice NS level).
    """
    form = RationalAuxiliaryThreePoint()
    coefficients = {}
    for ni in range(2 * max_level + 1):
        for n1 in range((2 * max_level - ni) // 2 + 1):
            for n0 in range((2 * max_level - ni - 2 * n1) // 2 + 1):
                value = fmpq(0)
                bases = (_strict_partitions(ni, ni, True),
                         _strict_partitions(n1, n1), _strict_partitions(n0, n0))
                for ns, r1, r0 in itertools.product(*bases):
                    for g1, g0 in itertools.product((0, 1), repeat=2):
                        r = form.value((ns, (r1, g1), (r0, g0)))
                        value += r * r / (1 << (g1 + g0))
                if value:
                    coefficients[n0, n1, ni] = value / 2
    # Analytic vacuum and first Clifford-mode Ward checkpoints.
    expected = {(0, 0, 0): fmpq(1), (0, 0, 1): fmpq(1, 2),
                (1, 0, 0): fmpq(1, 8), (0, 1, 0): fmpq(1, 8)}
    if any(coefficients[k] != v for k, v in expected.items()):
        raise ArithmeticError("Ramond ground or first-mode normalization failed")
    return coefficients


def ramond_value(coefficients, q, level, *, beta=(0, 0), logs=None):
    if beta not in ((0, 0), (1, 1)):
        raise ValueError("this chiral audit supports the two even RR spin sectors")
    logs = tuple(map(cmath.log, q)) if logs is None else logs
    primary = math.sqrt(2) * cmath.exp((logs[0] + logs[1]) / 16)
    x_ns = (-1) ** beta[0] * cmath.exp(logs[2] / 2)
    return primary * sum(float(c) * q[0]**n0 * q[1]**n1 * x_ns**ni
                         for (n0, n1, ni), c in coefficients.items()
                         if 2*n0 + 2*n1 + ni <= 2*level)


def ground_factor(q, spin):
    if spin.alpha == (0, 0):
        return 1 + 0j
    if spin.alpha == (1, 1) and spin.beta in ((0, 0), (1, 1)):
        return math.sqrt(2) * cmath.exp((cmath.log(q[0]) + cmath.log(q[1])) / 16)
    raise ValueError("unsupported degeneration in this audit")


def radial_roots(q, spins, *, steps=48, max_mode=32):
    """Continue the normalized theta expression from t*q, t=1e-6 to 1.

    No comparison to the direct Fock value is used to choose a sign.
    """
    phases = np.zeros(len(spins))
    previous = None
    max_step = 0.
    min_norm = math.inf
    anchor_error = 0.
    for i, scale in enumerate(np.geomspace(1e-6, 1., steps + 1)):
        qs = tuple(scale * z for z in q)
        frame = charged_frame(qs, max_mode=max_mode)
        ground = [ground_factor(qs, spin) for spin in spins]
        values = [frame.boson_chiral * charge_lattice_sum(
            frame.omega_charge, spin.pairs, cutoff=5) for spin in spins]
        normalized = np.asarray([v / g**2 for v, g in zip(values, ground)])
        min_norm = min(min_norm, float(min(abs(normalized))))
        if i == 0:
            anchor_error = float(max(abs(normalized - 1)))
            if anchor_error > .01:
                raise ArithmeticError("radial anchor is not in the vacuum neighborhood")
            phases = np.angle(normalized)
        else:
            increments = np.angle(normalized / previous)
            max_step = max(max_step, float(max(abs(increments))))
            if max_step >= math.pi / 4:
                raise ArithmeticError("refine the radial path before selecting a branch")
            phases += increments
        previous = normalized
    roots = [g * math.sqrt(abs(z)) * cmath.exp(.5j * phase)
             for g, z, phase in zip(ground, normalized, phases)]
    return roots, {"path": "q(t)=t*q, 1e-6 <= t <= 1", "steps": steps,
                   "anchor_normalized_squared_error": anchor_error,
                   "minimum_normalized_squared_modulus": min_norm,
                   "maximum_normalized_phase_step_radians": max_step}


def lattice_fock_value(coefficients, charges, q, beta, level):
    oscillators = evaluate(coefficients, q, level)
    a, b = charges.T
    logs = np.log(np.asarray(q))
    primary = np.exp(.5 * (a*a*logs[0] + b*b*logs[1] + (a+b)**2*logs[2])
                     + 1j * math.pi * (charges @ np.asarray(beta)))
    return complex(primary @ oscillators)


def ramond_winding_check(coefficients, *, steps_per_turn=48):
    """Eight windings expose a genuine sign invisible in M^2, in q^L0 framing."""
    q0 = (.013 + 0j, .017 + 0j, .011 + 0j)
    branch = ThetaLogBranch(q0)
    phase = 0.
    previous = None
    production = None
    maximum = 0.
    production_error = 0.
    max_step = 0.
    rows = []
    for i in range(8 * steps_per_turn + 1):
        q = (q0[0] * cmath.exp(2j * math.pi * i / steps_per_turn), *q0[1:])
        if i:
            branch = branch.advance(q, maximum_phase_step=math.pi / 4)
        frame = charged_frame(q, max_mode=24)
        omega_cont = frame.omega_charge + np.asarray(branch.period_shift)
        squared = frame.boson_chiral * charge_lattice_sum(
            omega_cont, ((1, 1), (0, 0)), cutoff=5)
        if previous is None:
            phase = cmath.phase(squared)
        else:
            step = cmath.phase(squared / previous)
            max_step = max(max_step, abs(step))
            phase += step
        root = math.sqrt(abs(squared)) * cmath.exp(.5j * phase)
        production = fixed_spin_chiral_partition(
            q, omega_cont, ((1, 1), (0, 0)),
            period_branch=-np.asarray(branch.period_shift), max_mode=24, previous=production)
        production_error = max(production_error, abs(production["majorana_chiral"]/root-1))
        direct = ramond_value(coefficients, q, max(LEVELS), logs=branch.logs)
        maximum = max(maximum, abs(direct / production["majorana_chiral"] - 1))
        if i % steps_per_turn == 0:
            rows.append({"turns": i // steps_per_turn, "log_windings": branch.windings,
                         "dirac": squared, "continued_majorana": production["majorana_chiral"],
                         "principal_majorana": cmath.sqrt(squared), "fock_majorana": direct})
        previous = squared
    squared_return = comparison(rows[-1]["dirac"], rows[0]["dirac"])
    root_return = comparison(rows[-1]["continued_majorana"], rows[0]["continued_majorana"])
    if (maximum > 1e-11 or production_error > 1e-11 or squared_return["complex_relative_error"] > 1e-11
            or abs(root_return["norm_ratio"] - 1) > 1e-11
            or abs(abs(root_return["phase_residual_radians"]) - math.pi) > 1e-11):
        raise ArithmeticError("the continued Ramond loop failed its monodromy check")
    return {"q_start": q0, "steps_per_turn": steps_per_turn,
            "maximum_fock_vs_continued_complex_error": maximum,
            "maximum_production_vs_independent_continuation_error": production_error,
            "maximum_dirac_phase_step_radians": max_step,
            "dirac_return": squared_return, "majorana_return": root_return, "rows": rows}


def run(config_path, output):
    started = time.monotonic()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    level = max(LEVELS)
    print("Building independent Heisenberg and Clifford Fock coefficients", flush=True)
    boson_coefficients = charged_boson_coefficients(CONTINUOUS_CHARGES, level)
    ns_coefficients = ns_majorana_coefficients(level)
    rr_coefficients = ramond_majorana_coefficients(level)
    charge_grids = {alpha: charge_grid(alpha, 5) for alpha in ((0, 0), (1, 1))}
    lattice_coefficients = {alpha: charged_boson_coefficients(grid, level)
                            for alpha, grid in charge_grids.items()}
    print("Fock coefficients ready", flush=True)
    points = []
    for point in config["points"]:
        for side in ("source", "target"):
            chart = point[side]
            q = tuple(map(complex, chart["q_values"]))
            marked_spin = SpinCharacteristic.from_pairs(chart["characteristic"])
            b_shift = np.asarray(chart["period_branch"], dtype=int)
            spin = marked_spin.charge_frame(b_shift)
            frame = charged_frame(q, max_mode=32)
            # Independent normalized one-form calculation at precisely the same q.
            geometry = solve_theta_collocation(*q, basis_order=chart["collocation_basis_order"],
                                               samples_per_seam=112)
            production = fixed_spin_chiral_partition(
                q, geometry.omega, marked_spin.pairs, period_branch=b_shift, max_mode=32)
            theta_marked = complex(riemann_theta_constant_genus2(
                geometry.omega, marked_spin.pairs, tol=1e-15))
            xi = theta_translation_phase(marked_spin, b_shift)
            theta_charge = charge_lattice_sum(frame.omega_charge, spin.pairs, cutoff=5)
            dirac_formula = production["dirac_chiral_from_marked_theta"]
            betas = tuple(itertools.product((0, 1), repeat=2)) if spin.alpha == (0, 0) else ((0, 0), (1, 1))
            spins = [SpinCharacteristic(spin.alpha, beta) for beta in betas]
            roots, branch_record = radial_roots(q, spins)
            charge_root = roots[betas.index(spin.beta)]
            # The geometric/charged-period residual is tiny. Continue its ratio near 1.
            period_ratio = xi * theta_marked / theta_charge
            if abs(period_ratio - 1) > 1e-9:
                raise ArithmeticError("independent geometric period is inconsistent")
            if abs(production["majorana_chiral"]/charge_root-1) > 1e-12:
                raise ArithmeticError("the production and independent radial branches disagree")
            majorana_formula = production["majorana_chiral"] * cmath.sqrt(period_ratio)
            p_formula = production["boson_chiral"]
            charges = np.asarray(CONTINUOUS_CHARGES)
            charged_exact = p_formula * np.exp(1j * math.pi * np.einsum(
                "ni,ij,nj->n", charges, frame.omega_charge, charges))
            logs = np.log(np.asarray(q))
            a, b = charges.T
            primaries = np.exp(.5 * (a*a*logs[0] + b*b*logs[1] + (a+b)**2*logs[2]))
            sweeps = []
            for cutoff in LEVELS:
                charged_direct = primaries * evaluate(boson_coefficients, q, cutoff)
                p_direct = complex(charged_direct[0])
                dirac_direct = lattice_fock_value(lattice_coefficients[spin.alpha],
                    charge_grids[spin.alpha], q, spin.beta, cutoff)
                if spin.alpha == (0, 0):
                    variables = tuple(e * cmath.sqrt(z)
                                      for e, z in zip(spin.all_ns_determinant_lifts(), q))
                    majorana_direct = complex(evaluate(ns_coefficients, variables, 2 * cutoff))
                else:
                    majorana_direct = ramond_value(rr_coefficients, q, cutoff, beta=spin.beta)
                free_formula = frame.loop_gaussian * abs(p_formula * majorana_formula)**2
                free_direct = frame.loop_gaussian * abs(p_direct * majorana_direct)**2
                sweeps.append({"level": cutoff,
                    "boson": comparison(p_direct, p_formula),
                    "dirac": comparison(dirac_direct, dirac_formula),
                    "majorana": comparison(majorana_direct, majorana_formula),
                    "superfield_chiral_at_zero_charge": comparison(
                        p_direct * majorana_direct, p_formula * majorana_formula),
                    "continuous_charges": [comparison(x, y) for x, y in zip(charged_direct, charged_exact)],
                    "majorana_squared_complex_error": abs(majorana_direct**2 / dirac_formula - 1),
                    "Z_free_formula_shared_gaussian": free_formula,
                    "Z_free_fock_shared_gaussian": free_direct,
                    "Z_free_relative_error": abs(free_direct / free_formula - 1)})
            spin_checks = []
            for other, root in zip(spins, roots):
                if other.alpha == (0, 0):
                    variables = tuple(e * cmath.sqrt(z)
                                      for e, z in zip(other.all_ns_determinant_lifts(), q))
                    direct = complex(evaluate(ns_coefficients, variables, 2 * level))
                    fredholm = theta_physical_fermion_fredholm(
                        q, other.all_ns_determinant_lifts(), max_mode=32).determinant_values[0]
                    extra = {"unsquared_fredholm": comparison(fredholm, root)}
                else:
                    direct = ramond_value(rr_coefficients, q, level, beta=other.beta)
                    extra = {}
                spin_checks.append({"charge_spin": other.pairs,
                                    "unsquared_fock": comparison(direct, root), **extra})
            lower = charged_frame(q, max_mode=24)
            d_lower = lower.boson_chiral * charge_lattice_sum(lower.omega_charge, spin.pairs, cutoff=5)
            naive = cmath.sqrt(p_formula * theta_marked)
            points.append({"point_id": point["point_id"], "side": side,
                "q_values": q, "marked_spin": marked_spin.pairs, "charge_spin": spin.pairs,
                "period_branch": b_shift, "theta_translation_phase": xi,
                "theta_translation_complex_error": abs(period_ratio - 1),
                "charge_vs_geometric_period_residual": float(np.max(abs(
                    frame.omega_charge - geometry.omega - b_shift))),
                "mode_24_to_32_complex_error": abs(d_lower / (p_formula * theta_charge) - 1),
                "lattice_4_to_5_complex_error": abs(charge_lattice_sum(
                    frame.omega_charge, spin.pairs, cutoff=4) / theta_charge - 1),
                "naive_marked_principal_root": comparison(naive, majorana_formula),
                "corrected_principal_root": comparison(cmath.sqrt(dirac_formula), majorana_formula),
                "corrected_evaluator": production,
                "branch": branch_record, "spin_checks": spin_checks, "sweeps": sweeps})
        print("checked", point["point_id"], "norm and phase in both charts", flush=True)
    maxima = []
    for side, cutoff, factor in itertools.product(("source", "target"), LEVELS,
                                                 ("boson", "dirac", "majorana", "superfield_chiral_at_zero_charge")):
        selected = [s[factor] for p in points if p["side"] == side
                    for s in p["sweeps"] if s["level"] == cutoff]
        maxima.append({"side": side, "level": cutoff, "factor": factor,
                       **{key: max(r[key] for r in selected) for key in (
                           "norm_relative_error", "phase_absolute_error_radians", "complex_relative_error")}})
    if max(r["complex_relative_error"] for r in maxima if r["level"] == level) > 1e-10:
        raise ArithmeticError("the independent Fock sums have not converged")
    if max(s["unsquared_fock"]["complex_relative_error"] for p in points for s in p["spin_checks"]) > 1e-10:
        raise ArithmeticError("an additional even spin sector did not converge")
    # Repeat branch tracking more finely on representative source/target charts.
    radial_refinement = []
    for p in points[:2]:
        spin = SpinCharacteristic.from_pairs(p["charge_spin"])
        coarse, _ = radial_roots(p["q_values"], [spin], steps=48)
        fine, _ = radial_roots(p["q_values"], [spin], steps=96)
        radial_refinement.append(comparison(coarse[0], fine[0]))
    if max(r["complex_relative_error"] for r in radial_refinement) > 1e-12:
        raise ArithmeticError("radial continuation changed on path refinement")
    winding = ramond_winding_check(rr_coefficients)
    translation_check = translation_identity_check()
    dependencies = [Path(__file__), Path(__file__).with_name("audit_fixed_spin_free_q_expansion.py"),
                    Path(__file__).with_name("fixed_spin_free_plumbing.py"),
                    Path(__file__).with_name("physical_free_plumbing_resummation.py"),
                    Path(__file__).with_name("spin_structure.py"),
                    ROOT / "Code/theta_fermion_ccy/direct_fermion.py",
                    ROOT / "Code/genus_2_cross_channel/free_majorana_pair_of_pants.py",
                    ROOT / "Code/genus_2_cross_channel/free_boson_pair_of_pants.py",
                    ROOT / "Code/genus_2_cross_channel/free_boson_plumbing.py",
                    ROOT / "Code/genus_2_cross_channel/plumbing_algorithms.py"]
    result = {"schema": "free-bosonization-norm-phase-v1", "status": "passed",
              "config_path": str(config_path.resolve()),
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in dependencies},
              "levels": LEVELS, "continuous_charges": CONTINUOUS_CHARGES,
              "conventions": {"edge_order": "0,1,infinity", "propagation": "q^L0",
                  "NS_anchor": "M=1+...", "RR_anchor": "M=sqrt(2)*exp((log(q0)+log(q1))/16)*(1+...)",
                  "phase_metric": "arg(Fock/formula) in radians; no pointwise sign fitting",
                  "norm_metric": "abs(abs(Fock/formula)-1)",
                  "Ramond_Ward_frame": "rho=i^p_one*r/2^((g_one+g_zero)/2); mutually dual pants frames",
                  "full_nonchiral": "analytic continuous-charge Gaussian shared; no new integration test",
                  "scope": "same-chart free-CFT identities; no interacting or cross-chart phase claim"},
              "maxima": maxima, "points": points, "radial_refinement": radial_refinement,
              "translation_identity_check": translation_check,
              "ramond_eight_turn_loop": winding, "elapsed_seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(serializable(result), indent=2, allow_nan=False) + "\n")
    write_report(result, output)
    print(json.dumps({"status": "passed", "elapsed_seconds": result["elapsed_seconds"],
                      "highest_level": [r for r in maxima if r["level"] == level]}, indent=2), flush=True)


def write_report(result, output):
    level = max(LEVELS)
    rows = []
    for p in result["points"]:
        s = p["sweeps"][-1]
        m = s["majorana"]
        rows.append({"point": p["point_id"], "chart": p["side"], "level": level,
            "formula_real": m["reference"].real, "formula_imag": m["reference"].imag,
            "fock_real": m["value"].real, "fock_imag": m["value"].imag,
            "formula_norm": abs(m["reference"]), "fock_norm": abs(m["value"]),
            "formula_phase_radians": m["reference_phase_radians"],
            "fock_phase_radians": m["phase_radians"],
            "norm_relative_error": m["norm_relative_error"],
            "phase_residual_radians": m["phase_residual_radians"],
            "naive_phase_error_degrees": math.degrees(p["naive_marked_principal_root"]["phase_residual_radians"]),
            "free_Z_formula": s["Z_free_formula_shared_gaussian"],
            "free_Z_fock": s["Z_free_fock_shared_gaussian"], "free_Z_relative_error": s["Z_free_relative_error"]})
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Free partition function: magnitude and phase", "",
        "Fresh computation on ten saved surfaces, each in its NSRR source and all-NS target theta-pants chart. Bosonization and Fock sewing are compared at the same q, before taking absolute values. All propagators are q^L0; edge Casimir powers are consistently absent.", "",
        "## Main result", "",
        "The production fixed_spin_chiral_partition evaluator is now used by this audit. Its unsquared single-Majorana answer agrees with independent Fock sewing in both charts, with the branch fixed at degeneration. The source chart has a nontrivial theta-translation phase: omitting it creates a -pi/8 (-22.5 degree) Majorana phase error with exactly the same norm. The corrected principal root agrees with the continued root on these saved samples. A separate eight-turn Ramond path checks the production continuation against Fock sewing and an independent phase-unwrapping calculation.", "",
        "## Definitions and independent calculations", "",
        "Write P for the Heisenberg vacuum oscillator factor, and theta_c for the theta constant in the charge period basis. The complex Dirac answer is D=P theta_c and the single chiral Majorana is M=sqrt_cont(D). At fixed scalar charges a, the free-superfield block is F_X(a) M; after pairing chirality and integrating continuous scalar charges, Z_free=|P M|^2/sqrt(det(2 Im Omega)).", "",
        "The formula uses the boson Gaussian/Fredholm resummation for P, and theta on a period matrix computed independently by normalized-one-form collocation at the same q. The charge-period theta sum is a separate crosscheck. Direct scalar/Dirac sewing enumerates Heisenberg current partitions, inverse Gram norms, and Wick contractions. Direct NS Majorana sewing enumerates distinct half-integer Clifford modes and Pfaffian pants coefficients. Direct NSRR sewing enumerates integer Ramond modes, their two ground labels, and half-integer NS modes using the free-fermion contour Ward identities; no theta, period matrix, bosonization, or super-Virasoro recursion enters its coefficients.", "",
        "For NSRR, the reused Ward form is rho=i^(p_one) r/2^((g_one+g_zero)/2). In mutually dual pants spin frames the coefficient product is r^2/2^(g_one+g_zero). The ground sum and standard chiral Ramond normalization give M=sqrt(2) q0^(1/16) q1^(1/16) times one half of this state sum. Its first normalized terms are 1 + sqrt(q_infinity)/2 + (q0+q1)/8 + ... . This explicitly states the spin-frame convention being tested; it is not a universal replacement for an interacting NSRR sewing convention.", "",
        "## Phase of the marked-to-charge theta translation", "",
        "Characteristics below are binary bits. Let Omega_c=Omega_m+B and beta_c=beta_m-B alpha+diag(B) mod 2. Expanding every theta summand gives", "",
        r"\[\theta[\alpha|\beta_c](\Omega_m+B)=\xi_B\,\theta[\alpha|\beta_m](\Omega_m),\qquad \xi_B=\exp\!\left[i\pi\left(\frac{\alpha^T B\alpha}{4}+\frac{\alpha^T(\beta_c-\beta_m)}{2}\right)\right].\]", "",
        "The integer-charge dependent part of the exponent is an even integer. Thus this factor is determined algebraically, without fitting any numerical phase. On every saved source, alpha=(1,1), beta_m=beta_c=(0,0), and B=diag(0,1), giving xi_B=exp(i*pi/4). Therefore the correct same-frame formula is M=sqrt_cont(exp(i*pi/4) P theta_m), while sqrt(P theta_m) has a -pi/8 offset. On all saved all-NS targets alpha=(0,0), hence xi_B=1 after beta has been transported.", "",
        "The literature also fixes a twist-operator normalization: [Tuite–Zuevsky, Eq. (78)](https://arxiv.org/html/1007.5203#S5.E78) displays exp(-2*pi*i*alpha*beta) for half-characteristics. The present lattice convention is explicitly sum_a exp(i*pi*a*Omega*a+i*pi*a*beta_bits). The selected saved characteristics have zero alpha*beta, so that additional convention factor is one here. A torus-sewing oscillator factor cannot be transplanted into pants coordinates without its frame conversion.", "",
        "## Maximum errors over ten surfaces", "",
        "The norm error is | |Fock/formula| - 1 |. The phase error is |arg(Fock/formula)| in radians. Level L truncates the sum of oscillator levels; Ramond primary powers are retained exactly. The Ramond/NS Majorana truncations use their appropriate integer/half-integer oscillator levels.", "",
        "| Chart | Factor | L | Norm relative error | Phase error (rad) | Complex relative error |",
        "|---|---|---:|---:|---:|---:|"]
    for r in result["maxima"]:
        if r["factor"] == "superfield_chiral_at_zero_charge":
            continue
        lines.append(f"| {r['side']} | {r['factor']} | {r['level']} | {r['norm_relative_error']:.6e} | {r['phase_absolute_error_radians']:.6e} | {r['complex_relative_error']:.6e} |")
    lines += ["", "## Single-Majorana values at the highest cutoff", "",
        "| Point | Chart | Formula magnitude | Formula phase (rad) | Fock magnitude | Fock phase (rad) | Phase residual (rad) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['point']} | {r['chart']} | {r['formula_norm']:.12f} | {r['formula_phase_radians']:.12f} | {r['fock_norm']:.12f} | {r['fock_phase_radians']:.12f} | {r['phase_residual_radians']:.3e} |")
    winding = result["ramond_eight_turn_loop"]
    lines += ["", "## Continued Ramond path: a sign invisible after squaring", "",
        "Keep q1=0.017 and q_infinity=0.011, and wind q0=0.013 exp(i*t) eight times with continuous logarithms. In this q^L0 convention the Ramond primary q0^(1/16) gains exp(i*pi*k/8) after k turns; its square gains exp(i*pi*k/4). The integer-mode oscillator series returns at each turn. After eight turns D returns to its initial value and M changes sign. This is a path on the plumbing/spin lift; endpoint q alone does not specify it.", "",
        f"Maximum direct-Fock versus continued-root complex error on the sampled path: {winding['maximum_fock_vs_continued_complex_error']:.6e}. The direct series uses the continued Ramond primary and is never sign-adjusted against the theta answer.", "",
        "| Turns | Dirac return ratio | Continued Majorana return ratio | Principal-root phase minus continued phase (rad) |",
        "|---:|---:|---:|---:|"]
    d0, m0 = winding["rows"][0]["dirac"], winding["rows"][0]["continued_majorana"]
    for r in winding["rows"]:
        lines.append(f"| {r['turns']} | {r['dirac']/d0:.9f} | {r['continued_majorana']/m0:.9f} | {cmath.phase(r['principal_majorana']/r['continued_majorana']):.9f} |")
    lines += ["", "## Numerical controls and limits", "",
        f"- Maximum independently computed period residual: {max(p['charge_vs_geometric_period_residual'] for p in result['points']):.6e}.",
        f"- Maximum complex theta-translation error: {max(p['theta_translation_complex_error'] for p in result['points']):.6e}.",
        f"- Translation formula separately checked on all ten even characteristics and three integer shifts: {result['translation_identity_check']['maximum_complex_relative_error']:.6e} maximum complex relative error.",
        f"- Dirac mode-cutoff change 24 to 32: {max(p['mode_24_to_32_complex_error'] for p in result['points']):.6e}.",
        f"- Dirac lattice-cutoff change 4 to 5: {max(p['lattice_4_to_5_complex_error'] for p in result['points']):.6e}.",
        f"- Maximum full nonchiral free-Z relative error at L={level}: {max(r['free_Z_relative_error'] for r in rows):.6e} (shared analytic continuous-charge Gaussian).",
        "- All four all-NS beta choices and both even RR beta choices are checked at the ten corresponding q values. Odd-spin vacuum amplitudes vanish and have no phase to compare.",
        "- Radial branches are anchored at the vacuum and checked at 48 and 96 steps on representative charts. No endpoint sign is fitted to the direct answer.",
        "- The fully paired nonchiral Majorana factor is |M|^2. A global sign of M cancels there; the spin-dependent Arf/regulator convention is separate. The displayed chiral phases therefore cannot explain a discrepancy between correctly assembled nonchiral free denominators.",
        "- Raw source and target chiral values have different plumbing frames and homology markings. This audit compares the two methods within each chart; it does not supply a source-to-target modular/anomaly multiplier or certify an interacting Liouville comparison.", "",
        "## Reproduce", "", "```bash",
        "env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\",
        "  /private/tmp/type0b-nsrr-smoke-venv/bin/python Code/genus_2/audit_free_bosonization_phase.py",
        "```", "", "The optional figure can be reproduced with `python Code/genus_2/plot_free_bosonization_phase.py` in a Python environment containing matplotlib.", "",
        "[Machine-readable results](summary.json) · [Numerical values](comparison.csv)", "",
        "![Magnitude, phase, and continuation checks](norm_phase_comparison.png)", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "Data Set/nsrr_double_virasoro_N7_L5_20260911/config.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Data Set/free_bosonization_phase_20260915")
    args = parser.parse_args()
    run(args.config, args.output)
