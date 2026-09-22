#!/usr/bin/env python3
r"""Fixed-spin X + Majorana on a theta-plumbed surface, including NSRR.

Public puncture order is (0,1,infinity). Propagators are q**L0, not
q**(L0-c/24). The scalar has h(a)=a**2/2 and completeness da0 da1;
its connected target-space zero-mode volume is set to one.

Two physical Majoranas bosonize to the odd charge lattice Z. In a sector
alpha, charges on the first two cuts are n+alpha/2, and the infinity
charge is minus their sum. Thus alpha=(1,1) gives (R,R,NS), precisely
NSRR in the Human Note's reversed slot order. We sum charged Heisenberg
pants blocks, not the auxiliary fermion used by double Virasoro.

If F_X(a;q)=P(q) exp(i*pi*a.T*Omega_charge(q)*a), bosonization gives
Z_Dirac,chiral=P*theta[alpha,beta](Omega_charge) in the stated charge-sum
convention. In marked periods this is P*xi_B*theta_marked, where xi_B is
returned explicitly. One chiral Majorana is its continued square root;
fixed_spin_chiral_partition retains its phase and continuation data.
Consequently Z_Majorana,nonchiral=abs(P*theta), NOT abs(P*theta)**2.
The full answer is det(2 Im Omega_charge)**(-1/2)*abs(P)**3*abs(theta).
The factor 2 in the Gaussian follows from da and h=a**2/2, not fitting.

The marked period may differ by a symmetric integer B. This module
requires that branch explicitly and transports beta with its affine term.
It never uses the legacy filtered NS block as a fixed-spin determinant.

Bosonization reference: Tuite--Zuevsky, arXiv:1007.5203, equation (78).
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math

import numpy as np

from physical_free_plumbing_resummation import theta_charged_boson_resummation
from free_boson_pair_of_pants import integer_partitions, heisenberg_gram_norm
from free_boson_plumbing import riemann_theta_constant_genus2
from spin_structure import SpinCharacteristic, ThetaLogBranch, ThetaSpinFrame


def _q_values(q_values):
    q = tuple(complex(x) for x in q_values)
    if len(q) != 3 or any(not np.isfinite(z) or not 0 < abs(z) < 1 for z in q):
        raise ValueError("three finite, nonzero plumbing q with |q|<1 are required")
    return q


def _characteristic(characteristic):
    value = np.asarray(characteristic)
    if value.shape != (2, 2) or not np.all((value == 0) | (value == 1)):
        raise ValueError("characteristic must be two pairs of binary bits")
    return tuple(tuple(int(x) for x in row) for row in value)


@dataclass(frozen=True)
class ChargedFrame:
    q_values: tuple[complex, complex, complex]
    max_mode: int
    boson_chiral: complex
    omega_charge: np.ndarray
    loop_gaussian: float


def charged_frame(q_values, *, max_mode=24):
    """Extract the COMPLEX period from charge exponents, without log(F).

    Principal log(q_e) is part of the specified plumbing lift. Keeping it
    separate from the Schur exponent avoids losing integer period branches.
    """
    q = _q_values(q_values)
    if int(max_mode) != max_mode or max_mode < 1:
        raise ValueError("positive integer max_mode required")
    logs = np.log(np.asarray(q))
    values = []
    for a, b in ((1, 0), (0, 1), (1, 1)):
        block = theta_charged_boson_resummation(
            q, alpha_zero=a, alpha_one=b, max_mode=int(max_mode))
        values.append(.5*(a*a*logs[0] + b*b*logs[1] + (a+b)**2*logs[2])
                      + block.charged_exponent)
    u, v, w = values
    cross = (w-u-v)/2
    omega = np.asarray([[u, cross], [cross, v]])/(1j*math.pi)
    if np.linalg.eigvalsh(omega.imag)[0] <= 0:
        raise ArithmeticError("charged sewing has nonpositive Im Omega")
    gaussian = float(np.linalg.det(2*omega.imag)**(-.5))
    return ChargedFrame(q, int(max_mode), block.vacuum_chiral, omega, gaussian)


def characteristic_in_charge_frame(characteristic, period_branch):
    """Omega_charge=Omega_marked+B, with a fixed symmetric integer B."""
    return SpinCharacteristic.from_pairs(_characteristic(characteristic)).charge_frame(period_branch).pairs


def theta_translation_phase(characteristic, period_branch):
    r"""Return xi_B in theta_charge(Omega+B)=xi_B*theta_marked(Omega).

    Binary characteristics satisfy beta_c=beta-B alpha+diag(B) mod 2.
    The constant phase, including reduction of beta_c to binary bits, is
    exp(i*pi*(alpha.T B alpha/4 + alpha.T (beta_c-beta)/2)).
    """
    spin = (characteristic if isinstance(characteristic, SpinCharacteristic)
            else SpinCharacteristic.from_pairs(_characteristic(characteristic)))
    charge_spin = spin.charge_frame(period_branch)  # Validates B before casting.
    a = np.asarray(spin.alpha)
    b = np.asarray(period_branch, dtype=int)
    exponent = a @ b @ a / 4 + a @ (np.asarray(charge_spin.beta) - spin.beta) / 2
    return cmath.exp(1j*math.pi*exponent)


def charge_grid(alpha, cutoff):
    if int(cutoff) != cutoff or cutoff < 1:
        raise ValueError("positive integer lattice cutoff required")
    axes = [np.arange(-cutoff, cutoff+1-bit, dtype=float)+bit/2 for bit in alpha]
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def charge_lattice_sum(omega_charge, characteristic, *, cutoff=5):
    """Direct sum over conserved integer/half-integer loop charges."""
    alpha, beta = _characteristic(characteristic)
    charges = charge_grid(alpha, cutoff)
    exponents = (1j*math.pi*np.einsum("ni,ij,nj->n", charges, omega_charge, charges)
                 + 1j*math.pi*(charges@np.asarray(beta)))
    # Symmetric charge boxes also cancel odd-spin zero modes.
    return complex(np.sum(np.exp(exponents)))


def fixed_spin_partition(q_values, omega_marked, characteristic, *, period_branch,
                         max_mode=24, lattice_cutoff=5, period_tolerance=1e-8):
    """Return nonchiral free factors and phase-correct marked-theta diagnostics.

    No global local-coordinate multiplier is imported from a different
    chart. P, the charge quadratic form, and the Gaussian all come from
    these same three plumbing parameters. Use fixed_spin_chiral_partition
    for the unsquared, complex single-Majorana amplitude.
    """
    frame = charged_frame(q_values, max_mode=max_mode)
    marked = np.asarray(omega_marked, dtype=complex)
    if (marked.shape != (2, 2) or not np.all(np.isfinite(marked))
            or np.max(abs(marked-marked.T)) > 1e-10
            or np.linalg.eigvalsh(marked.imag)[0] <= 0):
        raise ValueError("marked Omega must be symmetric with positive imaginary part")
    spin = _characteristic(characteristic)
    spin_frame = ThetaSpinFrame(SpinCharacteristic.from_pairs(spin), frame.q_values, period_branch)
    charge_spin = spin_frame.charge_spin.pairs
    try:
        error = spin_frame.validate_periods(marked, frame.omega_charge, tolerance=period_tolerance)
    except ValueError as exc:
        raise ArithmeticError(str(exc)) from exc
    lattice = charge_lattice_sum(frame.omega_charge, charge_spin, cutoff=lattice_cutoff)
    previous = charge_lattice_sum(frame.omega_charge, charge_spin,
                                 cutoff=max(1, lattice_cutoff-1))
    theta = riemann_theta_constant_genus2(marked, spin, tol=1e-15)
    theta_phase = theta_translation_phase(spin, period_branch)
    translated_theta = theta_phase*theta
    boson = frame.loop_gaussian*abs(frame.boson_chiral)**2
    fermion = abs(frame.boson_chiral*lattice)
    odd = sum(a*b for a, b in zip(*spin)) % 2
    if odd:
        if abs(lattice) > 1e-11:
            raise ArithmeticError("odd spin did not vanish")
        fermion = 0.0
    return {
        "q_values": frame.q_values, "characteristic_marked": spin,
        "spin_frame": spin_frame.record(), "spin_frame_digest": spin_frame.digest,
        "characteristic_charge": charge_spin, "period_branch": np.asarray(period_branch).tolist(),
        "omega_charge": frame.omega_charge.tolist(), "period_residual": error,
        "max_mode": max_mode, "lattice_cutoff": lattice_cutoff,
        "boson_chiral": frame.boson_chiral, "loop_gaussian": frame.loop_gaussian,
        "dirac_charge_sum": lattice, "marked_theta": complex(theta),
        "theta_translation_phase": theta_phase,
        "marked_theta_in_charge_frame": complex(translated_theta),
        "dirac_chiral": complex(frame.boson_chiral*lattice) if not odd else 0j,
        "dirac_chiral_from_marked_theta": complex(frame.boson_chiral*translated_theta) if not odd else 0j,
        "theta_complex_relative_error": float(abs(lattice-translated_theta)/max(1e-300, abs(translated_theta))) if not odd else None,
        "theta_absolute_relative_error": float(abs(abs(lattice)-abs(theta))/max(1e-300, abs(theta))) if not odd else None,
        "lattice_relative_change": float(abs(lattice-previous)/max(1e-300, abs(lattice))) if not odd else None,
        "Z_boson": float(boson), "Z_majorana": float(fermion),
        "Z_free": float(boson*fermion), "has_fermion_zero_mode": bool(odd),
    }


def _continue_root(squared, previous_root, maximum_phase_step):
    """Transport a square root over one sufficiently small, nonzero step."""
    if not np.isfinite(squared) or not np.isfinite(previous_root) or not squared or not previous_root:
        raise ArithmeticError("cannot continue a square root through a zero or nonfinite value")
    ratio = squared/(previous_root*previous_root)
    if not np.isfinite(ratio) or not ratio:
        raise ArithmeticError("square-root continuation underflowed or overflowed")
    phase_step = abs(cmath.phase(ratio))
    if phase_step >= maximum_phase_step:
        raise ValueError("sample the chiral path more finely before continuing its square root")
    # The ratio is near the positive real axis; this transports the old sign.
    return previous_root*cmath.sqrt(ratio), phase_step


def _majorana_ground(q, charge_spin):
    if charge_spin.alpha == (0, 0):
        return 1+0j
    ramond_logs = sum(cmath.log(z) for z, sector in zip(q, charge_spin.theta_edge_sectors)
                      if sector == "R")
    return math.sqrt(2)*cmath.exp(ramond_logs/16)


def _radial_chiral_anchor(q, charge_spin, end_frame, *, lattice_cutoff,
                         radial_steps, maximum_phase_step):
    """Fix the sign at degeneration, then transport it without a Fock fit."""
    p = 1+0j
    majorana = None
    maximum_step = 0.
    anchor_error = None
    for index, scale in enumerate(np.geomspace(1e-6, 1., radial_steps+1)):
        qs = tuple(scale*z for z in q)
        frame = end_frame if index == radial_steps else charged_frame(qs, max_mode=end_frame.max_mode)
        p, step = _continue_root(frame.boson_chiral**2, p, maximum_phase_step)
        maximum_step = max(maximum_step, step)
        if charge_spin.arf:
            majorana = 0j
            continue
        squared = p*charge_lattice_sum(frame.omega_charge, charge_spin.pairs, cutoff=lattice_cutoff)
        if index == 0:
            ground = _majorana_ground(qs, charge_spin)
            normalized = squared/ground**2
            anchor_error = abs(normalized-1)
            if not np.isfinite(normalized) or anchor_error > .01:
                raise ArithmeticError("chiral anchor is not in the normalized vacuum neighborhood")
            majorana = ground*cmath.sqrt(normalized)
        else:
            majorana, step = _continue_root(squared, majorana, maximum_phase_step)
            maximum_step = max(maximum_step, step)
    return p, majorana, {"anchor": "q(t)=t*q, t=1e-6; positive normalized ground coefficient",
                         "radial_steps": radial_steps, "anchor_normalized_squared_error": anchor_error,
                         "maximum_squared_phase_step_radians": maximum_step}


def fixed_spin_chiral_partition(q_values, omega_marked, characteristic, *, period_branch,
                                max_mode=24, lattice_cutoff=5, period_tolerance=1e-8,
                                previous=None, radial_steps=48, maximum_phase_step=math.pi/4):
    r"""Return the complex single-Majorana and free-superfield oscillator factors.

    The initial sample is normalized at degeneration: M_NS=1+... and
    M_RR=sqrt(2)*prod_{Ramond edges} q_e^(1/16)*(1+...). Subsequent samples
    pass the preceding return dictionary as ``previous`` to transport both
    the plumbing logs and the square-root sign. Inputs must describe one
    continuously marked chart and spin structure; a modular chart change
    requires its separate frame multiplier. Sample the path finely enough
    that each q phase and squared-amplitude phase changes by less than
    maximum_phase_step. Endpoint data alone cannot recover a path's winding.

    Input period_branch relates the PRINCIPAL-log charge period to the
    supplied marked period. Returned period_branch includes the continued
    log shift. For the saved NSRR chart the initial theta phase is exp(i*pi/4),
    hence the single-Majorana correction is exp(i*pi/8).

    ``majorana_chiral`` is unsquared. ``superfield_chiral_oscillator`` is
    P*M, before any continuous scalar-charge factor. Odd-spin vacuum
    amplitudes are zero and ``majorana_phase_radians`` is None.
    """
    if not np.isfinite(maximum_phase_step) or not 0 < maximum_phase_step < math.pi:
        raise ValueError("maximum_phase_step must lie strictly between zero and pi")
    if int(radial_steps) != radial_steps or radial_steps < 1:
        raise ValueError("positive integer radial_steps required")
    base = fixed_spin_partition(q_values, omega_marked, characteristic,
        period_branch=period_branch, max_mode=max_mode, lattice_cutoff=lattice_cutoff,
        period_tolerance=period_tolerance)
    q = base["q_values"]
    marked_spin = SpinCharacteristic.from_pairs(base["characteristic_marked"])
    frame = ChargedFrame(q, int(max_mode), base["boson_chiral"],
                         np.asarray(base["omega_charge"]), base["loop_gaussian"])
    if previous is None:
        log_branch = ThetaLogBranch(q)
        continuous_branch = np.asarray(period_branch, dtype=int)
        charge_spin = marked_spin.charge_frame(continuous_branch)
        p, majorana, branch_record = _radial_chiral_anchor(
            q, charge_spin, frame, lattice_cutoff=lattice_cutoff,
            radial_steps=int(radial_steps), maximum_phase_step=maximum_phase_step)
    else:
        if not isinstance(previous, dict) or previous.get("schema") != "fixed-spin-chiral-free-v1":
            raise ValueError("previous must be a result from fixed_spin_chiral_partition")
        if SpinCharacteristic.from_pairs(previous["characteristic_marked"]) != marked_spin:
            raise ValueError("chiral continuation requires the same marked spin structure")
        old_branch = ThetaLogBranch(previous["q_values"], previous["chiral_branch"]["log_windings"])
        log_branch = old_branch.advance(q, maximum_phase_step=maximum_phase_step)
        continuous_branch = np.asarray(period_branch, dtype=int) + np.asarray(log_branch.period_shift)
        if not np.array_equal(continuous_branch, previous["period_branch"]):
            raise ValueError("period marking changed along the chiral path; preserve the continued homology frame")
        charge_spin = marked_spin.charge_frame(continuous_branch)
        p, step = _continue_root(frame.boson_chiral**2, previous["boson_chiral"], maximum_phase_step)
        branch_record = dict(previous["chiral_branch"])
        branch_record["maximum_squared_phase_step_radians"] = max(
            branch_record["maximum_squared_phase_step_radians"], step)
        if marked_spin.arf:
            majorana = 0j
        else:
            squared = p*charge_lattice_sum(frame.omega_charge + np.asarray(log_branch.period_shift),
                                           charge_spin.pairs, cutoff=lattice_cutoff)
            majorana, step = _continue_root(squared, previous["majorana_chiral"], maximum_phase_step)
            branch_record["maximum_squared_phase_step_radians"] = max(
                branch_record["maximum_squared_phase_step_radians"], step)
    omega_continued = frame.omega_charge + np.asarray(log_branch.period_shift)
    theta = charge_lattice_sum(omega_continued, charge_spin.pairs, cutoff=lattice_cutoff)
    phase = theta_translation_phase(marked_spin, continuous_branch)
    translated = phase*base["marked_theta"]
    dirac = p*theta if not marked_spin.arf else 0j
    branch_record.update(log_branch.record())
    return {
        "schema": "fixed-spin-chiral-free-v1", "q_values": q,
        "characteristic_marked": marked_spin.pairs, "characteristic_charge": charge_spin.pairs,
        "principal_period_branch": base["period_branch"], "period_branch": continuous_branch.tolist(),
        "omega_charge": omega_continued.tolist(), "marked_theta": base["marked_theta"],
        "theta_translation_phase": phase, "dirac_charge_sum": theta,
        "theta_complex_relative_error": float(abs(theta-translated)/max(1e-300, abs(translated))) if not marked_spin.arf else None,
        "boson_chiral": p, "dirac_chiral": dirac, "majorana_chiral": majorana,
        "dirac_chiral_from_marked_theta": p*translated if not marked_spin.arf else 0j,
        "majorana_phase_radians": cmath.phase(majorana) if majorana else None,
        "superfield_chiral_oscillator": p*majorana,
        "Z_majorana_nonchiral": base["Z_majorana"], "Z_free_nonchiral": base["Z_free"],
        "has_fermion_zero_mode": bool(marked_spin.arf),
        "max_mode": int(max_mode), "lattice_cutoff": int(lattice_cutoff),
        "chiral_branch": branch_record,
    }


def direct_charged_fock_sum(q_values, characteristic_charge, *, total_level=6,
                            lattice_cutoff=4):
    r"""Independent finite Fock sewing for the bosonized Dirac fermion.

    Enumerates current partitions, inverse diagonal Gram norms, and Wick
    contractions including single-current/exponential contractions. Does
    NOT use a determinant, charged_frame, a period matrix, or theta.
    Current-pair contractions are unnormalized, as are the Fock states.
    Fractional Ramond ground weights stay in the primary prefactor.
    """
    q = _q_values(q_values)
    spin = _characteristic(characteristic_charge)
    if int(total_level) != total_level or total_level < 0:
        raise ValueError("nonnegative integer total Fock level required")
    charges = charge_grid(spin[0], lattice_cutoff)
    a, b = charges.T
    size = len(charges)

    def single(field):
        slot, mode = field
        return b if slot == 0 else ((-1)**(mode-1)*a if slot == 1 else -b)

    def pair(left, right):
        (s, m), (t, n) = left, right
        if s == t:
            return 0
        if (s, t) == (0, 1):
            return m*math.comb(m-1, n-1) if m >= n else 0
        if (s, t) == (0, 2):
            return m if m == n else 0
        if (s, t) == (1, 2):
            return (-1)**(m-1)*n*math.comb(n+m-1, m-1)
        raise AssertionError("current slots not ordered")

    @lru_cache(maxsize=None)
    def wick(fields):
        if not fields:
            return np.ones(size)
        first, rest = fields[0], fields[1:]
        value = single(first)*wick(rest)
        for j, other in enumerate(rest):
            contraction = pair(first, other)
            if contraction:
                value = value + contraction*wick(rest[:j]+rest[j+1:])
        return value

    oscillators = np.zeros(size, dtype=complex)
    q_slots = (q[2], q[1], q[0])
    count = 0
    for l_inf in range(total_level+1):
        for l_one in range(total_level+1-l_inf):
            for l_zero in range(total_level+1-l_inf-l_one):
                factor = q_slots[0]**l_inf*q_slots[1]**l_one*q_slots[2]**l_zero
                for states in itertools.product(*(integer_partitions(n) for n in (l_inf, l_one, l_zero))):
                    fields = tuple((s, m) for s, state in enumerate(states) for m in state)
                    rho = wick(fields)
                    norm = math.prod(heisenberg_gram_norm(state) for state in states)
                    oscillators += factor*(rho*rho)/norm
                    count += 1
    logs = np.log(np.asarray(q))
    primary = np.exp(.5*(a*a*logs[0]+b*b*logs[1]+(a+b)**2*logs[2])
                     + 1j*math.pi*(charges@np.asarray(spin[1])))
    value = complex(np.sum(primary*oscillators))
    return {"dirac_chiral": value, "total_oscillator_level": total_level,
            "lattice_cutoff": lattice_cutoff, "fock_triples": count,
            "charge_pairs": size}
