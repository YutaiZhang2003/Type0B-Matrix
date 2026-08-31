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
Z_Dirac,chiral=P*theta[alpha,beta](Omega_charge), up to a unit phase.
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
    alpha, beta = _characteristic(characteristic)
    branch = np.asarray(period_branch)
    if (branch.shape != (2, 2) or not np.all(np.isfinite(branch))
            or not np.array_equal(branch, branch.T)
            or not np.array_equal(branch, np.rint(branch))):
        raise ValueError("period branch must be a symmetric integer 2x2 matrix")
    branch = branch.astype(int)
    shifted = (np.asarray(beta)-branch@np.asarray(alpha)+np.diag(branch)) % 2
    return alpha, tuple(int(x) for x in shifted)


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
    """Return the free partition and independent marked-theta diagnostics.

    No global local-coordinate multiplier is imported from a different
    chart. P, the charge quadratic form, and the Gaussian all come from
    these same three plumbing parameters.
    """
    frame = charged_frame(q_values, max_mode=max_mode)
    marked = np.asarray(omega_marked, dtype=complex)
    if (marked.shape != (2, 2) or not np.all(np.isfinite(marked))
            or np.max(abs(marked-marked.T)) > 1e-10
            or np.linalg.eigvalsh(marked.imag)[0] <= 0):
        raise ValueError("marked Omega must be symmetric with positive imaginary part")
    spin = _characteristic(characteristic)
    charge_spin = characteristic_in_charge_frame(spin, period_branch)
    error = float(np.max(abs(frame.omega_charge-marked-np.asarray(period_branch))))
    if error > period_tolerance:
        raise ArithmeticError(f"charged and marked periods disagree: {error:.6g}")
    lattice = charge_lattice_sum(frame.omega_charge, charge_spin, cutoff=lattice_cutoff)
    previous = charge_lattice_sum(frame.omega_charge, charge_spin,
                                 cutoff=max(1, lattice_cutoff-1))
    theta = riemann_theta_constant_genus2(marked, spin, tol=1e-15)
    boson = frame.loop_gaussian*abs(frame.boson_chiral)**2
    fermion = abs(frame.boson_chiral*lattice)
    odd = sum(a*b for a, b in zip(*spin)) % 2
    if odd:
        if abs(lattice) > 1e-11:
            raise ArithmeticError("odd spin did not vanish")
        fermion = 0.0
    return {
        "q_values": frame.q_values, "characteristic_marked": spin,
        "characteristic_charge": charge_spin, "period_branch": np.asarray(period_branch).tolist(),
        "omega_charge": frame.omega_charge.tolist(), "period_residual": error,
        "max_mode": max_mode, "lattice_cutoff": lattice_cutoff,
        "boson_chiral": frame.boson_chiral, "loop_gaussian": frame.loop_gaussian,
        "dirac_charge_sum": lattice, "marked_theta": complex(theta),
        "theta_absolute_relative_error": float(abs(abs(lattice)-abs(theta))/max(1e-300, abs(theta))) if not odd else None,
        "lattice_relative_change": float(abs(lattice-previous)/max(1e-300, abs(lattice))) if not odd else None,
        "Z_boson": float(boson), "Z_majorana": float(fermion),
        "Z_free": float(boson*fermion), "has_fermion_zero_mode": bool(odd),
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
