#!/usr/bin/env python3
r"""Fixed-spin X + Majorana from a direct Schottky product and bosonization.

P(q) = product_[gamma primitive modulo conjugacy and inversion]
                  product_(m >= 1) (1-k_gamma**m)**(-1)

Z_free = det(2 Im Omega_marked)**(-1/2) |P|**3 |theta_delta(Omega_marked)|.

The marked period and its characteristic must be supplied in the same
homology basis. The caller is responsible for using a validated q-to-period
map; this evaluator does not reconstruct the period from charged blocks.
The explicit integer period branch records the corresponding charge-frame
spin. Evaluation uses the marked pair directly, so it needs no eta lookup.

Conventions: geometry edges (0,1,infinity), q**L0 propagation, h(a)=a**2/2,
measure da0 da1, and unit connected scalar zero-mode volume. Fock states and
finite total-q-degree truncations do not enter this calculation.
"""
from __future__ import annotations

import cmath
from functools import lru_cache
import math

import numpy as np

from fixed_spin_free_plumbing import charge_lattice_sum
from free_boson_plumbing import free_boson_chiral_log_factor
from genus2_vacuum_blocks import primitive_conjugacy_words, word_multiplier
from plumbing_algorithms import generators_for_theta
from spin_structure import SpinCharacteristic, ThetaSpinFrame


@lru_cache(maxsize=None)
def primitive_words(max_word_length):
    """Cache the moduli-independent enumeration, as immutable words."""
    return tuple(primitive_conjugacy_words(2, max_word_length))


def fixed_spin_partition_schottky(q_values, omega_marked, characteristic, *,
                                  period_branch, max_word_length=8, max_mode=50,
                                  lattice_cutoff=5, tolerance=1e-15):
    """Evaluate both NSRR and all-NS free denominators by the same formula.

    The returned inner-product tail estimate excludes omitted primitive
    words. Inspect successive word shells to assess that separate cutoff.
    The word and multiplier-mode cutoffs are unrelated to block order.
    """
    for value in (max_word_length, max_mode, lattice_cutoff):
        if int(value) != value or value < 1:
            raise ValueError("positive integer truncation cutoffs required")
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie between zero and one")
    spin_frame = ThetaSpinFrame(SpinCharacteristic.from_pairs(characteristic),
                               tuple(q_values), period_branch)
    omega = np.asarray(omega_marked, dtype=complex)
    if (omega.shape != (2, 2) or not np.all(np.isfinite(omega))
            or np.max(abs(omega - omega.T)) > 1e-10
            or np.linalg.eigvalsh(omega.imag)[0] <= 0):
        raise ValueError("marked period must be symmetric with positive imaginary part")
    generators = generators_for_theta(*spin_frame.q_values)
    shell_terms = {length: [] for length in range(1, max_word_length + 1)}
    tails = []
    for word in primitive_words(max_word_length):
        multiplier = word_multiplier(generators, word)
        value, tail = free_boson_chiral_log_factor(
            multiplier, max_mode=max_mode, tolerance=tolerance)
        shell_terms[len(word)].append(value)
        tails.append(tail)
    shells = {length: complex(math.fsum(z.real for z in values),
                              math.fsum(z.imag for z in values))
              for length, values in shell_terms.items()}
    log_product = complex(math.fsum(z.real for z in shells.values()),
                          math.fsum(z.imag for z in shells.values()))
    oscillator = cmath.exp(log_product)
    theta = charge_lattice_sum(omega, spin_frame.marked_spin.pairs, cutoff=lattice_cutoff)
    gaussian = float(np.linalg.det(2 * omega.imag)**(-.5))
    boson = gaussian * abs(oscillator)**2
    majorana = abs(oscillator * theta)
    if spin_frame.marked_spin.arf:
        if abs(theta) > 1e-11:
            raise ArithmeticError("odd-spin zero mode did not vanish")
        majorana = 0.0
    return {
        "method": "primitive Schottky product plus marked-spin bosonization",
        "q_values": spin_frame.q_values,
        "omega_marked": omega.tolist(),
        "spin_frame": spin_frame.record(), "spin_frame_digest": spin_frame.digest,
        "max_word_length": max_word_length, "max_mode": max_mode,
        "lattice_cutoff": lattice_cutoff, "tolerance": tolerance,
        "primitive_count": sum(map(len, shell_terms.values())),
        "primitive_count_by_length": {length: len(values) for length, values in shell_terms.items()},
        "chiral_log_product_by_word_length": shells,
        "omitted_multiplier_mode_tail_estimate": math.fsum(tails),
        "boson_chiral": oscillator, "theta_marked": theta,
        "loop_gaussian": gaussian,
        "Z_boson": boson, "Z_majorana": majorana, "Z_free": boson * majorana,
    }
