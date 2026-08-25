#!/usr/bin/env python3
"""Exact mixed-sheet NS-primary--R--R contraction over ``GF(p)``.

This module is the first executable piece of glue between the reflection
intertwiner and the ground-resolved free-fermion backend.  It deliberately
starts with the zero-screening chart

    (n_1,n_2,n_3) = (0,n,-n),

because there the Heisenberg part is completely transparent and the result
can be checked against every stored Ward form without invoking a Selberg
identity.  No super-Virasoro descendant or branching state is constructed.

After reflection, a physical endpoint is a sparse sum of ordinary-chart
Fock states ``c_-lambda psi_-mu |g>``.  With the second vertex at one and no
screenings, commuting a bosonic creator through its exponential gives

    c_-m -> i (Q/2 + P_2),

independently of ``m``.  The physical fermions are contracted by the native
ground-resolved Pfaffian of this Coulomb chart.  To compare with the stored
grid through ``W_5/4``, the auxiliary fermions use the explicitly labelled
``fermion_value_virasoro`` fallback.  This is the corrected Virasoro-primary
sewing convention; it is distinct from the native free-Majorana spin-OPE
functional on some raw mode-string labels.  Reflection and the final scalar
assembly are done in a prime field; the fallback is evaluated exactly over
``Q(i,sqrt(2))`` and then embedded in that field.  Repetition at several
primes followed by CRT therefore loses no numerical accuracy.

The general-screening extension needs only the replacement

    c_-m -> i (Q/2 + P_2 + b sum_j t_j**(-m)),

before the existing symmetric-polynomial/Selberg extractor is called.  A
public function for this insertion is supplied below; routing a sum of
reflected Fock monomials through the bounded-width extractor is kept as the
remaining interface rather than hidden behind a purported scalar
``reflected covariance``.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import itertools
import sys
import time

import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
CHI_DIR = ROOT / "python 2" / "nsrr_chi_branching"
GRID_DIR = ROOT / "python 2" / "ramond_three_point_grid"
for directory in (CHI_DIR, GRID_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import nsrr_chi_formula as chi  # noqa: E402
import compute_grid as grid  # noqa: E402

from ..pfaffian.core import pfaffian
from ..profile import modular_transition as modular
from .intertwiner_recurrence import (
    reflect_fock_expression_mod,
    reflection_blocks_mod,
)


I = sp.I
SQRT2 = sp.sqrt(2)
@lru_cache(None)
def ordinary_eta_minus_fock_majorana_value(
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Physical Majorana form for the ordinary ``eta=-, f=0`` chart.

    The ground matrix must be changed to the Fock frame *before* a fermion
    flips a ground index.  Multiplying an SCblock ground matrix by a phase
    only after the Wick contraction is wrong and is precisely the mistake
    that makes the first crossed polynomial ``H`` disappear.

    In the repository's radial spin frame the Fock ground matrix is

        diag(1,i).

    Conditional on the second ground index ``g_2``, a physical fermion at
    the puncture one carries the local phase ``i(2 g_2-1)`` in an even Wick
    pair; a zero-leg fermion carries phase one.  Odd matrix elements use
    the usual one-point kernel and the same conditional covariance.  This
    is the two-spin Gaussian functional obtained from the ordinary
    ``E^(Q/2+P_2)`` Coulomb intertwiner.  The complementary ``eta=+`` form
    uses ``E^(Q/2-P_2)`` and the other charge-neutral chart; it is not
    silently folded into this callback.
    """

    second_modes = tuple(second_modes)
    third_modes = tuple(third_modes)
    second_ground = int(second_ground)
    third_ground = int(third_ground)
    fields = tuple((2, mode) for mode in second_modes) + tuple(
        (3, mode) for mode in third_modes
    )
    size = len(fields)
    ground_matrix = sp.diag(1, I)
    one_phase = I * (2 * second_ground - 1)

    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    for left in range(size):
        leg_left, mode_left = fields[left]
        phase_left = one_phase if leg_left == 2 else 1
        for right in range(left + 1, size):
            leg_right, mode_right = fields[right]
            phase_right = one_phase if leg_right == 2 else 1
            value = (
                phase_left
                * phase_right
                * grid.fermion_pair_coefficient(
                    leg_left, mode_left, leg_right, mode_right
                )
            )
            covariance[left][right] = value
            covariance[right][left] = -value

    if size % 2 == 0:
        return sp.factor(
            ground_matrix[second_ground, third_ground]
            * pfaffian(covariance)
        )

    mean = []
    for leg, mode in fields:
        if leg == 2:
            ground_value = ground_matrix[1 - second_ground, third_ground]
        else:
            ground_value = ground_matrix[second_ground, 1 - third_ground]
        mean.append(
            sp.factor(grid.fermion_one_coefficient(leg, mode) * ground_value)
        )
    augmented = [row + [mean[index]] for index, row in enumerate(covariance)]
    augmented.append([-value for value in mean] + [sp.Integer(0)])
    return sp.factor(pfaffian(augmented))


def quadratic_mod(expression, root_i, root_two, prime):
    """Embed an element of ``Q(i,sqrt(2))`` in ``GF(prime)``."""

    components = grid.quadratic_number_components(sp.cancel(expression))
    answer = 0
    multipliers = (1, root_two, root_i, root_i * root_two)
    for coefficient, multiplier in zip(components, multipliers):
        answer += modular.rational_mod(coefficient, prime) * multiplier
    return int(answer) % int(prime)


def heisenberg_zero_leg_factor(
    boson_partition,
    q_value,
    p2_value,
    root_i,
    prime,
    *,
    b_value=None,
    screenings=(),
):
    """Wick factor for reflected zero-leg bosonic creators.

    Arguments are residues in ``GF(prime)``.  If screenings are present,
    their coordinates and ``b_value`` must also be residues.  The formula
    follows directly from ``[c_m,c_n]=m delta_(m+n,0)`` and the ordered
    exponential of the 2013 free-field representation.
    """

    prime = int(prime)
    half = pow(2, -1, prime)
    alpha2 = (int(q_value) * half + int(p2_value)) % prime
    points = tuple(int(value) % prime for value in screenings)
    if points and b_value is None:
        raise ValueError("b_value is required when screenings are present")
    answer = 1
    for mode in boson_partition:
        current = alpha2
        if points:
            power_sum = sum(
                pow(point, -int(mode), prime) for point in points
            ) % prime
            current = (current + int(b_value) * power_sum) % prime
        answer = answer * int(root_i) * current % prime
    return answer


def _reflected_third_sectors_mod(
    branch_label,
    parity,
    blocks,
    root_i,
    root_two,
    prime,
):
    """Reflect each physical endpoint, retaining the auxiliary endpoint."""

    grouped = defaultdict(dict)
    for state, coefficient in chi.ramond_fock_paths(branch_label, parity):
        auxiliary_modes, auxiliary_ground, physical_modes, physical_ground = state
        physical_state = ((), physical_modes, physical_ground)
        bucket = grouped[(auxiliary_modes, auxiliary_ground)]
        bucket[physical_state] = (
            bucket.get(physical_state, 0)
            + quadratic_mod(coefficient, root_i, root_two, prime)
        ) % prime

    answer = []
    for auxiliary_state, physical_expression in grouped.items():
        reflected = reflect_fock_expression_mod(
            physical_expression, blocks, prime
        )
        for physical_state, coefficient in reflected.items():
            if coefficient:
                answer.append(
                    (
                        auxiliary_state[0],
                        auxiliary_state[1],
                        physical_state[0],
                        physical_state[1],
                        physical_state[2],
                        int(coefficient) % prime,
                    )
                )
    return tuple(answer)


@lru_cache(None)
def _reflection_blocks_cached(
    maximum_level,
    q_mod,
    p3_mod,
    root_i,
    root_two,
    prime,
):
    return reflection_blocks_mod(
        maximum_level,
        q_mod,
        p3_mod,
        root_i,
        root_two,
        prime,
    )


def mixed_sheet_zero_screening_mod(
    branch_magnitude,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    q_value,
    p2_value,
    p3_value,
    prime=1_000_033,
):
    """Return ``rhohat(1,W_n,W_-n)`` on its zero-screening chart.

    ``q_value``, ``p2_value`` and ``p3_value`` are ordinary rational
    numbers, not residues.  The required neutrality relation is

        P_1 = -Q/2-P_2-P_3.

    The result is the chosen embedding in ``GF(prime)``.  Changing either
    square root returned by :func:`modular.roots` gives the other algebraic
    embeddings used by CRT reconstruction.
    """

    n = abs(sp.Rational(branch_magnitude))
    epsilon2, epsilon3 = int(epsilon2), int(epsilon3)
    form_parity, eta = int(form_parity), int(eta)
    if form_parity != 0 or eta != -1:
        raise NotImplementedError(
            "this Coulomb chart realizes exactly f=0, eta=-; use the "
            "complementary Q/2-P2 chart for eta=+"
        )
    prime = int(prime)
    root_i, root_two = modular.roots(prime)
    q_mod = modular.rational_mod(sp.Rational(q_value), prime)
    p2_mod = modular.rational_mod(sp.Rational(p2_value), prime)
    p3_mod = modular.rational_mod(sp.Rational(p3_value), prime)

    maximum_mode = chi.ramond_mode_count(-n)
    maximum_level = maximum_mode * (maximum_mode + 1) // 2
    blocks = _reflection_blocks_cached(
        maximum_level,
        q_mod,
        p3_mod,
        root_i,
        root_two,
        prime,
    )
    third = _reflected_third_sectors_mod(
        -n, epsilon3, blocks, root_i, root_two, prime
    )
    auxiliary_form = (epsilon2 + epsilon3 - form_parity) % 2
    answer = 0

    for state2, coefficient2_exact in chi.ramond_fock_paths(n, epsilon2):
        aux2, ground_a2, phys2, ground_p2 = state2
        coefficient2 = quadratic_mod(
            coefficient2_exact, root_i, root_two, prime
        )
        physical_parity2 = (len(phys2) + ground_p2) % 2
        for (
            aux3,
            ground_a3,
            bosons3,
            phys3,
            ground_p3,
            coefficient3,
        ) in third:
            # The stored three-point grid fixes its auxiliary Ising form by
            # Virasoro Ward transport.  That sewing convention is not a
            # drop-in copy of the native two-spin OPE kernel on the same raw
            # labels (for example their one-leg (2,1) values have opposite
            # signs).  Use the corrected sewing convention here so this audit
            # tests only the new reflection/Heisenberg/physical-spin glue.
            auxiliary_exact = grid.fermion_value_virasoro(
                auxiliary_form,
                (),
                aux2,
                ground_a2,
                aux3,
                ground_a3,
            )
            if auxiliary_exact == 0:
                continue
            physical_exact = ordinary_eta_minus_fock_majorana_value(
                phys2,
                ground_p2,
                phys3,
                ground_p3,
            )
            if physical_exact == 0:
                continue
            auxiliary = quadratic_mod(
                auxiliary_exact, root_i, root_two, prime
            )
            physical = quadratic_mod(
                physical_exact, root_i, root_two, prime
            )
            auxiliary_parity3 = (len(aux3) + ground_a3) % 2
            koszul = -1 if physical_parity2 * auxiliary_parity3 else 1
            bosonic = heisenberg_zero_leg_factor(
                bosons3,
                q_mod,
                p2_mod,
                root_i,
                prime,
            )
            term = coefficient2 * coefficient3 % prime
            term = term * auxiliary * physical * bosonic % prime
            answer = (answer + koszul * term) % prime
    return answer


def _ward_mod(
    branch_magnitude,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    b_value,
    p2_value,
    p3_value,
    prime,
):
    n = abs(sp.Rational(branch_magnitude))
    q_value = sp.cancel(b_value + 1 / b_value)
    p1_value = sp.cancel(-q_value / 2 - p2_value - p3_value)
    exact = grid.enlarged_raw_three_point(
        0,
        n,
        -n,
        int(epsilon2),
        int(epsilon3),
        int(form_parity),
        int(eta),
        b_value,
        p1_value,
        p2_value,
        p3_value,
    )[1]
    root_i, root_two = modular.roots(prime)
    return quadratic_mod(exact, root_i, root_two, prime)


def audit(levels=(sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4))):
    """Check the native form for all copy choices through the stored grid."""

    prime = 1_000_033
    samples = (
        (sp.Rational(3, 2), sp.Rational(2, 5), -sp.Rational(3, 7)),
        (sp.Rational(5, 3), sp.Rational(3, 8), -sp.Rational(5, 9)),
    )
    checked = 0
    timings = {}
    for n in levels:
        start = time.perf_counter()
        for b_value, p2_value, p3_value in samples:
            q_value = sp.cancel(b_value + 1 / b_value)
            for epsilon2, epsilon3 in itertools.product((0, 1), repeat=2):
                form_parity, eta = 0, -1
                calculated = mixed_sheet_zero_screening_mod(
                    n,
                    epsilon2,
                    epsilon3,
                    form_parity,
                    eta,
                    q_value,
                    p2_value,
                    p3_value,
                    prime,
                )
                expected = _ward_mod(
                    n,
                    epsilon2,
                    epsilon3,
                    form_parity,
                    eta,
                    b_value,
                    p2_value,
                    p3_value,
                    prime,
                )
                if calculated != expected:
                    raise AssertionError(
                        (
                            n,
                            b_value,
                            p2_value,
                            p3_value,
                            epsilon2,
                            epsilon3,
                            form_parity,
                            eta,
                            calculated,
                            expected,
                            (calculated - expected) % prime,
                        )
                    )
                checked += 1
        timings[n] = time.perf_counter() - start
    # W_7/4 is beyond the stored Ward grid.  Exercise the complete reflected
    # endpoint (maximum physical level six) for all four copy choices; the
    # recurrence itself independently checks every intertwining equation.
    n = sp.Rational(7, 4)
    b_value, p2_value, p3_value = samples[0]
    q_value = sp.cancel(b_value + 1 / b_value)
    start = time.perf_counter()
    values = tuple(
        mixed_sheet_zero_screening_mod(
            n, epsilon2, epsilon3, 0, -1,
            q_value, p2_value, p3_value, prime,
        )
        for epsilon2, epsilon3 in itertools.product((0, 1), repeat=2)
    )
    timings[n] = time.perf_counter() - start
    print(f"mixed-sheet reflection/free-field: {checked} prime-field Ward checks")
    print(
        "elapsed by branch: "
        + ", ".join(f"W_{n}: {elapsed:.3f}s" for n, elapsed in timings.items())
    )
    print(f"W_7/4 four-copy residues mod {prime}: {values}")


if __name__ == "__main__":
    audit()
