#!/usr/bin/env python3
"""Direct NS--R--R branching three-point functions on a low-level grid.

Nothing in the Ward evaluator or in the auxiliary-fermion Pfaffian uses the
conjectured branching coefficient.  The script performs the following steps.

* Construct the raw NS chi-string branch at n=0,1/2,1 (and n=3/2 for the
  independent check) in an abstract NS PBW basis.
* Import the complete raw Ramond chi-string branches at n=1/4,3/4,5/4.
* Reduce all three physical descendant slots with the plane NS--R--R Ward
  identities in the SCblock order (NS at infinity, R at one, R at zero).
* Convert every auxiliary Fock state exactly to an Ising Virasoro descendant
  and reduce its three-point function with the c=1/2 Ward identities.
* Test the resulting raw three-point functions against the proposed
  ell-products and determine which discrete factors really remain constant.

The main output is root independent: kappa^2 and B^2.  A sign of kappa itself
depends on the arbitrary sign chosen for each normalized branch state.
"""

from __future__ import annotations

from functools import lru_cache
import argparse
import importlib.util
import itertools
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
RAMOND_DIR = ROOT / "python" / "ramond_branching_coefficient_check"
NS_CHECK = ROOT / "agent_notes" / "check_ns_branch_norms.py"
for directory in (RAMOND_DIR,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import check_ramond_branching as ramond_branch  # noqa: E402
import compute_ramond_kappa as boundary  # noqa: E402


def _load_ns_check():
    specification = importlib.util.spec_from_file_location(
        "ns_branch_norm_check", NS_CHECK
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load {NS_CHECK}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


ns_branch = _load_ns_check()

I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2
ALGEBRAIC_FIELD = sp.QQ.algebraic_field(I, SQRT2)


def quadratic_number_components(expression):
    """Write an exact sample value in the basis 1,sqrt(2),i,i*sqrt(2)."""

    real_part, imaginary_part = sp.expand_complex(
        sp.radsimp(sp.cancel(expression))
    ).as_real_imag()
    real_part = sp.expand(real_part)
    imaginary_part = sp.expand(imaginary_part)
    real_sqrt = sp.cancel(real_part.coeff(SQRT2))
    real_rational = sp.cancel(real_part - real_sqrt * SQRT2)
    imaginary_sqrt = sp.cancel(imaginary_part.coeff(SQRT2))
    imaginary_rational = sp.cancel(
        imaginary_part - imaginary_sqrt * SQRT2
    )
    components = (
        real_rational,
        real_sqrt,
        imaginary_rational,
        imaginary_sqrt,
    )
    if any(component.free_symbols or component.has(sp.sqrt) for component in components):
        raise AssertionError(
            f"Sample value left Q(i,sqrt(2)): {sp.factor(expression)}"
        )
    return components


def add_term(out, key, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    out[key] = sp.cancel(out.get(key, 0) + coefficient)
    if out[key] == 0:
        del out[key]


def state_level(word):
    return sum(-mode for _, mode in word)


def state_parity(word, ground=0):
    return (sum(kind == "G" for kind, _ in word) + ground) % 2


def physical_ground(form_parity, eta, second_ground, third_ground):
    return boundary.physical_ground(
        form_parity, eta, second_ground, third_ground
    )


class PhysicalNRREvaluator:
    """Exact plane rho_f^eta(NS descendant,R descendant,R descendant)."""

    def __init__(
        self,
        form_parity,
        eta,
        h_ns,
        h_second,
        h_third,
        beta_second,
        beta_third,
        central_charge,
    ):
        self.form_parity = int(form_parity)
        self.eta = int(eta)
        self.weights = (h_ns, h_second, h_third)
        self.betas = (None, beta_second, beta_third)
        self.central_charge = central_charge

    def bracket(self, first, second):
        first_kind, first_mode = first
        second_kind, second_mode = second
        answer = []
        if first_kind == "L" and second_kind == "L":
            answer.append(
                (first_mode - second_mode, (("L", first_mode + second_mode),))
            )
            if first_mode + second_mode == 0:
                answer.append(
                    (
                        self.central_charge
                        * (first_mode**3 - first_mode)
                        / 12,
                        (),
                    )
                )
        elif first_kind == "L" and second_kind == "G":
            answer.append(
                (
                    sp.Rational(1, 2) * first_mode - second_mode,
                    (("G", first_mode + second_mode),),
                )
            )
        elif first_kind == "G" and second_kind == "L":
            answer.append(
                (
                    first_mode - sp.Rational(1, 2) * second_mode,
                    (("G", first_mode + second_mode),),
                )
            )
        else:
            answer.append((sp.Integer(2), (("L", first_mode + second_mode),)))
            if first_mode + second_mode == 0:
                answer.append(
                    (
                        self.central_charge
                        * (first_mode**2 - sp.Rational(1, 4))
                        / 3,
                        (),
                    )
                )
        return tuple(answer)

    @staticmethod
    def canonical_key(operator):
        kind, mode = operator
        return (0 if kind == "L" else 1, mode)

    @lru_cache(None)
    def canonicalize(self, word):
        """Put a word of negative modes in L-before-G PBW order."""

        if any(mode >= 0 for _, mode in word):
            raise AssertionError(f"canonicalize received a nonnegative mode: {word}")
        for position in range(len(word) - 1):
            first, second = word[position], word[position + 1]
            if first == second and first[0] == "G":
                replacement = (("L", 2 * first[1]),)
                reduced = word[:position] + replacement + word[position + 2 :]
                return self.canonicalize(reduced)
            if self.canonical_key(first) <= self.canonical_key(second):
                continue
            exchange_sign = -1 if first[0] == second[0] == "G" else 1
            out = {}
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            for canonical, coefficient in self.canonicalize(exchanged).items():
                add_term(out, canonical, exchange_sign * coefficient)
            for bracket_coefficient, replacement in self.bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                for canonical, coefficient in self.canonicalize(reduced).items():
                    add_term(out, canonical, bracket_coefficient * coefficient)
            return out
        return {word: sp.Integer(1)}

    @lru_cache(None)
    def act(self, slot, kind, mode, word, ground):
        """Act one mode on a canonical descendant in the indicated module."""

        if mode < 0:
            out = {}
            for canonical, coefficient in self.canonicalize(((kind, mode),) + word).items():
                add_term(out, (canonical, ground), coefficient)
            return out

        if not word:
            if kind == "L":
                if mode == 0:
                    return {((), ground): self.weights[slot]}
                return {}
            if slot == 0:
                if mode == 0:
                    raise AssertionError("There is no NS G_0 mode.")
                return {}
            if mode == 0:
                coefficient, flipped = boundary.physical_g0(
                    self.betas[slot], ground
                )
                return {((), flipped): coefficient}
            return {}

        first = word[0]
        rest = word[1:]
        out = {}
        exchange_sign = -1 if kind == first[0] == "G" else 1
        for (reduced_word, reduced_ground), coefficient in self.act(
            slot, kind, mode, rest, ground
        ).items():
            for canonical, canonical_coefficient in self.canonicalize(
                (first,) + reduced_word
            ).items():
                add_term(
                    out,
                    (canonical, reduced_ground),
                    exchange_sign * coefficient * canonical_coefficient,
                )

        for bracket_coefficient, replacement in self.bracket((kind, mode), first):
            if not replacement:
                add_term(out, (rest, ground), bracket_coefficient)
                continue
            replacement_kind, replacement_mode = replacement[0]
            for key, coefficient in self.act(
                slot, replacement_kind, replacement_mode, rest, ground
            ).items():
                add_term(out, key, bracket_coefficient * coefficient)
        return out

    def state_sum(self, word1, states2, states3):
        answer = sp.Integer(0)
        for (word2, ground2), coefficient2 in states2.items():
            for (word3, ground3), coefficient3 in states3.items():
                answer += coefficient2 * coefficient3 * self.raw_value(
                    word1, word2, ground2, word3, ground3
                )
        return sp.cancel(answer)

    def first_state_sum(self, states1, word2, ground2, word3, ground3):
        answer = sp.Integer(0)
        for (word1, _), coefficient in states1.items():
            answer += coefficient * self.raw_value(
                word1, word2, ground2, word3, ground3
            )
        return sp.cancel(answer)

    @lru_cache(None)
    def raw_value(self, word1, word2, ground2, word3, ground3):
        """Recursively evaluate the fixed-parity SCblock form at z=1."""

        # Equation (ward-G-first) in the Ramond Ward-identity notes.  Its
        # parity sign is the parity of the third state below the displayed
        # outer supercurrent; it does not depend on the first descendant.
        if word1 and word1[0][0] == "G":
            _, mode = word1[0]
            rest1 = word1[1:]
            r = -mode
            answer = sp.Integer(0)
            parity_sign = (-1) ** state_parity(word3, ground3)
            maximum = int(
                max(
                    r + state_level(rest1),
                    state_level(word2),
                    state_level(word3),
                    0,
                )
            ) + 3
            for p in range(maximum + 1):
                acted2 = self.act(1, "G", p, word2, ground2)
                answer += parity_sign * sp.binomial(r, p) * self.state_sum(
                    rest1, acted2, {(word3, ground3): sp.Integer(1)}
                )
                if p:
                    acted1 = self.act(0, "G", p - r, rest1, 0)
                    answer -= (
                        sp.binomial(sp.Rational(1, 2), p)
                        * (-1) ** p
                        * self.first_state_sum(
                            acted1, word2, ground2, word3, ground3
                        )
                    )
                acted3 = self.act(
                    2, "G", r - sp.Rational(1, 2) + p, word3, ground3
                )
                answer += (
                    -I
                    * sp.binomial(sp.Rational(1, 2), p)
                    * (-1) ** p
                    * self.state_sum(
                        rest1,
                        {(word2, ground2): sp.Integer(1)},
                        acted3,
                    )
                )
            return sp.cancel(answer)

        # First remove the descendant at the middle puncture.
        if word2:
            kind, mode = word2[0]
            rest2 = word2[1:]
            if kind == "L":
                n = int(-mode)
                if n == 1:
                    exponent = (
                        self.weights[0]
                        + state_level(word1)
                        - self.weights[1]
                        - state_level(rest2)
                        - self.weights[2]
                        - state_level(word3)
                    )
                    return sp.cancel(
                        exponent
                        * self.raw_value(word1, rest2, ground2, word3, ground3)
                    )
                answer = sp.Integer(0)
                maximum = int(
                    max(state_level(word1) - n, state_level(word3) + 1, 0)
                )
                for p in range(maximum + 1):
                    ward = sp.binomial(n - 2 + p, n - 2)
                    acted1 = self.act(0, "L", n + p, word1, 0)
                    answer += ward * self.first_state_sum(
                        acted1, rest2, ground2, word3, ground3
                    )
                    acted3 = self.act(2, "L", p - 1, word3, ground3)
                    answer += ward * (-1) ** n * self.state_sum(
                        word1,
                        {(rest2, ground2): sp.Integer(1)},
                        acted3,
                    )
                return sp.cancel(answer)

            n = int(-mode)
            answer = sp.Integer(0)
            parity_sign = (-1) ** state_parity(word3, ground3)
            maximum = int(
                max(
                    n + state_level(rest2),
                    state_level(word1) - n + sp.Rational(1, 2),
                    state_level(word3),
                    0,
                )
                ) + 2
            for p in range(maximum + 1):
                acted1 = self.act(
                    0, "G", sp.Rational(2 * p + 2 * n - 1, 2), word1, 0
                )
                answer += (
                    parity_sign
                    *
                    sp.binomial(sp.Rational(1, 2) - n, p)
                    * (-1) ** p
                    * self.first_state_sum(
                        acted1, rest2, ground2, word3, ground3
                    )
                )
                acted3 = self.act(2, "G", p, word3, ground3)
                answer += (
                    I
                    * parity_sign
                    * sp.binomial(sp.Rational(1, 2) - n, p)
                    * (-1) ** (n + p)
                    * self.state_sum(
                        word1,
                        {(rest2, ground2): sp.Integer(1)},
                        acted3,
                    )
                )
                if p:
                    acted2 = self.act(1, "G", p - n, rest2, ground2)
                    answer -= (
                        sp.binomial(sp.Rational(1, 2), p)
                        * self.state_sum(
                            word1,
                            acted2,
                            {(word3, ground3): sp.Integer(1)},
                        )
                    )
            return sp.cancel(answer)

        # Next remove the BPZ descendant at infinity.
        if word1:
            kind, mode = word1[0]
            rest1 = word1[1:]
            if kind == "L":
                n = int(-mode)
                answer = self.state_sum(
                    rest1,
                    {((), ground2): sp.Integer(1)},
                    self.act(2, "L", n, word3, ground3),
                )
                for m in range(-1, n + 1):
                    ward = sp.binomial(n + 1, m + 1)
                    answer += ward * self.state_sum(
                        rest1,
                        self.act(1, "L", m, (), ground2),
                        {(word3, ground3): sp.Integer(1)},
                    )
                return sp.cancel(answer)

            r = -mode
            answer = sp.Integer(0)
            parity_sign = (-1) ** state_parity(word3, ground3)
            maximum = int(
                max(r + state_level(rest1), state_level(word3), 0)
            ) + 3
            for p in range(maximum + 1):
                acted2 = self.act(1, "G", p, (), ground2)
                answer += parity_sign * sp.binomial(r, p) * self.state_sum(
                    rest1, acted2, {(word3, ground3): sp.Integer(1)}
                )
                if p:
                    acted1 = self.act(0, "G", p - r, rest1, 0)
                    answer -= (
                        sp.binomial(sp.Rational(1, 2), p)
                        * (-1) ** p
                        * self.first_state_sum(
                            acted1, (), ground2, word3, ground3
                        )
                    )
                acted3 = self.act(
                    2, "G", r - sp.Rational(1, 2) + p, word3, ground3
                )
                answer += (
                    -I
                    * sp.binomial(sp.Rational(1, 2), p)
                    * (-1) ** p
                    * self.state_sum(
                        rest1, {((), ground2): sp.Integer(1)}, acted3
                    )
                )
            return sp.cancel(answer)

        # Finally reduce the descendant at zero.
        if word3:
            kind, mode = word3[0]
            rest3 = word3[1:]
            if kind == "L":
                n = int(-mode)
                coefficient = (
                    self.weights[2]
                    + state_level(rest3)
                    + n * self.weights[1]
                    - self.weights[0]
                )
                return sp.cancel(
                    coefficient * self.raw_value((), (), ground2, rest3, ground3)
                )

            m = int(-mode)
            parity_sign = (-1) ** state_parity(rest3, ground3)
            answer = sp.Integer(0)
            maximum = int(m + state_level(rest3)) + 2
            for p in range(maximum + 1):
                acted2 = self.act(1, "G", p, (), ground2)
                answer += (
                    -I
                    * parity_sign
                    * sp.binomial(sp.Rational(1, 2) - m, p)
                    * self.state_sum((), acted2, {(rest3, ground3): 1})
                )
                acted1 = self.act(
                    0, "G", sp.Rational(2 * m - 1, 2) + p, (), 0
                )
                answer += (
                    I
                    * (-1) ** p
                    * sp.binomial(sp.Rational(1, 2), p)
                    * self.first_state_sum(
                        acted1, (), ground2, rest3, ground3
                    )
                )
                if not p:
                    continue
                acted3 = self.act(2, "G", -m + p, rest3, ground3)
                ward = sp.binomial(sp.Rational(1, 2), p) * (-1) ** p
                answer -= ward * self.state_sum(
                    (), {((), ground2): sp.Integer(1)}, acted3
                )
            return sp.cancel(answer)

        return physical_ground(
            self.form_parity, self.eta, ground2, ground3
        )

    def value(self, word1, word2, ground2, word3, ground3):
        """The fixed-parity SCblock form in the displayed NS--R--R order."""

        return self.raw_value(word1, word2, ground2, word3, ground3)

    def check_anchors(self):
        """Compare the generalized recursion with independent low-level data."""

        old = boundary.PhysicalNRREvaluator(
            self.form_parity,
            self.eta,
            self.weights[0],
            self.weights[1],
            self.weights[2],
            self.betas[1],
            self.betas[2],
            self.central_charge,
        )
        for second_ground in (0, 1):
            for third_ground in (0, 1):
                for word3 in (
                    (),
                    (("G", -1),),
                    (("L", -1),),
                    (("G", -2), ("G", -1)),
                ):
                    difference = sp.factor(
                        sp.cancel(
                            self.value(
                                (), (), second_ground, word3, third_ground
                            )
                            - old.value(second_ground, word3, third_ground)
                        )
                    )
                    if difference != 0:
                        raise AssertionError(
                            "General Ward recursion disagrees with the boundary "
                            f"evaluator: {difference}"
                        )


def coefficient_in_series(expression, variable, power, order=None):
    if order is None:
        order = max(8, int(power) + 6)
    expanded = sp.series(expression, variable, 0, order).removeO().expand()
    return sp.expand(expanded).coeff(variable, int(power))


def local_fermion_data(leg, variable):
    """Coordinate, square-root ratio, and odd kernel in one local frame."""

    if leg == 3:
        return (
            variable**2,
            variable / sp.sqrt(1 - variable**2),
            1 / (SQRT2 * variable * sp.sqrt(1 - variable**2)),
        )
    if leg == 2:
        return (
            1 + variable**2,
            -I * sp.sqrt(1 + variable**2) / variable,
            -I / (SQRT2 * variable * sp.sqrt(1 + variable**2)),
        )
    if leg == 1:
        return (
            1 / variable**2,
            -I / sp.sqrt(1 - variable**2),
            -I * variable**2 / (SQRT2 * sp.sqrt(1 - variable**2)),
        )
    raise ValueError(f"Unknown leg {leg}")


def local_power(leg, mode):
    return int(2 * sp.Rational(mode) + (1 if leg == 1 else -1))


def fermion_spin_frame(leg):
    """Lift from the global square-root branch to the three plumbing frames."""

    return {1: -sp.Integer(1), 2: I, 3: sp.Integer(1)}[int(leg)]


def fermion_pair_orientation(leg_a, leg_b):
    """Cocycle for pairs crossing the ordered middle Ramond puncture."""

    return (
        -1
        if (int(leg_a), int(leg_b)) in ((1, 2), (2, 3))
        else 1
    )


def auxiliary_virasoro_transport_phase(first_modes, second_modes):
    """Transport the ket Fock--Virasoro map to infinity and one.

    ``auxiliary_to_virasoro`` is constructed in the canonical ket frame at
    zero.  The NS state at infinity is BPZ ordered, while the Ramond state at
    one uses the other square-root chart.  Moving the two Fock strings into
    the displayed ``(infinity, one, zero)`` tensor order contributes

        (-1)^(N_1 N_2 + ell_2 (1+N_2)),

    where ``N_j`` is the number of nonzero fermion modes and ``ell_2`` is the
    integer Ramond level at one.  Omitting this transport was invisible for
    single Virasoro descendants but failed for mixed descendant insertions.
    """

    first_parity = len(tuple(first_modes)) % 2
    second_parity = len(tuple(second_modes)) % 2
    second_level_parity = int(sum(second_modes)) % 2
    exponent = (
        first_parity * second_parity
        + second_level_parity * (1 + second_parity)
    ) % 2
    return -1 if exponent else 1


@lru_cache(None)
def fermion_one_coefficient(leg, mode):
    y = sp.symbols("y")
    _, _, odd_kernel = local_fermion_data(leg, y)
    return sp.simplify(
        fermion_spin_frame(leg)
        * coefficient_in_series(odd_kernel, y, local_power(leg, mode), 20)
    )


@lru_cache(None)
def fermion_pair_coefficient(leg_a, mode_a, leg_b, mode_b):
    """Ordered coefficient of the two-spin-field fermion kernel."""

    y, x, ratio = sp.symbols("y x ratio")
    z_a, square_a, _ = local_fermion_data(leg_a, y)
    z_b, square_b, _ = local_fermion_data(leg_b, x)
    kernel = (square_a / square_b + square_b / square_a) / (
        2 * (z_a - z_b)
    )
    power_a = local_power(leg_a, mode_a)
    power_b = local_power(leg_b, mode_b)
    if leg_a == leg_b == 1:
        # At infinity the leftmost field has the larger global radius and
        # therefore the smaller inverse-radius coordinate.
        nested = sp.simplify(kernel.subs(y, ratio * x))
        nested = coefficient_in_series(nested, ratio, power_a, 24)
        return sp.simplify(
            fermion_spin_frame(leg_a)
            * fermion_spin_frame(leg_b)
            * fermion_pair_orientation(leg_a, leg_b)
            *
            coefficient_in_series(nested, x, power_a + power_b, 28)
        )
    if leg_a == leg_b:
        # At zero or one the right field is on the inner local contour.
        nested = sp.simplify(kernel.subs(x, ratio * y))
        nested = coefficient_in_series(nested, ratio, power_b, 24)
        return sp.simplify(
            fermion_spin_frame(leg_a)
            * fermion_spin_frame(leg_b)
            * fermion_pair_orientation(leg_a, leg_b)
            *
            coefficient_in_series(nested, y, power_a + power_b, 28)
        )
    expanded_a = coefficient_in_series(kernel, y, power_a, 24)
    return sp.simplify(
        fermion_spin_frame(leg_a)
        * fermion_spin_frame(leg_b)
        * fermion_pair_orientation(leg_a, leg_b)
        *
        coefficient_in_series(expanded_a, x, power_b, 24)
    )


@lru_cache(None)
def fermion_even_wick(fields):
    """Pfaffian of the ordered two-spin-field kernel."""

    if not fields:
        return sp.Integer(1)
    if len(fields) % 2:
        raise ValueError("fermion_even_wick requires an even number of fields")
    first = fields[0]
    answer = sp.Integer(0)
    for position in range(1, len(fields)):
        second = fields[position]
        remaining = fields[1:position] + fields[position + 1 :]
        answer += (
            (-1) ** (position + 1)
            * fermion_pair_coefficient(*first, *second)
            * fermion_even_wick(remaining)
        )
    return sp.simplify(answer)


@lru_cache(None)
def fermion_value(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Auxiliary NS--R--R descendant form in the canonical spin frames."""

    # BPZ reverses the first string and sends every creator to minus its
    # positive mode because psi_r^dagger=-psi_-r.
    fields = (
        tuple((1, mode) for mode in reversed(first_modes))
        + tuple((2, mode) for mode in second_modes)
        + tuple((3, mode) for mode in third_modes)
    )
    bpz_sign = (-1) ** len(first_modes)
    if not len(fields) % 2:
        ground = boundary.fermion_ground(
            form_parity, second_ground, third_ground
        )
        return sp.simplify(bpz_sign * fermion_even_wick(fields) * ground)

    # An unpaired fermion changes the local ground datum at the puncture
    # from which it came.  Keeping this information is essential for
    # cross-leg terms: a middle-leg fermion flips the second Ramond ground,
    # a zero-leg fermion flips the third, and an infinity fermion changes
    # the parity of the NS form.
    answer = sp.Integer(0)
    for position, (leg, mode) in enumerate(fields):
        remaining = fields[:position] + fields[position + 1 :]
        if leg == 1:
            ground = boundary.fermion_ground(
                1 - form_parity, second_ground, third_ground
            )
        elif leg == 2:
            ground = boundary.fermion_ground(
                form_parity, 1 - second_ground, third_ground
            )
        else:
            ground = boundary.fermion_ground(
                form_parity, second_ground, 1 - third_ground
            )
        answer += (
            (-1) ** position
            * fermion_one_coefficient(leg, mode)
            * fermion_even_wick(remaining)
            * ground
        )
    return sp.simplify(bpz_sign * answer)


@lru_cache(None)
def integer_partitions(total, largest=None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in integer_partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


def auxiliary_apply_mode(sector, mode, state):
    """Apply one free-fermion mode to an NS or Ramond Fock state."""

    modes, ground = state
    mode = sp.Rational(mode)
    if mode < 0:
        created = -mode
        if created in modes:
            return None, 0
        crossings = sum(bool(existing > created) for existing in modes)
        return (
            tuple(sorted(modes + (created,), reverse=True)),
            ground,
        ), (-1) ** crossings
    if mode > 0:
        if mode not in modes:
            return None, 0
        position = modes.index(mode)
        return (modes[:position] + modes[position + 1 :], ground), (-1) ** position
    if sector != "R":
        raise AssertionError("The NS fermion has no zero mode.")
    return (modes, 1 - ground), (-1) ** len(modes) / SQRT2


def auxiliary_apply_L(mode, state, sector):
    """Apply L_mode^F for a negative integer mode."""

    if mode >= 0:
        raise ValueError("Only negative Virasoro creators are needed here.")
    modes, _ = state
    if sector == "R":
        candidates = set(range(mode, 1))
        candidates.update(int(value) for value in modes)
        candidates.update(mode - int(value) for value in modes)
    else:
        candidates2 = set(range(2 * mode + 1, 1, 2))
        candidates2.update(int(2 * value) for value in modes)
        candidates2.update(2 * mode - int(2 * value) for value in modes)
        candidates = {sp.Rational(value, 2) for value in candidates2}
    out = {}
    for summation_mode in candidates:
        middle, second_coefficient = auxiliary_apply_mode(
            sector, summation_mode, state
        )
        if second_coefficient == 0:
            continue
        final, first_coefficient = auxiliary_apply_mode(
            sector, mode - summation_mode, middle
        )
        if first_coefficient == 0:
            continue
        add_term(
            out,
            final,
            sp.Rational(1, 2)
            * summation_mode
            * second_coefficient
            * first_coefficient,
        )
    return out


def auxiliary_apply_expression(action, expression):
    out = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            add_term(out, final, outer * inner)
    return out


def auxiliary_vir_descendant(partition, sector, primary_ground):
    if sector == "NS" and primary_ground == 1:
        expression = {((sp.Rational(1, 2),), 0): sp.Integer(1)}
    else:
        expression = {((), primary_ground): sp.Integer(1)}
    for mode in reversed(partition):
        expression = auxiliary_apply_expression(
            lambda state, n=-mode: auxiliary_apply_L(n, state, sector),
            expression,
        )
    return expression


@lru_cache(None)
def auxiliary_fock_basis(sector, primary_ground, level):
    """Fock basis at relative Virasoro level in one Ising module."""

    level = int(level)
    target_parity = int(primary_ground)
    target_sum2 = 2 * level + (1 if sector == "NS" and primary_ground else 0)
    allowed2 = (
        tuple(range(1, target_sum2 + 1, 2))
        if sector == "NS"
        else tuple(range(2, target_sum2 + 1, 2))
    )
    answer = []
    from itertools import combinations

    for count in range(len(allowed2) + 1):
        for subset in combinations(allowed2, count):
            if sum(subset) != target_sum2:
                continue
            modes = tuple(sorted((sp.Rational(value, 2) for value in subset), reverse=True))
            if sector == "NS":
                if count % 2 != target_parity:
                    continue
                ground = 0
            else:
                ground = (target_parity - count) % 2
            answer.append((modes, ground))
    return tuple(answer)


@lru_cache(None)
def auxiliary_to_virasoro(sector, modes, ground):
    """Expand one auxiliary Fock state in an independent Ising Vir basis."""

    modes = tuple(sp.Rational(mode) for mode in modes)
    total_parity = (len(modes) + int(ground)) % 2
    if sector == "NS":
        primary_ground = total_parity
        level = sum(modes) - sp.Rational(primary_ground, 2)
    else:
        primary_ground = total_parity
        level = sum(modes)
    if not level.is_integer or level < 0:
        raise AssertionError("Invalid auxiliary Virasoro relative level.")
    level = int(level)
    rows = auxiliary_fock_basis(sector, primary_ground, level)
    row_index = {state: index for index, state in enumerate(rows)}
    partitions = integer_partitions(level)
    matrix = sp.zeros(len(rows), len(partitions))
    for column, partition in enumerate(partitions):
        expression = auxiliary_vir_descendant(
            partition, sector, primary_ground
        )
        for state, coefficient in expression.items():
            matrix[row_index[state], column] = coefficient
    pivots = matrix.rref()[1]
    independent = tuple(partitions[index] for index in pivots)
    square = matrix[:, pivots]
    if square.rows != square.cols:
        raise AssertionError(
            f"Incomplete Ising Vir basis in {sector} at level {level}: "
            f"shape={square.shape}"
        )
    target = sp.zeros(len(rows), 1)
    target[row_index[(modes, int(ground))]] = 1
    coefficients = square.inv() * target
    return primary_ground, independent, tuple(coefficients)


@lru_cache(None)
def canonicalize_auxiliary_virasoro_word(word):
    """Put ``L_-word`` in partition order without dropping commutators.

    For an adjacent inversion ``a<b``, the Virasoro algebra gives

    ``L_-a L_-b = L_-b L_-a + (b-a) L_-(a+b)``.

    Sorting the tuple as though its entries commuted loses the second term.
    """

    word = tuple(int(value) for value in word)
    for position in range(len(word) - 1):
        first, second = word[position : position + 2]
        if first >= second:
            continue
        out = {}
        exchanged = word[:position] + (second, first) + word[position + 2 :]
        for canonical, coefficient in (
            canonicalize_auxiliary_virasoro_word(exchanged).items()
        ):
            add_term(out, canonical, coefficient)
        bracket = word[:position] + (first + second,) + word[position + 2 :]
        for canonical, coefficient in (
            canonicalize_auxiliary_virasoro_word(bracket).items()
        ):
            add_term(out, canonical, (second - first) * coefficient)
        return out
    return {word: sp.Integer(1)}


class AuxiliaryVirasoroEvaluator:
    """Ordinary Virasoro three-point form for the Ising modules."""

    def __init__(self, first_primary, form_parity, second_ground, third_ground):
        self.first_primary = int(first_primary)
        self.form_parity = int(form_parity)
        self.second_ground = int(second_ground)
        self.third_ground = int(third_ground)
        self.weights = (
            sp.Rational(self.first_primary, 2),
            sp.Rational(1, 16),
            sp.Rational(1, 16),
        )
        self.central_charge = sp.Rational(1, 2)

    def base(self):
        if self.first_primary == 0:
            return boundary.fermion_ground(
                self.form_parity, self.second_ground, self.third_ground
            )
        return (
            -I
            / SQRT2
            * boundary.fermion_ground(
                1 - self.form_parity,
                self.second_ground,
                self.third_ground,
            )
        )

    @lru_cache(None)
    def act(self, slot, mode, word):
        mode = int(mode)
        word = tuple(int(value) for value in word)
        if mode < 0:
            return canonicalize_auxiliary_virasoro_word((-mode,) + word)
        if not word:
            if mode == 0:
                return {(): self.weights[slot]}
            return {}
        first = word[0]
        rest = word[1:]
        out = {}
        for reduced, coefficient in self.act(slot, mode, rest).items():
            for canonical, ordering in (
                canonicalize_auxiliary_virasoro_word((first,) + reduced).items()
            ):
                add_term(out, canonical, coefficient * ordering)
        bracket_coefficient = mode + first
        replacement = mode - first
        if replacement < 0:
            for reduced, coefficient in (
                canonicalize_auxiliary_virasoro_word(
                    (-replacement,) + rest
                ).items()
            ):
                add_term(out, reduced, bracket_coefficient * coefficient)
        elif replacement == 0:
            # L_0 acts on the descendant ``rest`` as well as its primary.
            add_term(
                out,
                rest,
                bracket_coefficient * (self.weights[slot] + sum(rest)),
            )
        else:
            for reduced, coefficient in self.act(slot, replacement, rest).items():
                add_term(out, reduced, bracket_coefficient * coefficient)
        if mode == first:
            add_term(
                out,
                rest,
                self.central_charge * (mode**3 - mode) / 12,
            )
        return out

    def sum_first(self, states, word2, word3):
        return sp.cancel(
            sum(
                coefficient * self.value(word1, word2, word3)
                for word1, coefficient in states.items()
            )
        )

    def sum_third(self, word1, word2, states):
        return sp.cancel(
            sum(
                coefficient * self.value(word1, word2, word3)
                for word3, coefficient in states.items()
            )
        )

    @lru_cache(None)
    def value(self, word1, word2, word3):
        if word2:
            n = int(word2[0])
            rest2 = word2[1:]
            if n == 1:
                exponent = (
                    self.weights[0]
                    + sum(word1)
                    - self.weights[1]
                    - sum(rest2)
                    - self.weights[2]
                    - sum(word3)
                )
                return sp.cancel(exponent * self.value(word1, rest2, word3))
            answer = sp.Integer(0)
            maximum = max(sum(word1) - n, sum(word3) + 1, 0)
            for p in range(int(maximum) + 1):
                ward = sp.binomial(n - 2 + p, n - 2)
                answer += ward * self.sum_first(
                    self.act(0, n + p, word1), rest2, word3
                )
                answer += ward * (-1) ** n * self.sum_third(
                    word1, rest2, self.act(2, p - 1, word3)
                )
            return sp.cancel(answer)
        if word1:
            n = int(word1[0])
            rest1 = word1[1:]
            answer = self.sum_third(
                rest1, (), self.act(2, n, word3)
            )
            for m in range(-1, n + 1):
                ward = sp.binomial(n + 1, m + 1)
                acted2 = self.act(1, m, ())
                answer += ward * sum(
                    coefficient * self.value(rest1, reduced2, word3)
                    for reduced2, coefficient in acted2.items()
                )
            return sp.cancel(answer)
        if word3:
            n = int(word3[0])
            rest3 = word3[1:]
            coefficient = (
                self.weights[2]
                + sum(rest3)
                + n * self.weights[1]
                - self.weights[0]
            )
            return sp.cancel(coefficient * self.value((), (), rest3))
        return self.base()


@lru_cache(None)
def fermion_value_virasoro(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    first_primary, basis1, coefficients1 = auxiliary_to_virasoro(
        "NS", first_modes, 0
    )
    second_primary, basis2, coefficients2 = auxiliary_to_virasoro(
        "R", second_modes, second_ground
    )
    third_primary, basis3, coefficients3 = auxiliary_to_virasoro(
        "R", third_modes, third_ground
    )
    evaluator = AuxiliaryVirasoroEvaluator(
        first_primary, form_parity, second_primary, third_primary
    )
    answer = sp.Integer(0)
    for word1, coefficient1 in zip(basis1, coefficients1):
        for word2, coefficient2 in zip(basis2, coefficients2):
            for word3, coefficient3 in zip(basis3, coefficients3):
                answer += (
                    coefficient1
                    * coefficient2
                    * coefficient3
                    * evaluator.value(word1, word2, word3)
                )
    return sp.factor(
        auxiliary_virasoro_transport_phase(first_modes, second_modes)
        * sp.cancel(answer)
    )


@lru_cache(None)
def ns_components(branch_label, q_value, momentum):
    """Raw NS chi-string branch in auxiliary x abstract-SCA components."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return (((), (), sp.Integer(1)),)
    count = int(2 * branch_label)
    all_modes2 = tuple(range(4 * branch_label - 1, 0, -2))
    if len(all_modes2) != count:
        raise AssertionError("Unexpected NS chi-string length.")
    answer = []
    from itertools import combinations

    for physical_count in range(count + 1):
        for physical_modes2 in combinations(all_modes2, physical_count):
            physical_modes2 = tuple(sorted(physical_modes2, reverse=True))
            auxiliary_modes2 = tuple(
                mode for mode in all_modes2 if mode not in physical_modes2
            )
            chi_coefficient = ns_branch.coefficient_in_chi_product(
                all_modes2, physical_modes2
            )
            ordered_basis, coefficients = ns_branch.abstract_eta_coefficients(
                physical_modes2
            )
            coefficients = coefficients.subs(
                {ns_branch.Q: q_value, ns_branch.P: momentum}, simultaneous=True
            )
            for index, (virasoro_modes, supercurrent_modes2) in enumerate(
                ordered_basis
            ):
                coefficient = sp.cancel(chi_coefficient * coefficients[index])
                if coefficient == 0:
                    continue
                word = tuple(("L", -sp.Integer(mode)) for mode in virasoro_modes)
                word += tuple(
                    ("G", -sp.Rational(mode2, 2))
                    for mode2 in supercurrent_modes2
                )
                answer.append(
                    (
                        tuple(sp.Rational(mode2, 2) for mode2 in auxiliary_modes2),
                        word,
                        coefficient,
                    )
                )
    return tuple(answer)


@lru_cache(None)
def ramond_components(branch_label, parity, q_value, momentum):
    return boundary.branch_components(
        branch_label,
        parity,
        {ramond_branch.Q: q_value, ramond_branch.P: momentum},
    )


def enlarged_raw_three_point(
    n1,
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    b_value,
    p1,
    p2,
    p3,
    check_ward=False,
    orientation="left",
):
    q_value = sp.cancel(b_value + 1 / b_value)
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h1 = (q_value**2 / 4 - p1**2) / 2
    h2 = sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2
    h3 = sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2
    evaluator = PhysicalNRREvaluator(
        form_parity,
        eta,
        h1,
        h2,
        h3,
        p2 / SQRT2,
        p3 / SQRT2,
        central_charge,
    )
    if check_ward:
        evaluator.check_anchors()

    first = ns_components(n1, q_value, p1)
    second = ramond_components(n2, epsilon2, q_value, p2)
    third = ramond_components(n3, epsilon3, q_value, p3)
    ns_parity = int(2 * sp.Rational(n1)) % 2
    fermion_form_parity = (
        ns_parity + int(epsilon2) + int(epsilon3) - int(form_parity)
    ) % 2

    exact_number_field = not any(
        sp.sympify(value).free_symbols
        for value in (b_value, p1, p2, p3)
    )
    answer = sp.Integer(0)
    algebraic_answer = [sp.Integer(0)] * 4 if exact_number_field else None
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = state_parity(word1)
        auxiliary_parity1 = len(auxiliary1) % 2
        for (
            auxiliary2,
            auxiliary_ground2,
            word2,
            physical_ground2,
            coefficient2,
        ) in second:
            physical_parity2 = state_parity(word2, physical_ground2)
            auxiliary_parity2 = (len(auxiliary2) + auxiliary_ground2) % 2
            for (
                auxiliary3,
                auxiliary_ground3,
                word3,
                physical_ground3,
                coefficient3,
            ) in third:
                physical_parity3 = state_parity(word3, physical_ground3)
                auxiliary_parity3 = (len(auxiliary3) + auxiliary_ground3) % 2
                auxiliary = fermion_value_virasoro(
                    fermion_form_parity,
                    auxiliary1,
                    auxiliary2,
                    auxiliary_ground2,
                    auxiliary3,
                    auxiliary_ground3,
                )
                if auxiliary == 0:
                    continue
                physical = evaluator.value(
                    word1,
                    word2,
                    physical_ground2,
                    word3,
                    physical_ground3,
                )
                if physical == 0:
                    continue
                if orientation == "left":
                    tensor_exponent = (
                        physical_parity1
                        * (auxiliary_parity2 + auxiliary_parity3)
                        + physical_parity2 * auxiliary_parity3
                    )
                elif orientation == "right":
                    tensor_exponent = (
                        auxiliary_parity1
                        * (physical_parity2 + physical_parity3)
                        + auxiliary_parity2 * physical_parity3
                    )
                else:
                    raise ValueError("orientation must be 'left' or 'right'")
                tensor_sign = (-1) ** tensor_exponent
                term = (
                    tensor_sign
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary
                    * physical
                )
                if exact_number_field:
                    for index, component in enumerate(
                        quadratic_number_components(term)
                    ):
                        algebraic_answer[index] += component
                else:
                    answer += term
    if exact_number_field:
        answer = (
            algebraic_answer[0]
            + algebraic_answer[1] * SQRT2
            + I * algebraic_answer[2]
            + I * SQRT2 * algebraic_answer[3]
        )
    return fermion_form_parity, sp.factor(sp.cancel(answer))


@lru_cache(None)
def ns_raw_norm(branch_label, b_value, momentum):
    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return sp.Integer(1)
    q_value = sp.cancel(b_value + 1 / b_value)
    index = int(4 * branch_label)
    return sp.factor(
        (-1) ** int(2 * branch_label)
        * 2 ** int(2 * branch_label)
        * boundary.ell(2 * momentum, index, b_value)
        / boundary.ell(q_value + 2 * momentum, index, b_value)
    )


def ramond_raw_norm_product(branch_label, parity, b_value, momentum):
    """Closed direct norm for the three positive Ramond labels used here.

    The formula is not obtained from a one-leg Whittaker coefficient.  It is
    the factorization of the Gram contraction of the explicit chi-string
    state.  The restriction to M=0,1,2 prevents the low-level computation
    from being advertised as an unchecked all-level statement.
    """

    branch_label = sp.Rational(branch_label)
    mode_count = 2 * branch_label - sp.Rational(1, 2)
    if not mode_count.is_integer or int(mode_count) not in (0, 1, 2):
        raise ValueError(
            "The direct Ramond norm product is checked here only at "
            "n=1/4,3/4,5/4."
        )
    mode_count = int(mode_count)
    parity = int(parity)
    if parity == 0:
        coefficient = 2 ** (2 * (mode_count // 2) + 1)
    elif parity == 1:
        coefficient = -2 ** (2 * ((mode_count + 1) // 2))
    else:
        raise ValueError("The Ramond parity must be 0 or 1.")
    q_value = sp.cancel(b_value + 1 / b_value)
    index = int(4 * branch_label)
    return sp.factor(
        coefficient
        * boundary.ell(2 * momentum, index, b_value)
        / boundary.ell(q_value + 2 * momentum, index, b_value)
    )


@lru_cache(None)
def raw_norms(n1, n2, n3, epsilon2, epsilon3, b_value, p1, p2, p3):
    return (
        ns_raw_norm(n1, b_value, p1),
        boundary.branch_norm_at(n2, epsilon2, b_value, p2),
        boundary.branch_norm_at(n3, epsilon3, b_value, p3),
    )


def kappa_squared_candidate(
    raw,
    norms,
    labels,
    momenta,
    b_value,
    second_sheet,
    third_sheet,
):
    n1, n2, n3 = labels
    p1, p2, p3 = momenta
    denominator_product = sp.prod(
        boundary.leg_product(momentum, label, b_value)
        for momentum, label in zip(momenta, labels)
    )
    numerator = boundary.numerator_product(
        n1,
        n2,
        n3,
        p1,
        second_sheet * p2,
        third_sheet * p3,
        b_value,
    )
    return sp.factor(
        sp.cancel(
            raw**2
            * denominator_product
            / (sp.prod(norms) * numerator**2)
        )
    )


def direct_certificate(
    n1,
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    sample,
    check_ward=False,
):
    b_value, p1, p2, p3 = sample
    fermion_form_parity, raw = enlarged_raw_three_point(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        b_value,
        p1,
        p2,
        p3,
        check_ward=check_ward,
    )
    norms = raw_norms(
        n1, n2, n3, epsilon2, epsilon3, b_value, p1, p2, p3
    )
    candidates = {}
    for second_sheet in (1, -1):
        for third_sheet in (1, -1):
            candidates[(second_sheet, third_sheet)] = kappa_squared_candidate(
                raw,
                norms,
                (n1, n2, n3),
                (p1, p2, p3),
                b_value,
                second_sheet,
                third_sheet,
            )
    return {
        "g": fermion_form_parity,
        "raw": raw,
        "norms": norms,
        "candidates": candidates,
    }


SAMPLES = (
    (
        sp.Rational(3, 2),
        sp.Rational(1, 3),
        sp.Rational(2, 5),
        sp.Rational(3, 7),
    ),
    (
        sp.Rational(5, 3),
        sp.Rational(1, 4),
        sp.Rational(3, 8),
        sp.Rational(5, 9),
    ),
)


def kernel_self_checks():
    ordering = canonicalize_auxiliary_virasoro_word((1, 2))
    if ordering != {(2, 1): sp.Integer(1), (3,): sp.Integer(1)}:
        raise AssertionError(
            "The [L_-1,L_-2]=L_-3 ordering term was lost."
        )
    level_check = AuxiliaryVirasoroEvaluator(0, 0, 0, 0).act(
        1, 2, (2, 1)
    )
    if level_check != {(1,): sp.Rational(9, 2)}:
        raise AssertionError(
            "L_0 failed to include the level of its remaining descendant."
        )
    if fermion_one_coefficient(3, 1) != 1 / (2 * SQRT2):
        raise AssertionError("The third-leg one-fermion coefficient is wrong.")
    if fermion_pair_coefficient(3, 2, 3, 1) != sp.Rational(1, 32):
        raise AssertionError("The ordered third-leg two-fermion coefficient is wrong.")
    # The first-slot sign is the NS--R--R plumbing-frame convention selected
    # by the (1/2,1/4,3/4) factorization.  The third-slot value is the
    # independent two-spin-kernel anchor used in the boundary calculation.
    if fermion_value_virasoro(0, (sp.Rational(1, 2),), (), 0, (), 1) != I / SQRT2:
        raise AssertionError("The first-slot Ising-primary phase is wrong.")
    if fermion_value_virasoro(0, (), (), 0, (2, 1), 0) != sp.Rational(1, 32):
        raise AssertionError("The third-slot Ising descendant is wrong.")
    # The old commuting-sort implementation gave -361/128 here.  Exact
    # negative-mode ordering and the descendant contribution to L_0 give
    # -425/128.
    if fermion_value_virasoro(0, (), (2,), 0, (2,), 0) != -sp.Rational(
        425, 128
    ):
        raise AssertionError("The mixed mode-two Ising regression failed.")


GRID_NS_LEVELS = (sp.Integer(0), sp.Rational(1, 2), sp.Integer(1))
GRID_R_LEVELS = (
    sp.Rational(1, 4),
    sp.Rational(3, 4),
    sp.Rational(5, 4),
)
DISCRETE_CHOICES = tuple(
    itertools.product((0, 1), (0, 1), (0, 1), (1, -1))
)


def direct_norm_report():
    """Check ||W_n^epsilon||^2 directly at n=1/4,3/4,5/4."""

    print("DIRECT RAMOND RAW-NORM CHECK")
    symbolic_expected = {
        (sp.Rational(1, 4), 0): sp.Integer(2),
        (sp.Rational(1, 4), 1): -sp.Integer(1),
        (
            sp.Rational(3, 4),
            0,
        ): 2
        * (
            4 * ramond_branch.P**2
            + 2 * ramond_branch.P * ramond_branch.Q
            + 1
        )
        / (
            4 * ramond_branch.P**2
            + 6 * ramond_branch.P * ramond_branch.Q
            + 2 * ramond_branch.Q**2
            + 1
        ),
        (
            sp.Rational(3, 4),
            1,
        ): -4
        * (
            4 * ramond_branch.P**2
            + 2 * ramond_branch.P * ramond_branch.Q
            + 1
        )
        / (
            4 * ramond_branch.P**2
            + 6 * ramond_branch.P * ramond_branch.Q
            + 2 * ramond_branch.Q**2
            + 1
        ),
    }
    for (branch_label, parity), expected in symbolic_expected.items():
        component_count, calculated = ramond_branch.branch_norm(
            branch_label, parity
        )[2:4]
        residual = sp.factor(sp.cancel(calculated - expected))
        if residual != 0:
            raise AssertionError(
                f"Symbolic norm mismatch at n={branch_label}, "
                f"epsilon={parity}: {residual}"
            )
        print(
            f"n={branch_label} epsilon={parity} components={component_count} "
            "symbolic residual=0"
        )

    norm_samples = (
        (sp.Rational(3, 2), sp.Rational(2, 5)),
        (sp.Rational(5, 3), sp.Rational(7, 10)),
    )
    for branch_label in GRID_R_LEVELS:
        for b_value, momentum in norm_samples:
            for parity in (0, 1):
                component_count, calculated = ramond_branch.branch_norm(
                    branch_label,
                    parity,
                    substitutions={
                        ramond_branch.Q: b_value + 1 / b_value,
                        ramond_branch.P: momentum,
                    },
                )[2:4]
                expected = ramond_raw_norm_product(
                    branch_label, parity, b_value, momentum
                )
                residual = sp.factor(sp.cancel(calculated - expected))
                if residual != 0:
                    raise AssertionError(
                        f"Exact norm mismatch at n={branch_label}, "
                        f"epsilon={parity}, b={b_value}, P={momentum}: "
                        f"{residual}"
                    )
                print(
                    f"n={branch_label} epsilon={parity} "
                    f"components={component_count} b={b_value} "
                    f"P={momentum} norm={calculated} residual=0"
                )


def normalized_branching_square(labels, discrete, sample):
    """Return g and B^2 before imposing a finite-product conjecture."""

    epsilon2, epsilon3, form_parity, eta = discrete
    result = direct_certificate(
        *labels,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        sample,
    )
    return result["g"], sp.factor(
        sp.cancel(result["raw"] ** 2 / sp.prod(result["norms"]))
    ), result


def check_discrete_reduction(labels, sample):
    """Check all sixteen restrictions against four master amplitudes.

    At fixed levels the only independent continuous functions are the four
    choices (epsilon_2,eta).  Squaring removes branch-root signs, and the
    remaining epsilon_3 and f dependence is completely universal.
    """

    values = {}
    certificates = {}
    for discrete in DISCRETE_CHOICES:
        fermion_parity, value, certificate = normalized_branching_square(
            labels, discrete, sample
        )
        expected_g = (
            int(2 * sp.Rational(labels[0]))
            + discrete[0]
            + discrete[1]
            - discrete[2]
        ) % 2
        if fermion_parity != expected_g:
            raise AssertionError(
                f"Parity mismatch at {labels}, {discrete}: "
                f"g={fermion_parity}, expected {expected_g}"
            )
        values[discrete] = value
        certificates[discrete] = certificate

    for epsilon2, epsilon3, form_parity, eta in DISCRETE_CHOICES:
        base = values[(epsilon2, 0, 0, eta)]
        expected = (-1) ** epsilon3 * (-I) ** form_parity * base
        difference = sp.factor(
            sp.cancel(
                values[(epsilon2, epsilon3, form_parity, eta)] - expected
            )
        )
        if difference != 0:
            raise AssertionError(
                "The discrete reduction failed at "
                f"labels={labels}, choice={(epsilon2, epsilon3, form_parity, eta)}: "
                f"{difference}"
            )

    # The raw states have fixed phases, so before taking norm roots there is
    # a stronger unsquared statement.  This is the convenient form in which
    # to record every one of the sixteen three-point functions.
    _, n2, n3 = map(sp.Rational, labels)
    second_mode_count = int(2 * n2 - sp.Rational(1, 2))
    third_mode_count = int(2 * n3 - sp.Rational(1, 2))
    third_parity_scale = sp.Pow(
        2, sp.Rational((-1) ** (third_mode_count + 1), 2)
    )
    raw_values = {
        discrete: certificates[discrete]["raw"]
        for discrete in DISCRETE_CHOICES
    }
    for epsilon2, epsilon3, form_parity, eta in DISCRETE_CHOICES:
        base = raw_values[(epsilon2, 0, 0, eta)]
        expected = (
            base
            * third_parity_scale**epsilon3
            * (
                eta
                * (-1) ** (second_mode_count + 1 + epsilon2)
                * EIGHTH_MINUS
            )
            ** form_parity
            * (-1) ** (epsilon3 * form_parity)
        )
        difference = sp.factor(
            sp.cancel(
                raw_values[(epsilon2, epsilon3, form_parity, eta)]
                - expected
            )
        )
        if difference != 0:
            raise AssertionError(
                "The raw discrete reduction failed at "
                f"labels={labels}, choice={(epsilon2, epsilon3, form_parity, eta)}: "
                f"{difference}"
            )
    masters = {
        (epsilon2, eta): values[(epsilon2, 0, 0, eta)]
        for epsilon2 in (0, 1)
        for eta in (1, -1)
    }
    return masters, certificates


def product_matches_between_samples(certificates_by_sample, discrete):
    """Return Ramond momentum sheets with sample-independent kappa^2."""

    first = certificates_by_sample[0][discrete]["candidates"]
    second = certificates_by_sample[1][discrete]["candidates"]
    return tuple(
        sheets
        for sheets in first
        if sp.factor(sp.cancel(first[sheets] - second[sheets])) == 0
    )


def full_grid_report():
    """Run the requested 27 level triples and all 16 restrictions."""

    kernel_self_checks()
    total_restrictions = 0
    product_successes = 0
    product_failures = 0
    print("FULL NS--R--R THREE-POINT GRID")
    print("samples:", SAMPLES)
    for labels in itertools.product(
        GRID_NS_LEVELS, GRID_R_LEVELS, GRID_R_LEVELS
    ):
        sample_results = []
        master_values = []
        for sample in SAMPLES:
            masters, certificates = check_discrete_reduction(labels, sample)
            master_values.append(masters)
            sample_results.append(certificates)
        matches = {}
        for discrete in DISCRETE_CHOICES:
            sheets = product_matches_between_samples(sample_results, discrete)
            matches[discrete] = sheets
            total_restrictions += 1
            if sheets:
                product_successes += 1
            else:
                product_failures += 1
        print(
            "levels=",
            labels,
            "restrictions=16",
            "single-product matches=",
            sum(bool(value) for value in matches.values()),
            "master fingerprints=",
            tuple(
                sp.N(master_values[0][key], 8)
                for key in ((0, 1), (0, -1), (1, 1), (1, -1))
            ),
        )
    print(
        "grid summary:",
        f"{total_restrictions} exact restrictions; ",
        f"single-product matches={product_successes}; ",
        f"single-product failures={product_failures}",
    )


def independent_high_check():
    """Check n1=3/2 with both Ramond legs at level at least 3/4."""

    labels = (
        sp.Rational(3, 2),
        sp.Rational(3, 4),
        sp.Rational(3, 4),
    )
    masters, _ = check_discrete_reduction(labels, SAMPLES[0])
    print(
        "independent check:",
        labels,
        "16 restrictions, master fingerprints=",
        tuple(
            sp.N(masters[key], 8)
            for key in ((0, 1), (0, -1), (1, 1), (1, -1))
        ),
    )


def master_table_report():
    """Print the four continuous master values for all 27 level triples."""

    kernel_self_checks()
    print("MASTER B^2 TABLE AT", SAMPLES[0])
    print("columns: n1 n2 n3 | (epsilon2,eta)=(0,+),(0,-),(1,+),(1,-)")
    for labels in itertools.product(
        GRID_NS_LEVELS, GRID_R_LEVELS, GRID_R_LEVELS
    ):
        values = []
        for epsilon2, eta in ((0, 1), (0, -1), (1, 1), (1, -1)):
            value = normalized_branching_square(
                labels, (epsilon2, 0, 0, eta), SAMPLES[0]
            )[1]
            values.append(sp.N(value / I, 10))
        print(*labels, "|", *values)


def reflected_sector_report():
    """Check (n_3,P_3,eta)->(-n_3,-P_3,-eta) on selected states."""

    kernel_self_checks()
    cases = (
        (sp.Integer(0), sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Integer(1), sp.Rational(5, 4), sp.Rational(3, 4)),
    )
    for labels in cases:
        reflected_labels = (labels[0], labels[1], -labels[2])
        sample = SAMPLES[0]
        reflected_sample = (sample[0], sample[1], sample[2], -sample[3])
        for epsilon2, epsilon3, form_parity, eta in DISCRETE_CHOICES:
            raw = enlarged_raw_three_point(
                *labels,
                epsilon2,
                epsilon3,
                form_parity,
                eta,
                *sample,
            )[1]
            reflected = enlarged_raw_three_point(
                *reflected_labels,
                epsilon2,
                epsilon3,
                form_parity,
                -eta,
                *reflected_sample,
            )[1]
            residual = sp.factor(
                sp.cancel(raw - (-1) ** form_parity * reflected)
            )
            if residual != 0:
                raise AssertionError(
                    "Reflected-sector mismatch at "
                    f"labels={labels}, "
                    f"choice={(epsilon2, epsilon3, form_parity, eta)}: "
                    f"{residual}"
                )
        print("reflected sector:", labels, "16 restrictions residual=0")


def exploratory_report():
    """Print a small factorization table before running the full grid."""

    kernel_self_checks()
    cases = (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(1, 4)),
        (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(3, 4)),
    )
    for labels in cases:
        result = direct_certificate(
            *labels,
            0,
            0,
            0,
            1,
            SAMPLES[0],
            check_ward=labels == cases[0],
        )
        print(
            "labels=",
            labels,
            "g=",
            result["g"],
            "raw=",
            result["raw"],
        )
        for sheets, candidate in result["candidates"].items():
            print("  sheets", sheets, "kappa^2=", candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 432 requested restrictions at both exact samples",
    )
    parser.add_argument(
        "--high-check",
        action="store_true",
        help="also run the independent n1=3/2, n2=n3=3/4 test",
    )
    parser.add_argument(
        "--norm-check",
        action="store_true",
        help="check the direct raw Ramond norms through n=5/4",
    )
    parser.add_argument(
        "--master-table",
        action="store_true",
        help="print the four B^2/i master values on all 27 level triples",
    )
    parser.add_argument(
        "--reflection-check",
        action="store_true",
        help="check both P and -P Ramond sectors on selected low states",
    )
    arguments = parser.parse_args()
    if arguments.norm_check:
        direct_norm_report()
    if arguments.full:
        full_grid_report()
    if arguments.master_table:
        master_table_report()
    if arguments.reflection_check:
        reflected_sector_report()
    elif (
        not arguments.full
        and not arguments.norm_check
        and not arguments.master_table
        and not arguments.high_check
    ):
        exploratory_report()
    if arguments.high_check:
        independent_high_check()


if __name__ == "__main__":
    main()
