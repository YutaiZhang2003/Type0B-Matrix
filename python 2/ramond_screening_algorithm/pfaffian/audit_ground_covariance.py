"""Exact finite checks for the Ramond-ground-resolved chi Pfaffian.

This is a calibration harness, not a production evaluator.  It expands the
few chi strings used in the checks into their literal auxiliary/physical
Fock paths and compares that answer with a single combined covariance.  No
super-Virasoro PBW state enters: both factors are free Majorana theories.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHI_DIR = ROOT / "python 2" / "nsrr_chi_branching"
GRID_DIR = ROOT / "python 2" / "ramond_three_point_grid"
for directory in (CHI_DIR, GRID_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as grid  # noqa: E402
import nsrr_chi_formula as chi  # noqa: E402

from .core import pfaffian  # noqa: E402
from .screening_pfaffian import (  # noqa: E402
    ExternalRow,
    I,
    _row_pair_with_screening,
    _screen_kernel,
    vandermonde,
)
from .boundary_zero_modes import external_pair  # noqa: E402
from .selberg_jack import normalized_selberg_average  # noqa: E402
from .special_oracle import (  # noqa: E402
    ordinary_selberg,
    physical_nsrr_selberg,
)


def physical_majorana_value(
    target_form_parity,
    eta,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
    screenings=(),
):
    """Free physical-Majorana form with the SCblock ground matrix."""

    target = sp.Matrix(
        [
            [
                grid.physical_ground(target_form_parity, eta, row, column)
                for column in (0, 1)
            ]
            for row in (0, 1)
        ]
    )
    flip = sp.Matrix(((0, 1), (1, 0)))
    # N screening fermions act on the ground matrix from the zero side.
    # Precompose by the inverse flip so that the screened primary form is
    # exactly the requested SCblock Gamma_f^eta, entry by entry.
    base = target * (flip ** (len(screenings) % 2))

    def ground(second, third):
        return base[int(second), int(third)]

    fields = (
        tuple(("external", "one", sp.Integer(mode)) for mode in second_modes)
        + tuple(("screening", "screen", sp.sympify(value)) for value in screenings)
        + tuple(("external", "zero", sp.Integer(mode)) for mode in third_modes)
    )
    size = len(fields)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    for left in range(size):
        kind_left, leg_left, value_left = fields[left]
        for right in range(left + 1, size):
            kind_right, leg_right, value_right = fields[right]
            if kind_left == kind_right == "screening":
                value = _screen_kernel(value_left, value_right)
            elif kind_left == "external" and kind_right == "screening":
                value = _row_pair_with_screening(leg_left, value_left, value_right)
            elif kind_left == "screening" and kind_right == "external":
                value = _row_pair_with_screening(leg_right, value_right, value_left)
            else:
                value = grid.fermion_pair_coefficient(
                    {"one": 2, "zero": 3}[leg_left],
                    value_left,
                    {"one": 2, "zero": 3}[leg_right],
                    value_right,
                )
            covariance[left][right] = value
            covariance[right][left] = -value

    if size % 2 == 0:
        return sp.factor(
            pfaffian(covariance)
            * ground(second_ground, third_ground)
        )

    mean = []
    for kind, leg, value in fields:
        if kind == "screening":
            one = sp.sqrt(2) / 2
            ground_value = ground(second_ground, 1 - third_ground)
        elif leg == "one":
            one = grid.fermion_one_coefficient(2, value)
            ground_value = ground(1 - second_ground, third_ground)
        else:
            one = grid.fermion_one_coefficient(3, value)
            ground_value = ground(second_ground, 1 - third_ground)
        mean.append(sp.factor(one * ground_value))
    augmented = [row + [mean[index]] for index, row in enumerate(covariance)]
    augmented.append([-entry for entry in mean] + [sp.Integer(0)])
    return sp.factor(pfaffian(augmented))


def direct_free_field(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    screenings=(),
):
    """Literal path sum for the two Ramond legs and optional screenings."""

    auxiliary_form = (int(epsilon2) + int(epsilon3) - int(form_parity)) % 2
    answer = sp.Integer(0)
    for state2, coefficient2 in chi.ramond_fock_paths(n2, epsilon2):
        aux2, aux_ground2, physical2, physical_ground2 = state2
        for state3, coefficient3 in chi.ramond_fock_paths(n3, epsilon3):
            aux3, aux_ground3, physical3, physical_ground3 = state3
            auxiliary = grid.fermion_value(
                auxiliary_form,
                (),
                aux2,
                aux_ground2,
                aux3,
                aux_ground3,
            )
            if auxiliary == 0:
                continue
            physical = physical_majorana_value(
                form_parity,
                eta,
                physical2,
                physical_ground2,
                physical3,
                physical_ground3,
                screenings,
            )
            if physical == 0:
                continue
            physical_parity2 = (len(physical2) + physical_ground2) % 2
            auxiliary_parity3 = (len(aux3) + aux_ground3) % 2
            # The tensor-product Koszul sign belongs to the three external
            # states.  Screening operators are evaluated only after the
            # physical form has been separated, so they do not contribute
            # an additional crossing through the auxiliary third state.
            koszul = (-1) ** (physical_parity2 * auxiliary_parity3)
            answer += (
                coefficient2
                * coefficient3
                # ``ramond_fock_paths`` uses the free-field minus ground.
                # The SCblock Ward basis is
                # |Delta,->=-exp(-i*pi/4)|free minus> on each R leg.
                * (-((1 - I) / sp.sqrt(2)))
                ** (physical_ground2 + physical_ground3)
                * koszul
                * auxiliary
                * physical
            )
    return sp.factor(answer)


def direct_fixed_free_field(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    screenings,
    auxiliary_ground2,
    auxiliary_ground3,
):
    """Literal path sum restricted to one pair of zero-mode endpoints."""

    auxiliary_form = (int(epsilon2) + int(epsilon3) - int(form_parity)) % 2
    conversion = -(1 - I) / sp.sqrt(2)
    answer = sp.Integer(0)
    for state2, coefficient2 in chi.ramond_fock_paths(n2, epsilon2):
        aux2, ground2, physical2, physical_ground2 = state2
        if ground2 != int(auxiliary_ground2):
            continue
        for state3, coefficient3 in chi.ramond_fock_paths(n3, epsilon3):
            aux3, ground3, physical3, physical_ground3 = state3
            if ground3 != int(auxiliary_ground3):
                continue
            auxiliary = grid.fermion_value(
                auxiliary_form, (), aux2, ground2, aux3, ground3
            )
            physical = physical_majorana_value(
                form_parity,
                eta,
                physical2,
                physical_ground2,
                physical3,
                physical_ground3,
                screenings,
            )
            physical_parity2 = (len(physical2) + physical_ground2) % 2
            auxiliary_parity3 = (len(aux3) + ground3) % 2
            answer += (
                coefficient2
                * coefficient3
                * conversion ** (physical_ground2 + physical_ground3)
                * (-1) ** (physical_parity2 * auxiliary_parity3)
                * auxiliary
                * physical
            )
    return sp.factor(answer)


def _ground_matrix(sector, form_parity, eta):
    if sector == "auxiliary":
        return sp.Matrix(
            [
                [
                    grid.boundary.fermion_ground(form_parity, row, column)
                    for column in (0, 1)
                ]
                for row in (0, 1)
            ]
        )
    target = sp.Matrix(
        [
            [
                grid.physical_ground(form_parity, eta, row, column)
                for column in (0, 1)
            ]
            for row in (0, 1)
        ]
    )
    return target


@dataclass(frozen=True)
class PhysicalInsertion:
    """One ordered insertion seen by a physical covariance callback.

    An external insertion has ``kind='external'``, its puncture in ``leg``,
    its positive contour-mode magnitude in ``value``, and the complete
    literal-chain coefficient in ``coefficient``.  A screening has
    ``kind=leg='screening'``, its coordinate in ``value``, and coefficient
    one.  A reflected ``psi^R`` implementation can therefore replace the
    physical covariance without changing zero-mode or Koszul bookkeeping.
    """

    kind: str
    leg: str
    value: sp.Expr
    coefficient: sp.Expr


@dataclass(frozen=True)
class PhysicalGroundContext:
    """The retained physical Ramond ground indices for one Pfaffian."""

    form_parity: int
    eta: int
    ground2: int
    ground3: int
    matrix: sp.Matrix
    screenings: tuple[sp.Expr, ...]


def _physical_insertion(kind, item):
    if kind == "screening":
        return PhysicalInsertion("screening", "screening", item, sp.Integer(1))
    return PhysicalInsertion(
        "external",
        item.leg,
        item.mode,
        item.physical_coefficient,
    )


def ordinary_physical_covariance(left, right, context):
    """The ordinary same-sheet two-spin Majorana covariance.

    The return value includes both literal-chain coefficients.  ``context``
    is accepted so that this function has exactly the same interface as a
    momentum-dependent reflected covariance.
    """

    del context
    if left.kind == right.kind == "screening":
        return _screen_kernel(left.value, right.value)
    if left.kind == "external" and right.kind == "screening":
        return left.coefficient * _row_pair_with_screening(
            left.leg, left.value, right.value
        )
    if left.kind == "screening" and right.kind == "external":
        return right.coefficient * _row_pair_with_screening(
            right.leg, right.value, left.value
        )
    return (
        left.coefficient
        * right.coefficient
        * external_pair(left.leg, left.value, right.leg, right.value)
    )


def ordinary_physical_mean(insertion, context):
    """The ordinary odd physical functional with its ground-index flip."""

    matrix = context.matrix
    ground2 = context.ground2
    ground3 = context.ground3
    if insertion.kind == "screening":
        return sp.sqrt(2) / 2 * matrix[ground2, 1 - ground3]
    if insertion.leg == "one":
        one = grid.fermion_one_coefficient(2, insertion.value)
        ground_value = matrix[1 - ground2, ground3]
    elif insertion.leg == "zero":
        one = grid.fermion_one_coefficient(3, insertion.value)
        ground_value = matrix[ground2, 1 - ground3]
    else:
        raise ValueError(f"unsupported Ramond insertion leg: {insertion.leg}")
    return sp.factor(insertion.coefficient * one * ground_value)


def _nonzero_rows(
    n2,
    n3,
    screen_count,
    auxiliary_ground2,
    auxiliary_ground3,
    leg3_auxiliary_twist=True,
):
    maximum2 = int(2 * sp.Rational(n2) - sp.Rational(1, 2))
    maximum3 = int(2 * sp.Rational(n3) - sp.Rational(1, 2))
    rows = []
    for mode in range(maximum2, 0, -1):
        rows.append(
            ExternalRow(
                "one",
                sp.Integer(mode),
                1,
                -I
                * (-1) ** (
                    int(auxiliary_ground2) + int(auxiliary_ground3)
                ),
            )
        )
    auxiliary3 = (
        (-1) ** (
            1 - int(auxiliary_ground2) + int(screen_count)
        )
        if leg3_auxiliary_twist
        else 1
    )
    for mode in range(maximum3, 0, -1):
        rows.append(
            ExternalRow(
                "zero",
                sp.Integer(mode),
                auxiliary3,
                -I * (-1) ** int(auxiliary_ground3),
            )
        )
    return tuple(rows)


def combined_fixed_candidate(
    n2,
    n3,
    form_parity,
    eta,
    screenings,
    auxiliary_ground2,
    auxiliary_ground3,
    leg3_auxiliary_twist=True,
):
    """One fixed zero sector evaluated by a single combined Pfaffian."""

    screenings = tuple(screenings)
    count = len(screenings)
    physical_ground2 = 1 - int(auxiliary_ground2)
    physical_ground3 = 1 - int(auxiliary_ground3)
    epsilon2 = (int(2 * sp.Rational(n2) - sp.Rational(1, 2)) + 1) % 2
    epsilon3 = (int(2 * sp.Rational(n3) - sp.Rational(1, 2)) + 1) % 2
    auxiliary_form = (epsilon2 + epsilon3 - int(form_parity)) % 2
    auxiliary_matrix = _ground_matrix(
        "auxiliary", auxiliary_form, eta
    )
    flip = sp.Matrix(((0, 1), (1, 0)))
    physical_matrix = _ground_matrix("physical", form_parity, eta) * (
        flip ** (count % 2)
    )

    rows = _nonzero_rows(
        n2,
        n3,
        count,
        auxiliary_ground2,
        auxiliary_ground3,
        leg3_auxiliary_twist,
    )
    one_count = sum(row.leg == "one" for row in rows)
    objects = (
        tuple(("external", row) for row in rows[:one_count])
        + tuple(("screening", value) for value in screenings)
        + tuple(("external", row) for row in rows[one_count:])
    )
    auxiliary_required = (
        auxiliary_form + int(auxiliary_ground2) + int(auxiliary_ground3)
    ) % 2
    physical_matrix_parity = (int(form_parity) + count) % 2
    physical_required = (
        physical_matrix_parity + physical_ground2 + physical_ground3
    ) % 2
    if auxiliary_required + physical_required != 1:
        raise AssertionError((auxiliary_required, physical_required))

    odd_sector = "auxiliary" if auxiliary_required else "physical"
    even_sector = "physical" if auxiliary_required else "auxiliary"
    even_matrix = physical_matrix if even_sector == "physical" else auxiliary_matrix
    even_ground2 = physical_ground2 if even_sector == "physical" else auxiliary_ground2
    even_ground3 = physical_ground3 if even_sector == "physical" else auxiliary_ground3
    even_scalar = even_matrix[int(even_ground2), int(even_ground3)]

    size = len(objects)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    border = [sp.Integer(0)] * size
    odd_matrix = physical_matrix if odd_sector == "physical" else auxiliary_matrix
    odd_ground2 = physical_ground2 if odd_sector == "physical" else auxiliary_ground2
    odd_ground3 = physical_ground3 if odd_sector == "physical" else auxiliary_ground3

    for index, (kind, item) in enumerate(objects):
        if kind == "screening":
            if odd_sector == "physical":
                border[index] = (
                    sp.sqrt(2)
                    / 2
                    * odd_matrix[int(odd_ground2), 1 - int(odd_ground3)]
                )
            continue
        row = item
        coefficient_value = (
            row.auxiliary_coefficient
            if odd_sector == "auxiliary"
            else row.physical_coefficient
        )
        if row.leg == "one":
            matrix_value = odd_matrix[1 - int(odd_ground2), int(odd_ground3)]
            one_value = grid.fermion_one_coefficient(2, row.mode)
        else:
            matrix_value = odd_matrix[int(odd_ground2), 1 - int(odd_ground3)]
            one_value = grid.fermion_one_coefficient(3, row.mode)
        border[index] = sp.factor(coefficient_value * one_value * matrix_value)

    for left in range(size):
        kind_left, item_left = objects[left]
        for right in range(left + 1, size):
            kind_right, item_right = objects[right]
            if kind_left == kind_right == "screening":
                value = _screen_kernel(item_left, item_right)
            elif kind_left == "external" and kind_right == "screening":
                value = item_left.physical_coefficient * _row_pair_with_screening(
                    item_left.leg, item_left.mode, item_right
                )
            elif kind_left == "screening" and kind_right == "external":
                value = item_right.physical_coefficient * _row_pair_with_screening(
                    item_right.leg, item_right.mode, item_left
                )
            else:
                pair = external_pair(
                    item_left.leg, item_left.mode,
                    item_right.leg, item_right.mode,
                )
                value = pair * (
                    item_left.auxiliary_coefficient
                    * item_right.auxiliary_coefficient
                    + item_left.physical_coefficient
                    * item_right.physical_coefficient
                )
            covariance[left][right] = value
            covariance[right][left] = -value
    augmented = [row + [border[index]] for index, row in enumerate(covariance)]
    augmented.append([-entry for entry in border] + [sp.Integer(0)])
    return sp.factor(even_scalar * pfaffian(augmented))


def combined_candidate(n2, n3, form_parity, eta, screenings, twist=True):
    """Sum the four zero sectors, including literal chain coefficients."""

    maximum2 = int(2 * sp.Rational(n2) - sp.Rational(1, 2))
    maximum3 = int(2 * sp.Rational(n3) - sp.Rational(1, 2))
    reorder = (-1) ** (
        maximum2 * (maximum2 + 1) // 2
        + maximum3 * (maximum3 + 1) // 2
    )
    conversion = -(1 - I) / sp.sqrt(2)
    answer = sp.Integer(0)
    for auxiliary_ground2 in (0, 1):
        for auxiliary_ground3 in (0, 1):
            physical_ground2 = 1 - auxiliary_ground2
            physical_ground3 = 1 - auxiliary_ground3
            zero = (
                sp.Rational(1, 2)
                * (-I * conversion) ** (physical_ground2 + physical_ground3)
                * (-1) ** (physical_ground2 * auxiliary_ground3)
            )
            answer += zero * combined_fixed_candidate(
                n2,
                n3,
                form_parity,
                eta,
                screenings,
                auxiliary_ground2,
                auxiliary_ground3,
                leg3_auxiliary_twist=twist,
            )
    return sp.factor(reorder * answer)


def _literal_branch_action(branch_label, parity, zero_choices, physical_rows):
    """Act one chi chain with only its zero choices expanded explicitly."""

    chain = chi.ramond_chi_chain(branch_label, parity)
    zero_choices = iter(zero_choices)
    tagged = []
    for position, (mode, realization) in enumerate(chain):
        if mode == 0:
            tagged.append((mode, realization, next(zero_choices)))
        else:
            tagged.append(
                (mode, realization, "physical" if position in physical_rows else "auxiliary")
            )
    state = ((), 0, (), 0)
    outer = sp.Integer(1)
    for mode, realization, selection in reversed(tagged):
        auxiliary_modes, auxiliary_ground, physical_modes, physical_ground = state
        if selection == "auxiliary":
            final, coefficient_value = chi._fermion_action(  # noqa: SLF001
                mode, auxiliary_modes, auxiliary_ground
            )
            if coefficient_value == 0:
                return None, sp.Integer(0)
            state = (final[0], final[1], physical_modes, physical_ground)
            outer *= coefficient_value
        else:
            final, coefficient_value = chi._fermion_action(  # noqa: SLF001
                mode,
                physical_modes,
                physical_ground,
                zero_sign=1 if realization == -1 else -1,
            )
            if coefficient_value == 0:
                return None, sp.Integer(0)
            outer *= -I * (-1) ** ((len(auxiliary_modes) + auxiliary_ground) % 2)
            outer *= coefficient_value
            state = (auxiliary_modes, auxiliary_ground, final[0], final[1])
    return state, sp.factor(outer)


def literal_branch_sectors(branch_label, parity):
    """Constant zero-mode sectors and compressed nonzero row weights."""

    chain = chi.ramond_chi_chain(branch_label, parity)
    zero_positions = tuple(index for index, item in enumerate(chain) if item[0] == 0)
    row_positions = tuple(index for index, item in enumerate(chain) if item[0] != 0)
    answer = []
    for zero_choices in itertools.product(("auxiliary", "physical"), repeat=len(zero_positions)):
        state, base = _literal_branch_action(
            branch_label, parity, zero_choices, frozenset()
        )
        if base == 0:
            continue
        row_ratios = []
        for position in row_positions:
            switched_state, switched = _literal_branch_action(
                branch_label, parity, zero_choices, frozenset((position,))
            )
            if switched == 0:
                raise AssertionError((branch_label, parity, zero_choices, position))
            if switched_state[1] != state[1] or switched_state[3] != state[3]:
                raise AssertionError("a nonzero mode changed a Ramond ground label")
            row_ratios.append(sp.factor(switched / base))
        modes = tuple(-chain[position][0] for position in row_positions)
        # Rows are used in the canonical descending mode order of compute_grid.
        permutation = tuple(sorted(range(len(modes)), key=lambda index: modes[index], reverse=True))
        answer.append(
            (
                state[1],
                state[3],
                sp.factor(base),
                tuple(modes[index] for index in permutation),
                tuple(row_ratios[index] for index in permutation),
                zero_choices,
            )
        )
    return tuple(answer)


def compressed_zero_sector_data(n2, n3, epsilon2, epsilon3, screenings):
    """Yield ground endpoints, global prefactor, and chi row coefficients."""

    count = len(tuple(screenings))
    conversion = -(1 - I) / sp.sqrt(2)
    for sector2 in literal_branch_sectors(n2, epsilon2):
        a2, p2, coefficient2, modes2, ratios2, choices2 = sector2
        for sector3 in literal_branch_sectors(n3, epsilon3):
            a3, p3, coefficient3, modes3, ratios3, choices3 = sector3
            auxiliary_parity3 = (len(modes3) + a3) % 2
            base_koszul = (-1) ** (p2 * auxiliary_parity3)
            prefactor = sp.factor(
                coefficient2
                * coefficient3
                * conversion ** (p2 + p3)
                * base_koszul
                * (-1) ** (count * len(modes3))
            )

            rows = []
            total_external = len(modes2) + len(modes3)
            for index, (mode, local_ratio) in enumerate(zip(modes2, ratios2)):
                desired_ratio = local_ratio * (-1) ** auxiliary_parity3
                later_external = total_external - index - 1
                delta_inversions = later_external
                physical_coefficient = sp.factor(
                    desired_ratio * sp.Integer(-1) ** delta_inversions
                )
                rows.append(ExternalRow("one", mode, 1, physical_coefficient))
            for local_index, (mode, local_ratio) in enumerate(zip(modes3, ratios3)):
                index = len(modes2) + local_index
                desired_ratio = local_ratio * (-1) ** p2
                later_external = total_external - index - 1
                # Every screening precedes the zero-leg rows in object order.
                delta_inversions = later_external - count
                physical_coefficient = sp.factor(
                    desired_ratio * sp.Integer(-1) ** delta_inversions
                )
                rows.append(ExternalRow("zero", mode, 1, physical_coefficient))
            yield (
                a2,
                p2,
                a3,
                p3,
                prefactor,
                tuple(rows),
                choices2,
                choices3,
            )


def compressed_fixed_correlator(
    rows,
    auxiliary_form,
    physical_target,
    eta,
    screenings,
    auxiliary_ground2,
    physical_ground2,
    auxiliary_ground3,
    physical_ground3,
    *,
    physical_covariance=ordinary_physical_covariance,
    physical_mean_kernel=ordinary_physical_mean,
):
    """One constant ground sector; all nonzero assignments are one Pfaffian.

    ``physical_covariance(left, right, context)`` supplies the complete
    ordered physical contraction, including the coefficients stored on the
    two insertions.  ``physical_mean_kernel(insertion, context)`` supplies
    the odd physical border.  The defaults are the ordinary same-sheet
    Majorana kernels.  These hooks test a proposed *Gaussian reduction* of
    a reflected form.  They are not, by themselves, a general reflection
    interface: already ``R_1 psi_-1`` contains a bosonic ``c_-1`` state,
    and higher reflection blocks contain products of currents.  Such terms
    must first be converted to power-sum insertions in the Selberg
    polynomial (or supplied here only after that reduction).
    """

    screenings = tuple(screenings)
    count = len(screenings)
    auxiliary_matrix = _ground_matrix("auxiliary", auxiliary_form, eta)
    flip = sp.Matrix(((0, 1), (1, 0)))
    physical_matrix = _ground_matrix("physical", physical_target, eta) * (
        flip ** (count % 2)
    )
    physical_matrix_parity = (int(physical_target) + count) % 2
    auxiliary_required = (
        int(auxiliary_form) + int(auxiliary_ground2) + int(auxiliary_ground3)
    ) % 2
    physical_required = (
        physical_matrix_parity + int(physical_ground2) + int(physical_ground3)
    ) % 2
    auxiliary_scalar = auxiliary_matrix[
        int(auxiliary_ground2), int(auxiliary_ground3)
    ]
    physical_scalar = physical_matrix[
        int(physical_ground2), int(physical_ground3)
    ]

    one_count = sum(row.leg == "one" for row in rows)
    objects = (
        tuple(("external", row) for row in rows[:one_count])
        + tuple(("screening", value) for value in screenings)
        + tuple(("external", row) for row in rows[one_count:])
    )
    physical_insertions = tuple(
        _physical_insertion(kind, item) for kind, item in objects
    )
    physical_context = PhysicalGroundContext(
        int(physical_target),
        int(eta),
        int(physical_ground2),
        int(physical_ground3),
        physical_matrix,
        screenings,
    )
    size = len(objects)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    auxiliary_mean = [sp.Integer(0)] * size
    physical_mean = [sp.Integer(0)] * size

    for index, (kind, item) in enumerate(objects):
        physical_mean[index] = physical_mean_kernel(
            physical_insertions[index], physical_context
        )
        if kind == "screening":
            continue
        if item.leg == "one":
            one = grid.fermion_one_coefficient(2, item.mode)
            auxiliary_ground_value = auxiliary_matrix[
                1 - int(auxiliary_ground2), int(auxiliary_ground3)
            ]
        else:
            one = grid.fermion_one_coefficient(3, item.mode)
            auxiliary_ground_value = auxiliary_matrix[
                int(auxiliary_ground2), 1 - int(auxiliary_ground3)
            ]
        auxiliary_mean[index] = sp.factor(
            item.auxiliary_coefficient * one * auxiliary_ground_value
        )
        # The physical mean was supplied above.  These local quantities are
        # retained only for the independent auxiliary border.

    for left in range(size):
        kind_left, item_left = objects[left]
        for right in range(left + 1, size):
            kind_right, item_right = objects[right]
            value = physical_covariance(
                physical_insertions[left],
                physical_insertions[right],
                physical_context,
            )
            if kind_left == kind_right == "external":
                pair = external_pair(
                    item_left.leg, item_left.mode,
                    item_right.leg, item_right.mode,
                )
                value += pair * (
                    item_left.auxiliary_coefficient
                    * item_right.auxiliary_coefficient
                )
            covariance[left][right] = value
            covariance[right][left] = -value

    if not auxiliary_required and not physical_required:
        return sp.factor(
            auxiliary_scalar * physical_scalar * pfaffian(covariance)
        )
    if auxiliary_required and physical_required:
        unupdated = pfaffian(covariance)
        for left in range(size):
            for right in range(left + 1, size):
                update = (
                    auxiliary_mean[left] * physical_mean[right]
                    - physical_mean[left] * auxiliary_mean[right]
                )
                covariance[left][right] += update
                covariance[right][left] -= update
        # The rank-two update selects one unpaired auxiliary and one
        # unpaired physical field.  Subtract the zero-update term, which
        # would incorrectly contract both odd functionals as if they were
        # even ground forms.
        return sp.factor(pfaffian(covariance) - unupdated)

    border = auxiliary_mean if auxiliary_required else physical_mean
    even_scalar = physical_scalar if auxiliary_required else auxiliary_scalar
    augmented = [row + [border[index]] for index, row in enumerate(covariance)]
    augmented.append([-entry for entry in border] + [sp.Integer(0)])
    return sp.factor(even_scalar * pfaffian(augmented))


def compressed_correlator(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    screenings,
    *,
    physical_covariance=ordinary_physical_covariance,
    physical_mean_kernel=ordinary_physical_mean,
):
    """Ground-resolved constant-size sum of Pfaffians for both R copies.

    The optional callbacks are forwarded unchanged to each fixed ground
    sector.  They may close over momenta and branch-sheet data.
    """

    auxiliary_form = (int(epsilon2) + int(epsilon3) - int(form_parity)) % 2
    answer = sp.Integer(0)
    for data in compressed_zero_sector_data(
        n2, n3, epsilon2, epsilon3, screenings
    ):
        a2, p2, a3, p3, prefactor, rows, _, _ = data
        answer += prefactor * compressed_fixed_correlator(
            rows,
            auxiliary_form,
            form_parity,
            eta,
            screenings,
            a2,
            p2,
            a3,
            p3,
            physical_covariance=physical_covariance,
            physical_mean_kernel=physical_mean_kernel,
        )
    return sp.factor(answer)


def compressed_contour_polynomial_with_covariance(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    physical_covariance,
    physical_mean_kernel=ordinary_physical_mean,
):
    """Polynomialized correlator using a supplied physical covariance."""

    n2, n3 = sp.Rational(n2), sp.Rational(n3)
    count = int(2 * (n2 + n3))
    if count < 0:
        raise ValueError("the selected reflected hyperplane has negative screening number")
    screenings = sp.symbols(f"t0:{count}")
    correlator = compressed_correlator(
        n2,
        n3,
        int(epsilon2),
        int(epsilon3),
        int(form_parity),
        int(eta),
        screenings,
        physical_covariance=physical_covariance,
        physical_mean_kernel=physical_mean_kernel,
    )
    shift_A = int(2 * abs(n3) - sp.Rational(1, 2))
    shift_B = int(2 * abs(n2) - sp.Rational(1, 2))
    clearing = sp.prod(
        value**shift_A * (1 - value) ** shift_B
        for value in screenings
    )
    polynomial = sp.cancel(vandermonde(screenings) * clearing * correlator)
    numerator, denominator = sp.fraction(polynomial)
    if set(screenings) & denominator.free_symbols:
        raise AssertionError((n2, n3, epsilon2, epsilon3, form_parity, eta, denominator))
    polynomial = sp.expand(numerator / denominator)
    return screenings, polynomial, shift_A, shift_B


@lru_cache(None)
def compressed_contour_polynomial(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
):
    """Polynomialized ordinary-kernel maximal-screening correlator."""

    return compressed_contour_polynomial_with_covariance(
        n2,
        n3,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        ordinary_physical_covariance,
        ordinary_physical_mean,
    )


def compressed_selberg_ratio(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    A,
    B,
    g,
):
    """Ordinary-kernel maximal-screening candidate over its primary form."""

    screenings, polynomial, shift_A, shift_B = compressed_contour_polynomial(
        sp.Rational(n2),
        sp.Rational(n3),
        int(epsilon2),
        int(epsilon3),
        int(form_parity),
        int(eta),
    )
    shifted_A = A - shift_A
    shifted_B = B - shift_B
    numerator = normalized_selberg_average(
        polynomial, screenings, shifted_A, shifted_B, g
    )
    numerator *= ordinary_selberg(
        len(screenings), shifted_A, shifted_B, g
    )
    denominator = physical_nsrr_selberg(len(screenings), A, B, g)
    if len(screenings) % 2:
        denominator /= sp.sqrt(2)
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True)
    )


def compressed_selberg_ratio_with_covariance(
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    A,
    B,
    g,
    physical_covariance,
    physical_mean_kernel=ordinary_physical_mean,
):
    """Maximal-screening ratio with an injected reflected kernel.

    This is the executable integration path for a future ``psi^R``
    covariance.  No Ward or PBW object is used here.
    """

    screenings, polynomial, shift_A, shift_B = (
        compressed_contour_polynomial_with_covariance(
            sp.Rational(n2),
            sp.Rational(n3),
            int(epsilon2),
            int(epsilon3),
            int(form_parity),
            int(eta),
            physical_covariance,
            physical_mean_kernel,
        )
    )
    shifted_A = A - shift_A
    shifted_B = B - shift_B
    numerator = normalized_selberg_average(
        polynomial, screenings, shifted_A, shifted_B, g
    )
    numerator *= ordinary_selberg(
        len(screenings), shifted_A, shifted_B, g
    )
    denominator = physical_nsrr_selberg(len(screenings), A, B, g)
    if len(screenings) % 2:
        denominator /= sp.sqrt(2)
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True)
    )


def _assert_zero(expression, label):
    expression = sp.factor(expression)
    if expression != 0 and sp.simplify(expression) != 0:
        raise AssertionError((label, expression))


def audit_compression():
    """Compare one-Pfaffian compression with every literal chi path."""

    cases = (
        (sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Rational(5, 4), sp.Rational(5, 4)),
        (sp.Rational(5, 4), -sp.Rational(5, 4)),
    )
    checks = 0
    for n2, n3 in cases:
        count = int(2 * (n2 + n3))
        screenings = tuple(
            sp.Rational(index + 1, count + 2)
            for index in range(count)
        )
        for epsilon2, epsilon3, form_parity, eta in itertools.product(
            (0, 1), (0, 1), (0, 1), (1, -1)
        ):
            direct = direct_free_field(
                n2,
                n3,
                epsilon2,
                epsilon3,
                form_parity,
                eta,
                screenings,
            )
            compressed = compressed_correlator(
                n2,
                n3,
                epsilon2,
                epsilon3,
                form_parity,
                eta,
                screenings,
            )
            _assert_zero(
                direct - compressed,
                (n2, n3, epsilon2, epsilon3, form_parity, eta),
            )
            checks += 1
    print(f"ground-resolved compression: {checks} exact literal-path checks passed")


def audit_ward_boundary():
    """Certify the denominator and exhibit the reflected-kernel obstruction."""

    from .boundary_zero_modes import projected_contour_polynomial

    b = sp.Rational(3, 2)
    q = b + 1 / b
    p2 = sp.Rational(2, 5)
    p3 = sp.Rational(3, 7)
    A = -b * (q / 2 + p3) - sp.Rational(1, 2)
    B = -b * (q / 2 + p2) - sp.Rational(1, 2)
    g = -b * q / 2

    # At the Ramond ground, the full matrix-valued screening functional is
    # fixed by Gamma_f^eta and the BFL denominator.  Check all 16 copy/form
    # choices, not only the natural branch.
    ground = sp.Rational(1, 4)
    p1 = -b - q / 2 - p2 - p3
    for epsilon2, epsilon3, form_parity, eta in itertools.product(
        (0, 1), (0, 1), (0, 1), (1, -1)
    ):
        calculated = compressed_selberg_ratio(
            ground,
            ground,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            A,
            B,
            g,
        )
        expected = grid.enlarged_raw_three_point(
            0,
            ground,
            ground,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            b,
            p1,
            p2,
            p3,
        )[1]
        _assert_zero(
            calculated - expected,
            ("screened ground", epsilon2, epsilon3, form_parity, eta),
        )
    print("BFL denominator and 2x2 ground map: all 16 ground values passed")

    # The charge-preserving half-difference is the certified K channel.
    hard = sp.Rational(3, 4)
    count = 3
    p1 = -count * b - q / 2 - p2 - p3
    for form_parity in (0, 1):
        ts, polynomial, shift_A, shift_B = projected_contour_polynomial(
            0, hard, hard, form_parity, 1
        )
        numerator = normalized_selberg_average(
            polynomial, ts, A - shift_A, B - shift_B, g
        )
        numerator *= ordinary_selberg(
            count, A - shift_A, B - shift_B, g
        )
        denominator = physical_nsrr_selberg(count, A, B, g) / sp.sqrt(2)
        calculated = sp.factor(
            sp.powsimp(
                sp.cancel(sp.expand_func(numerator / denominator)), force=True
            )
        )
        expected = grid.enlarged_raw_three_point(
            0, hard, hard, 0, 0, form_parity, 1,
            b, p1, p2, p3,
        )[1]
        _assert_zero(calculated - expected, ("hard K", form_parity))
    print("positive maximal-screening projection: hard eta=+ passed for f=0,1")

    # Reflection identifies the crossed sheet, but an ordinary two-spin
    # covariance is not its kernel.  Ground mixed-sign branches still pass;
    # the first excited pair gives a nonzero exact residual.
    reflected_p1 = -q / 2 - p2 + p3
    for epsilon2, epsilon3, form_parity, eta in itertools.product(
        (0, 1), (0, 1), (0, 1), (1, -1)
    ):
        calculated = compressed_correlator(
            ground, -ground, epsilon2, epsilon3, form_parity, eta, ()
        )
        expected = grid.enlarged_raw_three_point(
            0,
            ground,
            -ground,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            b,
            reflected_p1,
            p2,
            -p3,
        )[1]
        _assert_zero(
            calculated - expected,
            ("reflected ground", epsilon2, epsilon3, form_parity, eta),
        )

    failures = 0
    obstruction = None
    for epsilon2, epsilon3, form_parity, eta in itertools.product(
        (0, 1), (0, 1), (0, 1), (1, -1)
    ):
        ordinary = compressed_correlator(
            hard,
            -hard,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            (),
            physical_covariance=ordinary_physical_covariance,
            physical_mean_kernel=ordinary_physical_mean,
        )
        crossed = grid.enlarged_raw_three_point(
            0,
            hard,
            -hard,
            epsilon2,
            epsilon3,
            form_parity,
            eta,
            b,
            reflected_p1,
            p2,
            -p3,
        )[1]
        residual = sp.factor(ordinary - crossed)
        if residual != 0:
            failures += 1
            if obstruction is None:
                obstruction = residual
    if failures != 16:
        raise AssertionError(f"expected all 16 reflected hard entries to fail: {failures}")
    print(
        "reflected ground: all 16 passed; reflected hard H: 0/16 passed "
        f"with ordinary covariance; representative residual={obstruction}"
    )


def audit():
    audit_compression()
    audit_ward_boundary()


if __name__ == "__main__":
    audit()
