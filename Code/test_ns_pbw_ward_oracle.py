"""Independent low-order audit of the NS PBW/Ward sewing oracle.

The production recursions are deliberately absent from this file.  The
checks start from the ordinary-c NS algebra, published low-level Gram
matrices, the human-note component table, the superspace invariant, and all
available contour-reduction identities.  Levels are stored as twice-levels.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import sympy as sp


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ns_genus12_finite_c_check import (  # noqa: E402
    NSDescendantThreeForm,
    theta_orientation_sign as numeric_theta_orientation_sign,
)
from ns_genus2_symbolic_low_order import (  # noqa: E402
    ExactDirectThetaOracle,
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    PBW_BASES,
    State,
    exact_pbw_basis,
    generalized_binomial,
    state_parity,
    state_twice_level,
    theta_orientation_sign as exact_theta_orientation_sign,
)
from mixed_ns_ramond_descendant_blocks import NSVermaModule  # noqa: E402
from ns_human_convention import (  # noqa: E402
    human_note_rho_sign,
    primary_parity_ward_sign,
)
from ns_osp_superspace import superspace_three_point  # noqa: E402
from ns_regular_block import THETA_ORIENTATION  # noqa: E402


G: State = (("G", -1),)


def _clean(value: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.together(value))


def _contour_value(
    form: ExactNSDescendantThreeForm,
    states: tuple[State, State, State],
) -> sp.Expr:
    """Undo the full human-note conversion for graded contour identities."""

    sign = human_note_rho_sign(
        tuple(state_parity(state) for state in states),
        form.primary_parities,
    )
    return _clean(sign * form.value(*states))


def _contour_action(
    form: ExactNSDescendantThreeForm,
    *,
    slot: int,
    mode: tuple[str, int],
    states: tuple[State, State, State],
) -> sp.Expr:
    result = sp.S.Zero
    for acted, coefficient in form.modules[slot].mode_action(mode, states[slot]):
        changed = list(states)
        changed[slot] = acted
        result += coefficient * _contour_value(form, tuple(changed))
    return _clean(result)


def _linear_mode_action(
    module: ExactNSVermaModule,
    mode: tuple[str, int],
    vector: dict[State, sp.Expr],
) -> dict[State, sp.Expr]:
    result: dict[State, sp.Expr] = {}
    for state, coefficient in vector.items():
        for acted, action_coefficient in module.mode_action(mode, state):
            result[acted] = _clean(
                result.get(acted, sp.S.Zero) + coefficient * action_coefficient
            )
    return {state: value for state, value in result.items() if value != 0}


def _vector_difference(
    left: dict[State, sp.Expr], right: dict[State, sp.Expr]
) -> dict[State, sp.Expr]:
    states = set(left) | set(right)
    return {
        state: value
        for state in states
        if (value := _clean(left.get(state, 0) - right.get(state, 0))) != 0
    }


def _null_data(x: sp.Expr):
    q_squared = x + 2 + 1 / x
    c = sp.Rational(3, 2) + 3 * q_squared
    return (
        ("(1,1)", 1, c, sp.S.Zero, {G: sp.S.One}),
        (
            "(3,1)",
            3,
            c,
            -x - sp.Rational(1, 2),
            {
                (("G", -3),): x,
                (("G", -1), ("L", -2)): sp.S.One,
            },
        ),
        (
            "(2,2)",
            4,
            c,
            -sp.Rational(3, 8) * q_squared,
            {
                (("L", -4),): (x - 1) ** 2 / (2 * x),
                (("L", -2), ("L", -2)): sp.S.One,
                (("G", -1), ("G", -3)): sp.S.One,
            },
        ),
        (
            "(5,1)",
            5,
            c,
            -3 * x - 1,
            {
                (("G", -5),): x * (6 * x + 1),
                (("G", -3), ("L", -2)): 3 * x,
                (("G", -1), ("L", -4)): 2 * x,
                (("G", -1), ("L", -2), ("L", -2)): sp.S.One,
            },
        ),
    )


def _act_descendant(
    module: ExactNSVermaModule,
    descendant: State,
    vector: dict[State, sp.Expr],
) -> dict[State, sp.Expr]:
    result = dict(vector)
    for mode in reversed(descendant):
        result = _linear_mode_action(module, mode, result)
    return result


def _fusion_polynomial(
    *,
    label: str,
    alpha: int,
    x: sp.Expr,
    first_weight: sp.Expr,
    second_weight: sp.Expr,
) -> sp.Expr:
    r, s = (int(value) for value in label.strip("()").split(","))
    b = sp.sqrt(x)
    q_squared = x + 2 + 1 / x
    lambda_i = sp.sqrt(q_squared - 8 * first_weight)
    lambda_j = sp.sqrt(q_squared - 8 * second_weight)
    result = sp.S.One
    congruence = 2 if alpha == 0 else 0
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - r - s) % 4 != congruence:
                continue
            shift = p * b + q / b
            result *= (lambda_i - lambda_j + shift) / (2 * sp.sqrt(2))
            result *= (lambda_i + lambda_j + shift) / (2 * sp.sqrt(2))
    return sp.simplify(sp.expand_power_base(result, force=True))


class NSPBWAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c = sp.Rational(37, 4)
        self.h = sp.Rational(7, 10)
        self.module = ExactNSVermaModule(c=self.c, weight=self.h)

    def assertExactZero(self, value: sp.Expr, message: object = None) -> None:
        self.assertFalse(value.atoms(sp.Float), f"floating atom in exact path: {value}")
        self.assertEqual(_clean(value), 0, message)

    def test_generalized_binomial_is_exact_for_integer_inputs(self) -> None:
        for value in range(-3, 7):
            for order in range(6):
                observed = generalized_binomial(value, order)
                self.assertFalse(observed.atoms(sp.Float), (value, order, observed))
                self.assertEqual(observed, sp.binomial(value, order))

    def test_theta_orientation_is_literal_human_note_quadratic_form(self) -> None:
        self.assertEqual(THETA_ORIENTATION.edge_linear_bits, (0, 0, 0))
        for bits in product((0, 1), repeat=3):
            expected = (-1) ** (
                bits[0] * bits[1]
                + bits[0] * bits[2]
                + bits[1] * bits[2]
            )
            self.assertEqual(THETA_ORIENTATION.sign(bits), expected, bits)
            self.assertEqual(exact_theta_orientation_sign(bits), expected, bits)
            self.assertEqual(numeric_theta_orientation_sign(bits), expected, bits)

    def test_published_gram_matrices_through_level_two(self) -> None:
        c, h = self.c, self.h
        expected = {
            0: sp.Matrix([[1]]),
            1: sp.Matrix([[2 * h]]),
            2: sp.Matrix([[2 * h]]),
            3: sp.Matrix(
                [[2 * h + 2 * c / 3, 4 * h], [4 * h, 2 * h * (2 * h + 1)]]
            ),
            # Oracle order: L_-2, L_-1^2, G_-1/2 G_-3/2.
            4: sp.Matrix(
                [
                    [4 * h + c / 2, 6 * h, 3 * h + c],
                    [6 * h, 4 * h * (2 * h + 1), 8 * h],
                    [3 * h + c, 8 * h, 4 * h * (h + c / 3) + 2 * (c - h)],
                ]
            ),
        }
        for level, matrix in expected.items():
            for entry in self.module.gram_matrix(level) - matrix:
                self.assertExactZero(entry, level)

    def test_dynamic_exact_basis_matches_independent_numeric_generator(self) -> None:
        """Guard every level used by Ward reductions, including level 7/2."""

        numeric_module = NSVermaModule(c=complex(self.c), weight=complex(self.h))
        for twice_level in range(10):
            self.assertEqual(
                exact_pbw_basis(twice_level),
                numeric_module.basis(twice_level),
                twice_level,
            )

    def test_mode_actions_realize_the_ns_superalgebra(self) -> None:
        pairs = (
            (("L", 2), ("L", -2)),
            (("L", 4), ("L", -4)),
            (("L", 2), ("G", -3)),
            (("L", -2), ("G", 1)),
            (("G", 1), ("G", -1)),
            (("G", 3), ("G", -3)),
            (("G", 1), ("G", -3)),
            (("G", -1), ("G", -3)),
        )
        checked = 0
        for level in range(5):
            for state in PBW_BASES[level]:
                source = {state: sp.S.One}
                for left, right in pairs:
                    intermediate_levels = (level - right[1], level - left[1])
                    final_level = level - left[1] - right[1]
                    if any(value > 6 for value in (*intermediate_levels, final_level)):
                        continue
                    left_right = _linear_mode_action(
                        self.module, left, _linear_mode_action(self.module, right, source)
                    )
                    right_left = _linear_mode_action(
                        self.module, right, _linear_mode_action(self.module, left, source)
                    )
                    graded_sign = -1 if left[0] == right[0] == "G" else 1
                    lhs = {
                        key: _clean(
                            left_right.get(key, 0) - graded_sign * right_left.get(key, 0)
                        )
                        for key in set(left_right) | set(right_left)
                    }
                    lhs = {key: value for key, value in lhs.items() if value != 0}
                    rhs: dict[State, sp.Expr] = {}
                    for coefficient, replacement in self.module.super_bracket(left, right):
                        contribution = (
                            source
                            if replacement is None
                            else _linear_mode_action(self.module, replacement, source)
                        )
                        for key, value in contribution.items():
                            rhs[key] = _clean(rhs.get(key, 0) + coefficient * value)
                    rhs = {key: value for key, value in rhs.items() if value != 0}
                    self.assertEqual(_vector_difference(lhs, rhs), {}, (state, left, right))
                    checked += 1
        self.assertGreaterEqual(checked, 40)


class NSPBWSewingCoefficientTests(unittest.TestCase):
    def test_closed_form_coefficients_on_all_three_edges_through_level_one(self) -> None:
        """Lock the analytic PBW block, not a numerical q-evaluation."""

        c, h0, h1, hinfinity = sp.symbols("c h0 h1 h_infinity")
        oracle = ExactDirectThetaOracle(
            c=c, weights=(h0, h1, hinfinity)
        )
        expected = {
            (0, 0, 0): sp.S.One,
            (1, 0, 0): 1 / (2 * h0),
            (0, 1, 0): 1 / (2 * h1),
            (0, 0, 1): 1 / (2 * hinfinity),
            (2, 0, 0): (h0 + h1 - hinfinity) ** 2 / (2 * h0),
            (0, 2, 0): (h0 - h1 - hinfinity) ** 2 / (2 * h1),
            (0, 0, 2): (h0 - h1 - hinfinity) ** 2 / (2 * hinfinity),
            (1, 1, 0): -(h0 + h1 - hinfinity) ** 2 / (4 * h0 * h1),
            (1, 0, 1): -(h0 - h1 + hinfinity) ** 2
            / (4 * h0 * hinfinity),
            (0, 1, 1): -(h0 - h1 - hinfinity) ** 2
            / (4 * h1 * hinfinity),
        }
        for levels, target in expected.items():
            observed = oracle.coefficient(levels)
            self.assertFalse(observed.atoms(sp.Float), levels)
            self.assertEqual(_clean(observed - target), 0, levels)


class NSWardIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c = sp.Rational(37, 4)
        cls.weights = (
            sp.Rational(7, 10),
            sp.Rational(9, 10),
            sp.Rational(6, 5),
        )
        cls.form = ExactNSDescendantThreeForm(c=cls.c, weights=cls.weights)
        cls.states = tuple(
            state for level in range(4) for state in PBW_BASES[level]
        )

    def assertExactZero(self, value: sp.Expr, message: object = None) -> None:
        self.assertFalse(value.atoms(sp.Float), f"floating atom in exact path: {value}")
        self.assertEqual(_clean(value), 0, message)

    @staticmethod
    def _global_state(n: int, epsilon: int) -> State:
        return (G if epsilon else ()) + (("L", -2),) * n

    def test_all_global_descendants_against_superspace_invariant(self) -> None:
        checked = 0
        for bits in product((0, 1), repeat=3):
            for occupations in product(range(3), repeat=3):
                states = tuple(
                    self._global_state(n, epsilon)
                    for n, epsilon in zip(occupations, bits)
                )
                expected = superspace_three_point(
                    n1=occupations[0],
                    n2=occupations[1],
                    n3=occupations[2],
                    epsilon1=bits[0],
                    epsilon2=bits[1],
                    epsilon3=bits[2],
                    d1=self.weights[0],
                    d2=self.weights[1],
                    d3=self.weights[2],
                )
                self.assertExactZero(self.form.value(*states) - expected, (bits, occupations))
                checked += 1
        self.assertEqual(checked, 216)

    def test_exact_and_numeric_ward_engines_on_level_seven_path(self) -> None:
        """Exercise the temporary twice-level-seven PBW space explicitly.

        The middle G_-1/2 contour reduction of a twice-level-six ket reaches
        twice-level seven before returning to the requested matrix element.
        This was the path hidden by the former fixed exact-basis cutoff.
        """

        critical_states = (
            ((), G, (("L", -2), ("L", -4))),
            ((), G, (("L", -2), ("L", -2), ("L", -2))),
            ((), G, (("G", -1), ("G", -5))),
            ((), G, (("G", -1), ("G", -3), ("L", -2))),
        )
        zero_form = ExactNSDescendantThreeForm(
            c=self.c,
            weights=self.weights,
            primary_parities=(0, 0, 0),
        )
        checked = 0
        for primary_parities in product((0, 1), repeat=3):
            exact_form = ExactNSDescendantThreeForm(
                c=self.c,
                weights=self.weights,
                primary_parities=primary_parities,
            )
            numeric_form = NSDescendantThreeForm(
                c=complex(self.c),
                bra_weight=complex(self.weights[0]),
                middle_weight=complex(self.weights[1]),
                ket_weight=complex(self.weights[2]),
                primary_parities=primary_parities,
            )
            for states in critical_states:
                exact_value = exact_form.value(*states)
                numeric_value = numeric_form.value(*states)
                self.assertLess(
                    abs(complex(sp.N(exact_value, 17)) - numeric_value),
                    1.0e-10,
                    (primary_parities, states, exact_value, numeric_value),
                )
                phase = primary_parity_ward_sign(
                    tuple(state_parity(state) for state in states),
                    primary_parities,
                )
                self.assertExactZero(
                    exact_value - phase * zero_form.value(*states),
                    (primary_parities, states),
                )
                checked += 1
        self.assertEqual(checked, 32)

    def test_alternative_bra_contour_reductions(self) -> None:
        checked_l = 0
        checked_g = 0
        for tail, middle, ket in product(((),), self.states, self.states):
            if not middle:
                continue
            if sum(map(state_twice_level, (tail, middle, ket))) > 4:
                continue
            for n in (1, 2):
                target = ((("L", -2 * n),) + tail, middle, ket)
                rhs = _contour_action(
                    self.form, slot=2, mode=("L", 2 * n), states=(tail, middle, ket)
                )
                for m in range(-1, n + 1):
                    rhs += sp.binomial(n + 1, m + 1) * _contour_action(
                        self.form,
                        slot=1,
                        mode=("L", 2 * m),
                        states=(tail, middle, ket),
                    )
                self.assertExactZero(_contour_value(self.form, target) - rhs, target)
                checked_l += 1

            for twice_k in (1, 3, 5):
                k = sp.Rational(twice_k, 2)
                target = ((("G", -twice_k),) + tail, middle, ket)
                # The parity is that of the full first-slot state G_-k tail.
                sign = (-1) ** (state_parity(tail) + state_parity(ket) + 1)
                rhs = sign * _contour_action(
                    self.form, slot=2, mode=("G", twice_k), states=(tail, middle, ket)
                )
                for m in range(-1, int(k - sp.Rational(1, 2)) + 1):
                    rhs += sp.binomial(k + sp.Rational(1, 2), m + 1) * _contour_action(
                        self.form,
                        slot=1,
                        mode=("G", 2 * m + 1),
                        states=(tail, middle, ket),
                    )
                self.assertExactZero(_contour_value(self.form, target) - rhs, target)
                checked_g += 1
        self.assertEqual((checked_l, checked_g), (24, 36))

    def test_middle_slot_ward_identities_and_translation(self) -> None:
        counts = {"translation": 0, "L+": 0, "G+": 0, "L-": 0, "G-": 0}
        for bra, middle, ket in product(self.states, repeat=3):
            if sum(map(state_twice_level, (bra, middle, ket))) > 5:
                continue
            if not any(kind == "G" for kind, _ in middle):
                target = (bra, (("L", -2),) + middle, ket)
                exponent = (
                    self.weights[0]
                    + sp.Rational(state_twice_level(bra), 2)
                    - self.weights[1]
                    - sp.Rational(state_twice_level(middle), 2)
                    - self.weights[2]
                    - sp.Rational(state_twice_level(ket), 2)
                )
                self.assertExactZero(
                    _contour_value(self.form, target)
                    - exponent * _contour_value(self.form, (bra, middle, ket)),
                    target,
                )
                counts["translation"] += 1

            for n in (0, 1, 2):
                if state_twice_level(bra) + 2 * n > 6:
                    continue
                lhs = _contour_action(
                    self.form, slot=1, mode=("L", 2 * n), states=(bra, middle, ket)
                )
                rhs = sp.S.Zero
                for m in range(n + 2):
                    coefficient = sp.binomial(n + 1, m) * (-1) ** m
                    rhs += coefficient * (
                        _contour_action(
                            self.form,
                            slot=0,
                            mode=("L", 2 * (m - n)),
                            states=(bra, middle, ket),
                        )
                        - _contour_action(
                            self.form,
                            slot=2,
                            mode=("L", 2 * (n - m)),
                            states=(bra, middle, ket),
                        )
                    )
                self.assertExactZero(lhs - rhs, ("L+", n, bra, middle, ket))
                counts["L+"] += 1

            for twice_k in (-1, 1, 3):
                k = sp.Rational(twice_k, 2)
                lhs = _contour_action(
                    self.form, slot=1, mode=("G", twice_k), states=(bra, middle, ket)
                )
                rhs = sp.S.Zero
                endpoint_sign = (-1) ** (state_parity(bra) + state_parity(ket))
                for m in range(int(k + sp.Rational(1, 2)) + 1):
                    coefficient = sp.binomial(k + sp.Rational(1, 2), m) * (-1) ** m
                    rhs += coefficient * (
                        _contour_action(
                            self.form,
                            slot=0,
                            mode=("G", 2 * m - twice_k),
                            states=(bra, middle, ket),
                        )
                        + endpoint_sign
                        * _contour_action(
                            self.form,
                            slot=2,
                            mode=("G", twice_k - 2 * m),
                            states=(bra, middle, ket),
                        )
                    )
                self.assertExactZero(lhs - rhs, ("G+", k, bra, middle, ket))
                counts["G+"] += 1

        # Negative middle modes use empty tails so every target is a PBW word.
        for bra, ket in product(self.states, repeat=2):
            if state_twice_level(bra) + state_twice_level(ket) > 5:
                continue
            for n in (2, 3):
                target = (bra, (("L", -2 * n),), ket)
                rhs = sp.S.Zero
                for m in range(4):
                    coefficient = sp.binomial(n - 2 + m, n - 2)
                    rhs += coefficient * _contour_action(
                        self.form,
                        slot=0,
                        mode=("L", 2 * (n + m)),
                        states=(bra, (), ket),
                    )
                    rhs += (-1) ** n * coefficient * _contour_action(
                        self.form,
                        slot=2,
                        mode=("L", 2 * (m - 1)),
                        states=(bra, (), ket),
                    )
                self.assertExactZero(_contour_value(self.form, target) - rhs, target)
                counts["L-"] += 1

            for twice_k in (3, 5):
                k = sp.Rational(twice_k, 2)
                target = (bra, (("G", -twice_k),), ket)
                rhs = sp.S.Zero
                endpoint_sign = -(-1) ** (
                    state_parity(bra)
                    + state_parity(ket)
                    + int(k + sp.Rational(1, 2))
                )
                for m in range(4):
                    coefficient = generalized_binomial(
                        k - sp.Rational(3, 2) + m, m
                    )
                    rhs += coefficient * _contour_action(
                        self.form,
                        slot=0,
                        mode=("G", twice_k + 2 * m),
                        states=(bra, (), ket),
                    )
                    rhs += endpoint_sign * coefficient * _contour_action(
                        self.form,
                        slot=2,
                        mode=("G", 2 * m - 1),
                        states=(bra, (), ket),
                    )
                self.assertExactZero(_contour_value(self.form, target) - rhs, target)
                counts["G-"] += 1

        self.assertEqual(
            counts,
            {"translation": 33, "L+": 174, "G+": 186, "L-": 42, "G-": 42},
        )

    def test_generic_primary_parities_obey_the_graded_g_ward_identity(self) -> None:
        """Check odd contour transport after restoring component ordering."""

        endpoint_states = ((), G, (("G", -3),))
        checked = 0
        for primaries in product((0, 1), repeat=3):
            form = ExactNSDescendantThreeForm(
                c=self.c,
                weights=self.weights,
                primary_parities=primaries,
            )
            for bra, ket in product(endpoint_states, repeat=2):
                if state_twice_level(bra) + state_twice_level(ket) > 3:
                    continue
                target = (bra, (("G", -3),), ket)
                endpoint_sign = -(-1) ** (
                    state_parity(bra) + state_parity(ket) + 2
                )
                rhs = sp.S.Zero
                for m in range(4):
                    coefficient = generalized_binomial(m, m)
                    rhs += coefficient * _contour_action(
                        form,
                        slot=0,
                        mode=("G", 3 + 2 * m),
                        states=(bra, (), ket),
                    )
                    rhs += endpoint_sign * coefficient * _contour_action(
                        form,
                        slot=2,
                        mode=("G", 2 * m - 1),
                        states=(bra, (), ket),
                    )
                self.assertExactZero(
                    _contour_value(form, target) - rhs,
                    (primaries, bra, ket),
                )
                checked += 1
        self.assertEqual(checked, 48)

    def test_exact_and_numeric_pbw_engines_agree(self) -> None:
        numeric = NSDescendantThreeForm(
            c=float(self.c),
            bra_weight=float(self.weights[0]),
            middle_weight=float(self.weights[1]),
            ket_weight=float(self.weights[2]),
        )
        checked = 0
        largest_error = 0.0
        for levels in product(range(7), repeat=3):
            if sum(levels) > 6:
                continue
            for states in product(*(PBW_BASES[level] for level in levels)):
                exact = complex(sp.N(self.form.value(*states), 17))
                observed = numeric.value(*states)
                error = abs(exact - observed)
                largest_error = max(largest_error, error)
                self.assertLessEqual(error, 2.0e-12 * max(1.0, abs(exact)), states)
                checked += 1
        self.assertEqual(checked, 192)
        self.assertLess(largest_error, 1.0e-13)


class NSNullVectorTests(unittest.TestCase):
    def assertExactZero(self, value: sp.Expr, message: object = None) -> None:
        self.assertFalse(value.atoms(sp.Float), f"floating atom in exact path: {value}")
        self.assertEqual(_clean(value), 0, message)

    def test_low_nulls_are_highest_weight_and_have_the_correct_slope(self) -> None:
        x = sp.Rational(2, 3)
        b = sp.sqrt(x)
        for label, twice_level, c, null_weight, vector in _null_data(x):
            module = ExactNSVermaModule(c=c, weight=null_weight)
            basis = PBW_BASES[twice_level]
            coordinates = sp.Matrix([vector.get(state, 0) for state in basis])
            self.assertExactZero(
                (coordinates.T * module.gram_matrix(twice_level) * coordinates)[0],
                label,
            )
            for twice_mode in range(1, twice_level + 1, 2):
                self.assertEqual(
                    _linear_mode_action(module, ("G", twice_mode), vector), {},
                    (label, "G", twice_mode),
                )
            for twice_mode in range(2, twice_level + 1, 2):
                self.assertEqual(
                    _linear_mode_action(module, ("L", twice_mode), vector), {},
                    (label, "L", twice_mode),
                )

            if label == "(1,1)":
                continue
            h = sp.Symbol("h")
            generic_module = ExactNSVermaModule(c=c, weight=h)
            norm = (
                coordinates.T
                * generic_module.gram_matrix(twice_level)
                * coordinates
            )[0]
            slope = sp.diff(norm, h).subs(h, null_weight)
            r, s = (int(value) for value in label.strip("()").split(","))
            expected_inverse_slope = sp.Rational(1, 2)
            for p in range(1 - r, r + 1):
                for q in range(1 - s, s + 1):
                    if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                        continue
                    expected_inverse_slope *= sp.sqrt(2) / (p * b + q / b)
            self.assertExactZero(1 / slope - expected_inverse_slope, label)

    def test_null_descendant_factorization_has_no_extra_slot_sign(self) -> None:
        x = sp.Rational(2, 3)
        h2 = sp.Rational(7, 10)
        h3 = sp.Rational(11, 13)
        cases = (
            ((), (), ()),
            (G, G, ()),
            ((), G, G),
            ((), G, ()),
            (G, (), ()),
            ((), (), G),
        )
        checked = 0
        for label, twice_level, c, null_weight, vector in _null_data(x):
            generic_h = sp.Symbol(f"h_generic_{twice_level}")
            parent = ExactNSDescendantThreeForm(
                c=c, weights=(generic_h, h2, h3)
            )
            shifted = ExactNSDescendantThreeForm(
                c=c,
                weights=(null_weight + sp.Rational(twice_level, 2), h2, h3),
            )
            action_module = ExactNSVermaModule(c=c, weight=generic_h)
            for first, middle, third in cases:
                descended = _act_descendant(action_module, first, vector)
                lhs = sp.S.Zero
                for state, coefficient in descended.items():
                    lhs += coefficient * parent.value(state, middle, third)
                lhs = _clean(lhs.subs(generic_h, null_weight))
                shifted_value = shifted.value(first, middle, third)
                if shifted_value == 0:
                    continue
                sector = (
                    state_parity(first)
                    + twice_level
                    + state_parity(middle)
                    + state_parity(third)
                ) % 2
                expected = _fusion_polynomial(
                    label=label,
                    alpha=sector,
                    x=x,
                    first_weight=h3,
                    second_weight=h2,
                )
                self.assertExactZero(lhs / shifted_value - expected, (label, first, middle, third))
                checked += 1
        self.assertEqual(checked, 24)


if __name__ == "__main__":
    unittest.main()
