#!/usr/bin/env python3
"""Check NS null factorization against the global osp(1|2) trinion.

The left-hand side is evaluated with the exact full NS PBW/Ward engine.  The
right-hand side is evaluated independently with ``exact_osp_three_point``.
Both APIs use the fixed-parity convention printed in
``Human Notes/SCblock.tex``:

    rho_1(100)=rho_1(010)=1,
    rho_1(001)=-1,
    rho_1(111)=-(h_1+h_2+h_3-1/2).

Intrinsic highest-weight parities are passed into both Ward tensors.  If
``a_absolute`` denotes the absolute parity of a trilinear form, the note's
relative fusion-polynomial label is

    a = a_absolute xor p_1 xor p_2 xor p_3.

The null vector has relative parity ``r*s mod 2``.  When it is regarded as the
highest-weight vector of the shifted module, its intrinsic primary parity is
therefore shifted by the same amount; the absolute parity of the trilinear
form is unchanged.  In the current fixed-parity convention this gives the
three local transport factors

    (-1)^(rs*(p_1+A)),  1,  (-1)^(rs*(1+p_2))

for a null in the first, second, and third slots respectively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import sympy as sp

from ns_genus2_symbolic_low_order import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    State,
    exact_osp_three_point,
)
from ns_human_convention import (
    ns_null_factorization_sign,
    primary_parity_ward_sign,
    relative_label_from_absolute,
)


G_HALF: State = (("G", -1),)
L_ONE = ("L", -2)


def _clean(value: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.together(value))


def _null_data(x: sp.Expr):
    """Return low NS null vectors in the normalization of the human note."""

    q_squared = x + 2 + 1 / x
    c = sp.Rational(3, 2) + 3 * q_squared
    return (
        (1, 1, 1, c, sp.S.Zero, {G_HALF: sp.S.One}),
        (
            3,
            1,
            3,
            c,
            -x - sp.Rational(1, 2),
            {
                (("G", -3),): x,
                (("G", -1), ("L", -2)): sp.S.One,
            },
        ),
        (
            2,
            2,
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
            5,
            1,
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


def _mode_action_on_vector(
    module: ExactNSVermaModule,
    mode: tuple[str, int],
    vector: dict[State, sp.Expr],
) -> dict[State, sp.Expr]:
    result: dict[State, sp.Expr] = {}
    for state, coefficient in vector.items():
        for acted, action_coefficient in module.mode_action(mode, state):
            result[acted] = _clean(
                result.get(acted, sp.S.Zero)
                + coefficient * action_coefficient
            )
    return {state: value for state, value in result.items() if value != 0}


def _act_descendant(
    module: ExactNSVermaModule,
    descendant: State,
    vector: dict[State, sp.Expr],
) -> dict[State, sp.Expr]:
    result = dict(vector)
    for mode in reversed(descendant):
        result = _mode_action_on_vector(module, mode, result)
    return result


def _global_state(occupation: int, epsilon: int) -> State:
    if occupation < 0 or epsilon not in (0, 1):
        raise ValueError("invalid global osp(1|2) state")
    return (G_HALF if epsilon else ()) + (L_ONE,) * occupation


def _fusion_polynomial(
    *,
    r: int,
    s: int,
    a: int,
    x: sp.Expr,
    first_weight: sp.Expr,
    second_weight: sp.Expr,
) -> sp.Expr:
    """Return the human-note NS fusion polynomial ``P_rs^a``."""

    if a not in (0, 1):
        raise ValueError("the relative tensor label must be zero or one")
    b = sp.sqrt(x)
    q_squared = x + 2 + 1 / x
    lambda_i = sp.sqrt(q_squared - 8 * first_weight)
    lambda_j = sp.sqrt(q_squared - 8 * second_weight)
    congruence = 2 if a == 0 else 0
    result = sp.S.One
    for lattice_p in range(1 - r, r, 2):
        for lattice_q in range(1 - s, s, 2):
            if (lattice_p + lattice_q - r - s) % 4 != congruence:
                continue
            shift = lattice_p * b + lattice_q / b
            result *= (lambda_i - lambda_j + shift) / (2 * sp.sqrt(2))
            result *= (lambda_i + lambda_j + shift) / (2 * sp.sqrt(2))
    return sp.simplify(sp.expand_power_base(result, force=True))


def _fusion_weights(weights: tuple[sp.Expr, sp.Expr, sp.Expr], slot: int):
    if slot == 0:
        return weights[2], weights[1]
    if slot == 1:
        return weights[2], weights[0]
    if slot == 2:
        return weights[0], weights[1]
    raise ValueError("a trilinear form has slots 0, 1, and 2")


human_note_endpoint_sign = ns_null_factorization_sign


def relative_structure_label(
    absolute_label: int, intrinsic_primary_parities: tuple[int, int, int]
) -> int:
    """Compatibility alias for the shared absolute-to-relative conversion."""

    return relative_label_from_absolute(
        absolute_label, intrinsic_primary_parities
    )


@dataclass(frozen=True)
class FusionOSPCheckSummary:
    null_labels: tuple[str, ...]
    maximum_total_occupation: int
    exact_factorization_count: int
    intrinsic_parity_label_count: int
    primary_parity_covariance_count: int
    slots_checked: tuple[str, ...]


def run_checks(maximum_total_occupation: int = 1) -> FusionOSPCheckSummary:
    """Run exact PBW/null versus global-osp factorization checks."""

    if maximum_total_occupation < 0:
        raise ValueError("maximum_total_occupation must be non-negative")

    x = sp.Rational(2, 3)
    spectator_weights = (
        sp.Rational(7, 10),
        sp.Rational(11, 13),
        sp.Rational(17, 19),
    )
    occupations = tuple(
        values
        for values in product(
            range(maximum_total_occupation + 1), repeat=3
        )
        if sum(values) <= maximum_total_occupation
    )
    exact_count = 0
    parity_count = 0
    covariance_count = 0
    labels: list[str] = []

    for r, s, twice_level, c, null_weight, null_vector in _null_data(x):
        labels.append(f"({r},{s})")
        null_parity = twice_level % 2
        for slot in range(3):
            generic_weight = sp.Symbol(f"h_null_{r}_{s}_{slot}")
            weights = list(spectator_weights)
            weights[slot] = generic_weight
            primary_assignments = tuple(product((0, 1), repeat=3))
            parents = {
                primary_parities: ExactNSDescendantThreeForm(
                    c=c,
                    weights=tuple(weights),
                    primary_parities=primary_parities,
                )
                for primary_parities in primary_assignments
            }
            module = ExactNSVermaModule(c=c, weight=generic_weight)
            pole_weights = list(spectator_weights)
            pole_weights[slot] = null_weight
            shifted_weights = list(pole_weights)
            shifted_weights[slot] += sp.Rational(twice_level, 2)
            first_weight, second_weight = _fusion_weights(
                tuple(pole_weights), slot
            )

            for occupation_values in occupations:
                for epsilons in product((0, 1), repeat=3):
                    states = tuple(
                        _global_state(n, epsilon)
                        for n, epsilon in zip(occupation_values, epsilons)
                    )
                    descended_null = _act_descendant(
                        module, states[slot], null_vector
                    )
                    parent_a = (sum(epsilons) + null_parity) % 2
                    polynomial = _fusion_polynomial(
                        r=r,
                        s=s,
                        a=parent_a,
                        x=x,
                        first_weight=first_weight,
                        second_weight=second_weight,
                    )
                    parent_descendant_parities = list(epsilons)
                    parent_descendant_parities[slot] ^= null_parity
                    zero_primary_lhs = sp.S.Zero
                    for state, coefficient in descended_null.items():
                        changed = list(states)
                        changed[slot] = state
                        zero_primary_lhs += coefficient * parents[
                            (0, 0, 0)
                        ].value(*changed)
                    zero_primary_lhs = _clean(
                        zero_primary_lhs.subs(
                            generic_weight, null_weight
                        )
                    )
                    zero_primary_child = exact_osp_three_point(
                        n1=occupation_values[0],
                        n2=occupation_values[1],
                        n3=occupation_values[2],
                        epsilon1=epsilons[0],
                        epsilon2=epsilons[1],
                        epsilon3=epsilons[2],
                        d1=shifted_weights[0],
                        d2=shifted_weights[1],
                        d3=shifted_weights[2],
                        primary_parities=(0, 0, 0),
                    )
                    for primary_parities in primary_assignments:
                        parent = parents[primary_parities]
                        lhs = sp.S.Zero
                        for state, coefficient in descended_null.items():
                            changed = list(states)
                            changed[slot] = state
                            lhs += coefficient * parent.value(*changed)
                        lhs = _clean(
                            lhs.subs(generic_weight, null_weight)
                        )

                        absolute_parent = (
                            parent_a + sum(primary_parities)
                        ) % 2
                        recovered_a = relative_structure_label(
                            absolute_parent, primary_parities
                        )
                        if recovered_a != parent_a:
                            raise AssertionError("relative label conversion failed")

                        shifted_primary_parities = list(primary_parities)
                        shifted_primary_parities[slot] ^= null_parity
                        shifted_primary_parities_tuple = tuple(
                            shifted_primary_parities
                        )
                        global_child = exact_osp_three_point(
                            n1=occupation_values[0],
                            n2=occupation_values[1],
                            n3=occupation_values[2],
                            epsilon1=epsilons[0],
                            epsilon2=epsilons[1],
                            epsilon3=epsilons[2],
                            d1=shifted_weights[0],
                            d2=shifted_weights[1],
                            d3=shifted_weights[2],
                            primary_parities=shifted_primary_parities_tuple,
                        )
                        expected_parent_phase = primary_parity_ward_sign(
                            tuple(parent_descendant_parities),
                            primary_parities,
                        )
                        if _clean(
                            lhs - expected_parent_phase * zero_primary_lhs
                        ) != 0:
                            raise AssertionError(
                                "primary parity changed the PBW null matrix "
                                "element by more than its convention phase"
                            )
                        expected_child_phase = primary_parity_ward_sign(
                            epsilons, shifted_primary_parities_tuple
                        )
                        if _clean(
                            global_child
                            - expected_child_phase * zero_primary_child
                        ) != 0:
                            raise AssertionError(
                                "primary parity changed the shifted OSp "
                                "matrix element by more than its convention phase"
                            )
                        covariance_count += 2
                        endpoint_sign = human_note_endpoint_sign(
                            slot=slot,
                            null_parity=null_parity,
                            descendant_parities=epsilons,
                            primary_parities=primary_parities,
                        )
                        difference = _clean(
                            lhs
                            - endpoint_sign * polynomial * global_child
                        )
                        if difference != 0:
                            raise AssertionError(
                                "graded human-note fusion/global-osp mismatch: "
                                f"null=({r},{s}), slot={slot + 1}, "
                                f"occupations={occupation_values}, "
                                f"epsilons={epsilons}, "
                                f"primary_parities={primary_parities}, "
                                f"difference={difference}"
                            )
                        exact_count += 1

                        child_a = sum(epsilons) % 2
                        absolute_child = (
                            child_a + sum(shifted_primary_parities)
                        ) % 2
                        if absolute_child != absolute_parent:
                            raise AssertionError(
                                "absolute trilinear parity changed across the "
                                "naturally parity-shifted null module"
                            )
                        parity_count += 1

    return FusionOSPCheckSummary(
        null_labels=tuple(dict.fromkeys(labels)),
        maximum_total_occupation=maximum_total_occupation,
        exact_factorization_count=exact_count,
        intrinsic_parity_label_count=parity_count,
        primary_parity_covariance_count=covariance_count,
        slots_checked=("infinity", "one", "zero"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-occupation", type=int, default=1)
    args = parser.parse_args()
    summary = run_checks(args.max_occupation)
    print("PASS: exact NS fusion factorization against global osp(1|2)")
    print(f"  null labels: {', '.join(summary.null_labels)}")
    print(f"  slots: {', '.join(summary.slots_checked)}")
    print(
        "  exact PBW/global-osp identities: "
        f"{summary.exact_factorization_count}"
    )
    print(
        "  intrinsic-parity label identities: "
        f"{summary.intrinsic_parity_label_count}"
    )
    print(
        "  pure parity-rephasing identities: "
        f"{summary.primary_parity_covariance_count}"
    )


if __name__ == "__main__":
    main()
