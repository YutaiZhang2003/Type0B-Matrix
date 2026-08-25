#!/usr/bin/env python3
"""Direct PBW audit of the two-null NS factorization formulas in Appendix A.6.

Both singular vectors are inserted into the exact three-point Ward tensor
before the two independent highest weights are specialized to ``h_{r,s}``.
The resulting degenerate-limit matrix element is compared with the ordered
product of the two fusion
polynomials printed in A.6 and the global osp(1|2) matrix element of the two
shifted primaries.

This is deliberately an independent sign audit: the observed sign is first
inferred by comparing the direct PBW answer with the unsigned factorization.
Only afterward is it compared with the closed sign obtained by successively
using the already-fixed one-null conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import sympy as sp

from check_ns_fusion_global_osp import (
    _act_descendant,
    _clean,
    _fusion_polynomial,
    _fusion_weights,
    _global_state,
    _null_data,
)
from ns_genus2_symbolic_low_order import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    exact_osp_three_point,
)
from ns_human_convention import ns_double_null_factorization_sign


PAIR_NAMES = {
    (0, 1): "(1,2)",
    (0, 2): "(1,3)",
    (1, 2): "(2,3)",
}


def _is_zero(value: sp.Expr) -> bool:
    """Test the exact expressions occurring in this finite audit."""

    cleaned = _clean(value)
    return cleaned == 0 or sp.simplify(cleaned) == 0


def _direct_double_null_value(
    *,
    form: ExactNSDescendantThreeForm,
    states: tuple[tuple[tuple[str, int], ...], ...],
    pair: tuple[int, int],
    descended_nulls: tuple[
        dict[tuple[tuple[str, int], ...], sp.Expr],
        dict[tuple[tuple[str, int], ...], sp.Expr],
    ],
    generic_weights: tuple[sp.Symbol, sp.Symbol],
    null_weight: sp.Expr,
) -> sp.Expr:
    """Insert two descended singular vectors in one exact Ward tensor."""

    result = sp.S.Zero
    first_slot, second_slot = pair
    for first_state, first_coefficient in descended_nulls[0].items():
        for second_state, second_coefficient in descended_nulls[1].items():
            changed = list(states)
            changed[first_slot] = first_state
            changed[second_slot] = second_state
            result += (
                first_coefficient
                * second_coefficient
                * form.value(*changed)
            )
    return _clean(
        result.subs(
            {
                generic_weights[0]: null_weight,
                generic_weights[1]: null_weight,
            },
            simultaneous=True,
        )
    )


def _ordered_unsigned_factorization(
    *,
    r: int,
    s: int,
    twice_level: int,
    x: sp.Expr,
    pair: tuple[int, int],
    pole_weights: tuple[sp.Expr, sp.Expr, sp.Expr],
    occupations: tuple[int, int, int],
    epsilons: tuple[int, int, int],
    primary_parities: tuple[int, int, int],
) -> sp.Expr:
    """Return the unsigned right side in the polynomial order used in A.6."""

    null_parity = twice_level % 2
    relative_label = sum(epsilons) % 2
    shifted_weights = list(pole_weights)

    first_slot, second_slot = pair
    first_weight, second_weight = _fusion_weights(
        tuple(shifted_weights), first_slot
    )
    first_polynomial = _fusion_polynomial(
        r=r,
        s=s,
        a=relative_label,
        x=x,
        first_weight=first_weight,
        second_weight=second_weight,
    )
    shifted_weights[first_slot] += sp.Rational(twice_level, 2)

    first_weight, second_weight = _fusion_weights(
        tuple(shifted_weights), second_slot
    )
    second_polynomial = _fusion_polynomial(
        r=r,
        s=s,
        a=(relative_label + null_parity) % 2,
        x=x,
        first_weight=first_weight,
        second_weight=second_weight,
    )
    shifted_weights[second_slot] += sp.Rational(twice_level, 2)

    shifted_primary_parities = list(primary_parities)
    shifted_primary_parities[first_slot] ^= null_parity
    shifted_primary_parities[second_slot] ^= null_parity
    child = exact_osp_three_point(
        n1=occupations[0],
        n2=occupations[1],
        n3=occupations[2],
        epsilon1=epsilons[0],
        epsilon2=epsilons[1],
        epsilon3=epsilons[2],
        d1=shifted_weights[0],
        d2=shifted_weights[1],
        d3=shifted_weights[2],
        primary_parities=tuple(shifted_primary_parities),
    )
    return _clean(first_polynomial * second_polynomial * child)


def predicted_a6_sign(
    *,
    pair: tuple[int, int],
    null_parity: int,
    descendant_parities: tuple[int, int, int],
    primary_parities: tuple[int, int, int],
) -> int:
    """Closed A.6 sign for the polynomial ordering used in the note.

    The labels are ``descendant_parities=(A,C,E)`` and
    ``primary_parities=(p_1,p_2,p_3)``.  Modulo two, the exponents are

        (1,2): rs (p_1 + A),
        (1,3): rs (1 + p_1 + p_2 + A),
        (2,3): rs p_2.
    """

    return ns_double_null_factorization_sign(
        pair=pair,
        null_parity=null_parity,
        descendant_parities=descendant_parities,
        primary_parities=primary_parities,
    )


@dataclass(frozen=True)
class DoubleNullCheckSummary:
    null_labels: tuple[str, ...]
    exhaustive_null_labels: tuple[str, ...]
    maximum_total_occupation: int
    exact_identity_count: int
    inferred_positive_count: int
    inferred_negative_count: int
    pair_signatures: tuple[str, ...]


def run_checks(
    maximum_total_occupation: int = 0,
    exhaustive_null_labels: tuple[tuple[int, int], ...] = ((1, 1),),
) -> DoubleNullCheckSummary:
    """Run the exact two-null PBW/fusion-polynomial audit."""

    if maximum_total_occupation < 0:
        raise ValueError("maximum_total_occupation must be non-negative")
    exhaustive_null_set = set(exhaustive_null_labels)

    x = sp.Rational(2, 3)
    spectator_weights = (
        sp.Rational(7, 10),
        sp.Rational(11, 13),
        sp.Rational(17, 19),
    )
    occupations_to_check = tuple(
        values
        for values in product(
            range(maximum_total_occupation + 1), repeat=3
        )
        if sum(values) <= maximum_total_occupation
    )
    primary_assignments = tuple(product((0, 1), repeat=3))
    exact_count = 0
    positive_count = 0
    negative_count = 0
    labels: list[str] = []

    for r, s, twice_level, c, null_weight, null_vector in _null_data(x):
        labels.append(f"({r},{s})")
        null_parity = twice_level % 2
        for pair in PAIR_NAMES:
            generic_weights = (
                sp.Symbol(f"h_null_{r}_{s}_{pair[0] + 1}"),
                sp.Symbol(f"h_null_{r}_{s}_{pair[1] + 1}"),
            )
            parent_weights = list(spectator_weights)
            parent_weights[pair[0]] = generic_weights[0]
            parent_weights[pair[1]] = generic_weights[1]
            modules = (
                ExactNSVermaModule(c=c, weight=generic_weights[0]),
                ExactNSVermaModule(c=c, weight=generic_weights[1]),
            )
            forms = {
                primary_parities: ExactNSDescendantThreeForm(
                    c=c,
                    weights=tuple(parent_weights),
                    primary_parities=primary_parities,
                )
                for primary_parities in primary_assignments
            }
            pole_weights = list(spectator_weights)
            pole_weights[pair[0]] = null_weight
            pole_weights[pair[1]] = null_weight

            for occupations in occupations_to_check:
                epsilon_assignments = (
                    tuple(product((0, 1), repeat=3))
                    if (r, s) in exhaustive_null_set
                    else ((0, 0, 0), (1, 0, 0))
                )
                parity_assignments = (
                    primary_assignments
                    if (r, s) in exhaustive_null_set
                    else ((0, 0, 0),)
                )
                for epsilons in epsilon_assignments:
                    states = tuple(
                        _global_state(n, epsilon)
                        for n, epsilon in zip(occupations, epsilons)
                    )
                    descended_nulls = (
                        _act_descendant(
                            modules[0], states[pair[0]], null_vector
                        ),
                        _act_descendant(
                            modules[1], states[pair[1]], null_vector
                        ),
                    )
                    for primary_parities in parity_assignments:
                        lhs = _direct_double_null_value(
                            form=forms[primary_parities],
                            states=states,
                            pair=pair,
                            descended_nulls=descended_nulls,
                            generic_weights=generic_weights,
                            null_weight=null_weight,
                        )
                        unsigned_rhs = _ordered_unsigned_factorization(
                            r=r,
                            s=s,
                            twice_level=twice_level,
                            x=x,
                            pair=pair,
                            pole_weights=tuple(pole_weights),
                            occupations=occupations,
                            epsilons=epsilons,
                            primary_parities=primary_parities,
                        )
                        plus_matches = _is_zero(lhs - unsigned_rhs)
                        minus_matches = _is_zero(lhs + unsigned_rhs)
                        if plus_matches == minus_matches:
                            raise AssertionError(
                                "could not infer a unique two-null sign: "
                                f"null=({r},{s}), pair={PAIR_NAMES[pair]}, "
                                f"occupations={occupations}, "
                                f"epsilons={epsilons}, "
                                f"primary_parities={primary_parities}, "
                                f"lhs={lhs}, unsigned_rhs={unsigned_rhs}"
                            )
                        observed_sign = 1 if plus_matches else -1
                        expected_sign = predicted_a6_sign(
                            pair=pair,
                            null_parity=null_parity,
                            descendant_parities=epsilons,
                            primary_parities=primary_parities,
                        )
                        if observed_sign != expected_sign:
                            raise AssertionError(
                                "direct PBW sign disagrees with the proposed "
                                "A.6 formula: "
                                f"null=({r},{s}), pair={PAIR_NAMES[pair]}, "
                                f"occupations={occupations}, "
                                f"epsilons={epsilons}, "
                                f"primary_parities={primary_parities}, "
                                f"observed={observed_sign}, "
                                f"expected={expected_sign}"
                            )
                        exact_count += 1
                        if observed_sign == 1:
                            positive_count += 1
                        else:
                            negative_count += 1

    return DoubleNullCheckSummary(
        null_labels=tuple(labels),
        exhaustive_null_labels=tuple(
            f"({r},{s})" for r, s in exhaustive_null_labels
        ),
        maximum_total_occupation=maximum_total_occupation,
        exact_identity_count=exact_count,
        inferred_positive_count=positive_count,
        inferred_negative_count=negative_count,
        pair_signatures=(
            "(1,2): (-1)^{rs(p_1+A)}",
            "(1,3): (-1)^{rs(1+p_1+p_2+A)}",
            "(2,3): (-1)^{rs p_2}",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-occupation", type=int, default=0)
    parser.add_argument(
        "--exhaustive-null",
        action="append",
        default=None,
        metavar="R,S",
        help="null label whose full epsilon/primary-parity table is checked",
    )
    args = parser.parse_args()
    exhaustive_labels = ((1, 1),)
    if args.exhaustive_null:
        exhaustive_labels = tuple(
            tuple(int(value) for value in label.split(","))
            for label in args.exhaustive_null
        )
        if any(len(label) != 2 for label in exhaustive_labels):
            parser.error("--exhaustive-null must have the form R,S")
    summary = run_checks(args.max_occupation, exhaustive_labels)
    print("PASS: direct two-null PBW factorization for Appendix A.6")
    print(f"  null labels: {', '.join(summary.null_labels)}")
    print(
        "  exhaustive parity tables: "
        f"{', '.join(summary.exhaustive_null_labels)}"
    )
    print(f"  maximum total global occupation: {summary.maximum_total_occupation}")
    print(f"  exact identities: {summary.exact_identity_count}")
    print(
        "  independently inferred signs: "
        f"{summary.inferred_positive_count} positive, "
        f"{summary.inferred_negative_count} negative"
    )
    for signature in summary.pair_signatures:
        print(f"  {signature}")


if __name__ == "__main__":
    main()
