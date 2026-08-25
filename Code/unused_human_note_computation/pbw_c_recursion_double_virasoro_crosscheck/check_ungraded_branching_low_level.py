#!/usr/bin/env python3
"""Direct low-level audit of the NS branching formula with an ungraded pairing.

The tensor-product pairing and normalized three-point form used here are

    B_0(u x x, v x y) = B_F(u,v) B_SCA(x,y),

    rho_hat_0(u1 x x1, u2 x x2, u3 x x3)
      = (-1)^(|x2||u3|) rho_F(u1,u2,u3) rho_SCA(x1,x2,x3).

Thus the BPZ-leg crossing ``|x1|(|u2|+|u3|)`` is deliberately absent.  The
intrinsic auxiliary-fermion convention psi_r^dagger=-psi_{-r} is retained.

The audit has two parts.

1. A symbolic calculation for n_i in {-1/2,0,1/2}.  The first Virasoro
   primary is expanded explicitly in the auxiliary-fermion x SCA PBW basis.
2. Exact rational calculations for n_i in {0,1/2,1}.  The n=1 primary is
   solved independently from the two Virasoro highest-weight equations in
   the full level-two tensor-product module.
3. A next-primary probe constructs v_{3/2} at physical level 9/2.  It checks
   its norm and identity matrix element, then tests all triples
   (k_1,k_2,k_3) in {0,1,2} x {0,1,2} x {3}.

Both the norm formula and the unnormalized three-point product are checked.
For the unsquared branching coefficient we state the square-root branch

    ||v_n|| = i^(2n) 2^(-n)
              sqrt(ell(2P,4n) ell(Q+2P,4n)),       n >= 0.

With this branch the script separately tests the coefficient with and without
the displayed third-slot factor (-1)^(2 n_3).  The squared coefficient is
independent of that last comparison.
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from ns_genus2_three_way_symbolic_check import paper_branching_candidate_squared
from ns_genus2_symbolic_low_order import PBW_BASES
from ns_human_convention import enlarged_ns_three_form_crossing_sign

from check_first_virasoro_primary import (
    branching_ell_product,
    direct_norm,
    ell,
    fermion_rho,
    primary_components,
    product_norm as first_product_norm,
    sca_rho,
    simplify,
)
from check_second_virasoro_primary import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    TensorVector,
    action_matrix,
    current_fermion_three_point,
    current_sca_three_point,
    first_copy_weight,
    product_norm as second_product_norm,
    product_rho,
    solve_v1,
    state_parity,
    tensor_basis,
    v0,
    vector_norm,
    vhalf,
)


HALF = sp.Rational(1, 2)


def require_zero(name: str, expression: sp.Expr) -> None:
    residual = simplify(expression)
    if residual != 0:
        raise AssertionError(f"{name}: residual={sp.factor(residual)}")


def _integer_partitions(total: int, minimum: int = 1):
    if total == 0:
        yield ()
        return
    for first in range(minimum, total + 1):
        for rest in _integer_partitions(total - first, first):
            yield (first,) + rest


def ensure_complete_ns_pbw_bases(max_twice_level: int) -> None:
    """Extend the low-order Ward engine's complete NS PBW table."""

    for twice_level in range(max_twice_level + 1):
        if twice_level in PBW_BASES:
            continue
        states = []
        odd_modes = tuple(range(1, twice_level + 1, 2))
        for mask in range(1 << len(odd_modes)):
            fermionic = tuple(
                mode for index, mode in enumerate(odd_modes) if mask >> index & 1
            )
            remaining = twice_level - sum(fermionic)
            if remaining < 0 or remaining % 2:
                continue
            for bosonic in _integer_partitions(remaining // 2):
                state = tuple(("G", -mode) for mode in fermionic)
                state += tuple(("L", -2 * mode) for mode in bosonic)
                states.append(state)
        PBW_BASES[twice_level] = tuple(states)


def solve_positive_branch_primary(
    b: sp.Expr, momentum: sp.Expr, label: int
) -> tuple[ExactNSVermaModule, TensorVector]:
    """Construct ``v_{label/2}`` from the two Virasoro HW equations.

    Its tensor-product level is ``label**2/2``.  The independently found
    eigendirection is normalized by the pure auxiliary Fermi-sea component

        2**(-label) ell(Q+2P,2 label)
        chi_{-(2 label-1)/2} ... chi_{-1/2}|P>.

    The canonical auxiliary basis stores creation modes in the opposite
    product order, giving ``(-1)**(label(label-1)/2)``.
    """

    if label < 0:
        raise ValueError("this direct solver expects a nonnegative label")
    if label == 0:
        q = b + 1 / b
        c = sp.Rational(3, 2) + 3 * q**2
        h = q**2 / 8 - momentum**2 / 2
        return ExactNSVermaModule(c=c, weight=h), v0()
    if label == 2:
        return solve_v1(b, momentum)

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    h = q**2 / 8 - momentum**2 / 2
    module = ExactNSVermaModule(c=c, weight=h)
    twice_level = label**2
    ensure_complete_ns_pbw_bases(twice_level)
    source = tensor_basis(module, twice_level)

    equations: list[sp.Matrix] = []
    for copy in (1, 2):
        for mode in range(1, twice_level // 2 + 1):
            target = tensor_basis(module, twice_level - 2 * mode)
            equations.append(
                action_matrix(module, b, copy, mode, source, target)
            )
    l0_first = action_matrix(module, b, 1, 0, source, source)
    eigenvalue = first_copy_weight(
        b, momentum, sp.Rational(label, 2)
    )
    equations.append(l0_first - eigenvalue * sp.eye(len(source)))
    system = equations[0]
    for equation in equations[1:]:
        system = system.col_join(equation)
    kernel = system.nullspace()
    if len(kernel) != 1:
        raise AssertionError(
            f"expected one k={label} eigendirection, got {len(kernel)}"
        )
    raw = kernel[0]

    pure_fermion = (tuple(range(1, label + 1)), ())
    fermi_sea_reversal = (-1) ** (label * (label - 1) // 2)
    target_coefficient = simplify(
        fermi_sea_reversal
        * sp.Pow(2, -label)
        * ell(q + 2 * momentum, 2 * label, b, q)
    )
    raw_coefficient = raw[source.index(pure_fermion)]
    if raw_coefficient == 0:
        raise AssertionError(f"k={label} eigendirection has no Fermi-sea term")
    scale = sp.cancel(target_coefficient / raw_coefficient)
    vector = {
        state: sp.cancel(scale * coefficient)
        for state, coefficient in zip(source, raw)
        if coefficient != 0
    }
    return module, vector


def ungraded_first_rho(
    labels: tuple[int, int, int],
    gammas: tuple[sp.Expr, sp.Expr, sp.Expr],
    weights: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> sp.Expr:
    """Expand the first primary with the note's ungraded product crossing."""

    result = sp.S.Zero
    for components in product(
        *(primary_components(label, gamma) for label, gamma in zip(labels, gammas))
    ):
        fermion_parities = tuple(
            component.auxiliary_parity for component in components
        )
        fermion_value = fermion_rho(fermion_parities)
        if fermion_value == 0:
            continue
        sca_parities = tuple(component.sca_parity for component in components)
        result += (
            sp.prod(component.coefficient for component in components)
            * enlarged_ns_three_form_crossing_sign(
                sca_parities, fermion_parities
            )
            * fermion_value
            * sca_rho(sca_parities, weights)
        )
    return simplify(result)


def positive_product_rho_symbolic(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
) -> sp.Expr:
    """Equation for rho_hat after reflecting every negative branch label."""

    effective_labels = tuple(abs(label) for label in labels)
    effective_momenta = tuple(
        -momentum if label < 0 else momentum
        for label, momentum in zip(labels, momenta)
    )
    total_label = sum(effective_labels)
    parity = total_label % 2
    ell_product = branching_ell_product(labels, momenta, b, q)
    return simplify(
        (-sp.I) ** total_label
        * sp.I**parity
        * sp.Pow(2, -sp.Rational(total_label, 2))
        * ell_product
    )


def common_root_numerator(
    rho: sp.Expr, labels: tuple[int, int, int]
) -> sp.Expr:
    """Remove the fixed powers/phases in the declared norm square roots."""

    total_label = sum(abs(label) for label in labels)
    return simplify(
        rho
        * sp.Pow(2, sp.Rational(total_label, 2))
        * sp.I ** (-total_label)
    )


def symbolic_first_primary_audit() -> tuple[int, int, int, int, int, int]:
    b = sp.symbols("b", nonzero=True)
    momenta = sp.symbols("P_1 P_2 P_3")
    q = b + 1 / b
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)

    norm_passes = 0
    for momentum, weight in zip(momenta, weights):
        for label in (-1, 0, 1):
            gamma = q / 2 + label * momentum if label else sp.S.Zero
            direct = direct_norm(label, weight, gamma)
            expected = first_product_norm(label, momentum, b, q)
            require_zero(f"symbolic norm k={label}", direct - expected)
            norm_passes += 1

    rho_with_slot_passes = 0
    rho_without_slot_passes = 0
    b_without_slot_passes = 0
    b_with_slot_passes = 0
    b_squared_passes = 0
    for labels in product((-1, 0, 1), repeat=3):
        gammas = tuple(
            q / 2 + label * momentum if label else sp.S.Zero
            for label, momentum in zip(labels, momenta)
        )
        direct_rho = ungraded_first_rho(labels, gammas, weights)
        rho_without_slot = positive_product_rho_symbolic(labels, momenta, b, q)
        rho_with_slot = simplify(
            (-1) ** abs(labels[2]) * rho_without_slot
        )
        if simplify(direct_rho - rho_with_slot) == 0:
            rho_with_slot_passes += 1
        if simplify(direct_rho - rho_without_slot) == 0:
            rho_without_slot_passes += 1

        ell_product = branching_ell_product(labels, momenta, b, q)
        parity = sum(abs(label) for label in labels) % 2
        direct_numerator = common_root_numerator(direct_rho, labels)
        no_slot_numerator = simplify((-sp.I) ** parity * ell_product)
        with_slot_numerator = simplify(
            (-1) ** abs(labels[2]) * no_slot_numerator
        )
        if simplify(direct_numerator - no_slot_numerator) == 0:
            b_without_slot_passes += 1
        if simplify(direct_numerator - with_slot_numerator) == 0:
            b_with_slot_passes += 1

        direct_norm_product = sp.prod(
            direct_norm(label, weight, gamma)
            for label, weight, gamma in zip(labels, weights, gammas)
        )
        expected_norm_product = sp.prod(
            first_product_norm(label, momentum, b, q)
            for label, momentum in zip(labels, momenta)
        )
        direct_b_squared = simplify(direct_rho**2 / direct_norm_product)
        expected_b_squared = simplify(rho_with_slot**2 / expected_norm_product)
        if simplify(direct_b_squared - expected_b_squared) == 0:
            b_squared_passes += 1

    return (
        norm_passes,
        rho_with_slot_passes,
        rho_without_slot_passes,
        b_without_slot_passes,
        b_with_slot_passes,
        b_squared_passes,
    )


def ungraded_tensor_three_point(
    form: ExactNSDescendantThreeForm,
    vectors: tuple[TensorVector, TensorVector, TensorVector],
    *,
    leg_twists: tuple[int, int, int] = (0, 0, 0),
) -> sp.Expr:
    """Exact algebraic product three-form, with optional diagnostics.

    ``form.primary_parities`` supplies the intrinsic-primary Koszul sign.
    The optional leg twists are retained only for the historical convention
    scan printed by this audit.
    """

    result = sp.S.Zero
    for components in product(*(tuple(vector.items()) for vector in vectors)):
        tensor_states = tuple(component[0] for component in components)
        coefficient = sp.prod(component[1] for component in components)
        fermions = tuple(state[0] for state in tensor_states)
        fermion_value = current_fermion_three_point(fermions)
        if fermion_value == 0:
            continue
        sca_states = tuple(state[1] for state in tensor_states)
        sca_parities = tuple(state_parity(state) for state in sca_states)
        fermion_parities = tuple(len(state) % 2 for state in fermions)
        crossing = enlarged_ns_three_form_crossing_sign(
            sca_parities,
            fermion_parities,
            form.primary_parities,
        )
        diagnostic_exponent = sum(
            leg_twists[index]
            * sca_parities[index]
            * fermion_parities[index]
            for index in range(3)
        )
        crossing *= -1 if diagnostic_exponent % 2 else 1
        result += (
            coefficient
            * crossing
            * fermion_value
            * current_sca_three_point(form, sca_states)
        )
    return simplify(result)


def exact_level_two_sample(
    b: sp.Rational,
    momenta: tuple[sp.Rational, sp.Rational, sp.Rational],
) -> tuple[int, int, int, int, int, int, int]:
    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)

    modules = []
    vectors_by_slot = []
    direct_norms_by_slot = []
    norm_passes = 0
    for momentum in momenta:
        module, level_two_vector = solve_v1(b, momentum)
        vectors = {0: v0(), 1: vhalf(q, momentum), 2: level_two_vector}
        modules.append(module)
        vectors_by_slot.append(vectors)
        direct_norms = {}
        for label, vector in vectors.items():
            direct = vector_norm(module, vector)
            direct_norms[label] = direct
            expected = second_product_norm(label, momentum, b, q)
            require_zero(f"level-two norm k={label}", direct - expected)
            norm_passes += 1
        direct_norms_by_slot.append(direct_norms)

    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    reflected_form = ExactNSDescendantThreeForm(
        c=c, weights=(weights[2], weights[1], weights[0])
    )
    rho_with_slot_passes = 0
    rho_without_slot_passes = 0
    b_without_slot_passes = 0
    b_with_slot_passes = 0
    b_squared_passes = 0
    endpoint_consistency_passes = 0
    mismatches = []
    endpoint_mismatches = []
    twist_counts = {mask: 0 for mask in product((0, 1), repeat=3)}
    for labels in product((0, 1, 2), repeat=3):
        vectors = tuple(
            vectors_by_slot[slot][label] for slot, label in enumerate(labels)
        )
        direct_rho = ungraded_tensor_three_point(form, vectors)
        reflection_sign = (-1) ** ((labels[0] + labels[2]) % 2)
        reflected_rho = simplify(
            reflection_sign
            * ungraded_tensor_three_point(
                reflected_form, (vectors[2], vectors[1], vectors[0])
            )
        )
        endpoint_residual = simplify(direct_rho - reflected_rho)
        if endpoint_residual != 0:
            endpoint_mismatches.append((labels, endpoint_residual))
        else:
            endpoint_consistency_passes += 1

        rho_without_slot = product_rho(
            labels, momenta, b, q, third_slot_sign=False
        )
        rho_with_slot = product_rho(
            labels, momenta, b, q, third_slot_sign=True
        )
        for mask in twist_counts:
            trial_rho = ungraded_tensor_three_point(
                form, vectors, leg_twists=mask
            )
            if simplify(trial_rho - rho_with_slot) == 0:
                twist_counts[mask] += 1
        if simplify(direct_rho - rho_with_slot) == 0:
            rho_with_slot_passes += 1
        else:
            mismatches.append(
                (
                    labels,
                    direct_rho,
                    rho_with_slot,
                    simplify(direct_rho - rho_with_slot),
                )
            )
        if simplify(direct_rho - rho_without_slot) == 0:
            rho_without_slot_passes += 1

        total_label = sum(labels)
        parity = total_label % 2
        direct_numerator = simplify(
            direct_rho
            * sp.Pow(2, sp.Rational(total_label, 2))
            * sp.I ** (-total_label)
        )
        ell_product = simplify(
            rho_without_slot
            / (
                (-sp.I) ** total_label
                * sp.I**parity
                * sp.Pow(2, -sp.Rational(total_label, 2))
            )
        )
        no_slot_numerator = simplify((-sp.I) ** parity * ell_product)
        with_slot_numerator = simplify(
            (-1) ** labels[2] * no_slot_numerator
        )
        if simplify(direct_numerator - no_slot_numerator) == 0:
            b_without_slot_passes += 1
        if simplify(direct_numerator - with_slot_numerator) == 0:
            b_with_slot_passes += 1

        direct_norm_product = sp.prod(
            direct_norms_by_slot[slot][label]
            for slot, label in enumerate(labels)
        )
        formula_norm_product = sp.prod(
            second_product_norm(label, momentum, b, q)
            for label, momentum in zip(labels, momenta)
        )
        direct_b_squared = simplify(direct_rho**2 / direct_norm_product)
        reconstructed_b_squared = simplify(
            rho_with_slot**2 / formula_norm_product
        )
        displayed_b_squared = simplify(
            paper_branching_candidate_squared(
                momenta=momenta,
                labels=labels,
                b=b,
            )
        )
        if simplify(reconstructed_b_squared - displayed_b_squared) != 0:
            raise AssertionError(
                f"branching-formula dictionary mismatch at k={labels}"
            )
        if simplify(direct_b_squared - displayed_b_squared) == 0:
            b_squared_passes += 1

    for (
        mismatch_labels,
        mismatch_direct,
        mismatch_expected,
        mismatch_residual,
    ) in mismatches:
        print(
            "  ungraded rho mismatch: "
            f"k={mismatch_labels}, direct={sp.factor(mismatch_direct)}, "
            f"product={sp.factor(mismatch_expected)}, "
            f"ratio={sp.factor(mismatch_direct / mismatch_expected)}, "
            f"residual={sp.factor(mismatch_residual)}"
        )
    if endpoint_mismatches:
        print(
            "  endpoint-exchange diagnostic failures: "
            + ", ".join(
                f"k={labels} (residual={sp.factor(residual)})"
                for labels, residual in endpoint_mismatches
            )
        )
    exact_twists = [mask for mask, count in twist_counts.items() if count == 27]
    print(
        "  leg-twist scan against the product formula: "
        f"exact masks={exact_twists}; counts={twist_counts}"
    )
    return (
        norm_passes,
        endpoint_consistency_passes,
        rho_with_slot_passes,
        rho_without_slot_passes,
        b_without_slot_passes,
        b_with_slot_passes,
        b_squared_passes,
    )


def exact_level_three_half_sample(
    b: sp.Rational,
    momenta: tuple[sp.Rational, sp.Rational, sp.Rational],
) -> tuple[
    int,
    int,
    int,
    int,
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int, int], ...],
    bool,
    bool,
]:
    """Probe the first branching order beyond the note's old direct check."""

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
    modules = []
    vectors_by_slot = []
    norm_passes = 0
    for momentum in momenta:
        module, level_two = solve_v1(b, momentum)
        module_three_half, level_three_half = solve_positive_branch_primary(
            b, momentum, 3
        )
        require_zero(
            f"level-nine-half norm k=3, P={momentum}",
            vector_norm(module_three_half, level_three_half)
            - second_product_norm(3, momentum, b, q),
        )
        norm_passes += 1
        modules.append(module_three_half)
        vectors_by_slot.append(
            {
                0: v0(),
                1: vhalf(q, momentum),
                2: level_two,
                3: level_three_half,
            }
        )

    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    reflected_form = ExactNSDescendantThreeForm(
        c=c, weights=(weights[2], weights[1], weights[0])
    )
    endpoint_consistency_passes = 0
    with_slot_passes = 0
    without_slot_passes = 0
    endpoint_failures = []
    product_failures = []
    for first_label, second_label in product((0, 1, 2), repeat=2):
        labels = (first_label, second_label, 3)
        vectors = tuple(
            vectors_by_slot[slot][label] for slot, label in enumerate(labels)
        )
        direct = ungraded_tensor_three_point(form, vectors)
        reflection_sign = (-1) ** ((labels[0] + labels[2]) % 2)
        reflected_direct = simplify(
            reflection_sign
            * ungraded_tensor_three_point(
                reflected_form, (vectors[2], vectors[1], vectors[0])
            )
        )
        if simplify(direct - reflected_direct) != 0:
            endpoint_failures.append(labels)
        else:
            endpoint_consistency_passes += 1
        with_slot = product_rho(
            labels, momenta, b, q, third_slot_sign=True
        )
        without_slot = product_rho(
            labels, momenta, b, q, third_slot_sign=False
        )
        if simplify(direct - with_slot) == 0:
            with_slot_passes += 1
        else:
            product_failures.append(labels)
        if simplify(direct - without_slot) == 0:
            without_slot_passes += 1

    # Because the three-point form is the matrix element defined by the same
    # ungraded product BPZ convention, an identity in the middle reproduces
    # the ungraded norm used in the branching denominator.
    momentum = momenta[0]
    weight = weights[0]
    vector = vectors_by_slot[0][3]
    identity_weight = sp.symbols("h_identity_three_half")
    identity_form = ExactNSDescendantThreeForm(
        c=c, weights=(weight, identity_weight, weight)
    )
    direct_identity = simplify(
        ungraded_tensor_three_point(identity_form, (vector, v0(), vector)).subs(
            identity_weight, 0
        )
    )
    with_identity = product_rho(
        (3, 0, 3), (momentum, q / 2, momentum), b, q,
        third_slot_sign=True,
    )
    without_identity = product_rho(
        (3, 0, 3), (momentum, q / 2, momentum), b, q,
        third_slot_sign=False,
    )
    return (
        norm_passes,
        endpoint_consistency_passes,
        with_slot_passes,
        without_slot_passes,
        tuple(endpoint_failures),
        tuple(product_failures),
        simplify(direct_identity - with_identity) == 0,
        simplify(direct_identity - without_identity) == 0,
    )


def main() -> None:
    symbolic = symbolic_first_primary_audit()
    print(
        "SYMBOLIC k=2n in {-1,0,1}: "
        f"norms {symbolic[0]}/9; "
        f"rho with (-1)^(2n3) {symbolic[1]}/27; "
        f"rho without it {symbolic[2]}/27; "
        f"B without (-1)^(2n3) {symbolic[3]}/27; "
        f"B with (-1)^(2n3) {symbolic[4]}/27; "
        f"B^2 {symbolic[5]}/27"
    )

    samples = (
        (
            sp.Rational(3, 2),
            (sp.Rational(2, 5), sp.Rational(-1, 3), sp.Rational(3, 7)),
        ),
        (
            sp.Rational(5, 4),
            (sp.Rational(-2, 7), sp.Rational(1, 6), sp.Rational(4, 9)),
        ),
        (
            sp.Rational(2, 3),
            (sp.Rational(1, 5), sp.Rational(-2, 7), sp.Rational(3, 8)),
        ),
    )
    totals = [0, 0, 0, 0, 0, 0, 0]
    for b, momenta in samples:
        counts = exact_level_two_sample(b, momenta)
        totals = [left + right for left, right in zip(totals, counts)]
        print(
            f"EXACT b={b}, P={momenta}: norms {counts[0]}/9; "
            f"endpoint-consistent rho {counts[1]}/27; "
            f"rho with (-1)^(2n3) {counts[2]}/27; "
            f"rho without it {counts[3]}/27; "
            f"B without (-1)^(2n3) {counts[4]}/27; "
            f"B with (-1)^(2n3) {counts[5]}/27; "
            f"B^2 {counts[6]}/27"
        )

    print(
        "EXACT TOTAL k=2n in {0,1,2}: "
        f"norms {totals[0]}/{9 * len(samples)}; "
        f"endpoint-consistent rho {totals[1]}/{27 * len(samples)}; "
        f"rho with (-1)^(2n3) {totals[2]}/{27 * len(samples)}; "
        f"rho without it {totals[3]}/{27 * len(samples)}; "
        f"B without (-1)^(2n3) {totals[4]}/{27 * len(samples)}; "
        f"B with (-1)^(2n3) {totals[5]}/{27 * len(samples)}; "
        f"B^2 {totals[6]}/{27 * len(samples)}"
    )

    for b, momenta in samples[:2]:
        higher = exact_level_three_half_sample(b, momenta)
        print(
            f"HIGHER b={b}, P={momenta}: k=3 norms {higher[0]}/3; "
            f"endpoint-consistent rho {higher[1]}/9; "
            f"ordered rho with (-1)^(2n3) {higher[2]}/9; "
            f"without it {higher[3]}/9; "
            f"endpoint failures={higher[4]}; "
            f"product failures={higher[5]}; "
            f"identity with sign={higher[6]}, without sign={higher[7]}"
        )


if __name__ == "__main__":
    main()
