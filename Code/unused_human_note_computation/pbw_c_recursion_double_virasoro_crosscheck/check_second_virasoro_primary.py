#!/usr/bin/env python3
"""Direct exact audit of the NS branching formula through ``n=1``.

This is deliberately independent of the blow-up/ell product when constructing
the branching states.  It builds the level-two tensor-product module

    F_NS x V_NS(h),

forms the two commuting Virasoro generators written in ``SCblock.tex``, and
solves their highest-weight equations.  The resulting ``v_1`` is normalized
by the note's Fermi-sea convention.  Norms and ordered three-point forms are
then evaluated from the BPZ algebra and NS Ward identities using one
ungraded tensor-product pairing throughout before being compared with the
norm and branching-product formulas in the note.

Labels in the code are ``k=2n``.  The audit covers every nonnegative triple

    k_i in {0,1,2},  equivalently n_i in {0,1/2,1}.

Thus it contains all 27 triples at the next branching order, including mixed
level-one-half/level-two cases.  Reflection supplies negative labels and is
not counted as a separate direct construction.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import sympy as sp

from check_first_virasoro_primary import ell, simplify
from ns_genus2_three_way_symbolic_check import paper_branching_candidate_squared
from ns_genus2_symbolic_low_order import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    State,
    state_parity,
    state_twice_level,
)
from ns_human_convention import enlarged_ns_three_form_crossing_sign
from free_majorana_pair_of_pants import (
    FermionState,
    majorana_three_point,
    ns_fermion_states_at_twice_level,
    ns_fermion_twice_level,
)


TensorState = tuple[FermionState, State]
TensorVector = dict[TensorState, sp.Expr]


def add_term(target: dict, key, value: sp.Expr) -> None:
    value = sp.cancel(value)
    if value:
        target[key] = sp.cancel(target.get(key, sp.S.Zero) + value)
        if target[key] == 0:
            del target[key]


def tensor_basis(module: ExactNSVermaModule, twice_level: int) -> tuple[TensorState, ...]:
    states: list[TensorState] = []
    for fermion_level in range(twice_level + 1):
        sca_level = twice_level - fermion_level
        for fermion in ns_fermion_states_at_twice_level(fermion_level):
            for sca in module.basis(sca_level):
                states.append((fermion, sca))
    return tuple(states)


def fermion_mode_action(twice_mode: int, state: FermionState) -> dict[FermionState, sp.Expr]:
    """Left action of one NS fermion mode on the canonical Fock state."""

    if twice_mode == 0 or twice_mode % 2 == 0:
        raise ValueError("an NS fermion mode must be a nonzero odd half-integer")
    state = tuple(state)
    if twice_mode < 0:
        mode_number = (1 - twice_mode) // 2
        if mode_number in state:
            return {}
        position = sum(existing < mode_number for existing in state)
        new_state = tuple(sorted(state + (mode_number,)))
        return {new_state: sp.Integer(-1 if position % 2 else 1)}

    mode_number = (twice_mode + 1) // 2
    if mode_number not in state:
        return {}
    position = state.index(mode_number)
    new_state = state[:position] + state[position + 1 :]
    return {new_state: sp.Integer(-1 if position % 2 else 1)}


def prepend_fermions(prefix: FermionState, vector: dict[FermionState, sp.Expr]) -> dict[FermionState, sp.Expr]:
    result = dict(vector)
    for mode_number in reversed(prefix):
        changed: dict[FermionState, sp.Expr] = {}
        for state, coefficient in result.items():
            for new_state, action_coefficient in fermion_mode_action(
                -(2 * mode_number - 1), state
            ).items():
                add_term(changed, new_state, coefficient * action_coefficient)
        result = changed
    return result


def fermion_stress_action(mode: int, state: FermionState) -> dict[FermionState, sp.Expr]:
    r"""Use ``[L_m^F,f_r]=-(m/2+r)f_{m+r}`` exactly."""

    result: dict[FermionState, sp.Expr] = {}
    for position, mode_number in enumerate(state):
        twice_r = -(2 * mode_number - 1)
        coefficient = -sp.Rational(mode + twice_r, 2)
        if coefficient == 0:
            continue
        suffix = state[position + 1 :]
        acted = fermion_mode_action(2 * mode + twice_r, suffix)
        acted = prepend_fermions(state[:position], acted)
        for new_state, action_coefficient in acted.items():
            add_term(result, new_state, coefficient * action_coefficient)
    return result


def sca_action(
    module: ExactNSVermaModule, kind: str, twice_mode: int, state: State
) -> dict[State, sp.Expr]:
    return dict(module.mode_action((kind, twice_mode), state))


def mixed_u_action(
    module: ExactNSVermaModule, mode: int, state: TensorState
) -> dict[TensorState, sp.Expr]:
    r"""Act with ``U_m=sum_r f_{m-r}G_r`` in auxiliary-first order."""

    fermion, sca = state
    fermion_level = ns_fermion_twice_level(fermion)
    sca_level = state_twice_level(sca)
    result: dict[TensorState, sp.Expr] = {}
    # Nonzero terms require both resulting factor levels to be nonnegative:
    #   2m-fermion_level <= 2r <= sca_level.
    lower = 2 * mode - fermion_level
    upper = sca_level
    first_odd = lower if lower % 2 else lower + 1
    # This is the graded tensor-action sign in
    # (psi x G)(u x x)=(-1)^|u| psi u x Gx.  It is part of the
    # F x SCA representation and is independent of whether the bilinear
    # pairing used later is graded or ungraded.
    crossing = -1 if len(fermion) % 2 else 1
    for twice_r in range(first_odd, upper + 1, 2):
        fermion_action = fermion_mode_action(2 * mode - twice_r, fermion)
        if not fermion_action:
            continue
        sca_result = sca_action(module, "G", twice_r, sca)
        for new_fermion, fermion_coefficient in fermion_action.items():
            for new_sca, sca_coefficient in sca_result.items():
                add_term(
                    result,
                    (new_fermion, new_sca),
                    crossing * fermion_coefficient * sca_coefficient,
                )
    return result


def double_virasoro_action(
    module: ExactNSVermaModule,
    b: sp.Expr,
    copy: int,
    mode: int,
    state: TensorState,
) -> dict[TensorState, sp.Expr]:
    """Act with either of the commuting Virasoro generators in the note."""

    denominator = 1 / b - b
    if copy == 1:
        l_coefficient = (1 / b) / denominator
        f_coefficient = -(1 / b + 2 * b) / denominator
        u_coefficient = 1 / denominator
    elif copy == 2:
        l_coefficient = -b / denominator
        f_coefficient = (b + 2 / b) / denominator
        u_coefficient = -1 / denominator
    else:
        raise ValueError("copy must be 1 or 2")

    fermion, sca = state
    result: dict[TensorState, sp.Expr] = {}
    for new_sca, coefficient in sca_action(module, "L", 2 * mode, sca).items():
        add_term(result, (fermion, new_sca), l_coefficient * coefficient)
    for new_fermion, coefficient in fermion_stress_action(mode, fermion).items():
        add_term(result, (new_fermion, sca), f_coefficient * coefficient)
    for new_state, coefficient in mixed_u_action(module, mode, state).items():
        add_term(result, new_state, u_coefficient * coefficient)
    return result


def action_matrix(
    module: ExactNSVermaModule,
    b: sp.Expr,
    copy: int,
    mode: int,
    source: tuple[TensorState, ...],
    target: tuple[TensorState, ...],
) -> sp.Matrix:
    target_positions = {state: index for index, state in enumerate(target)}
    matrix = sp.zeros(len(target), len(source))
    for column, state in enumerate(source):
        for changed, coefficient in double_virasoro_action(
            module, b, copy, mode, state
        ).items():
            matrix[target_positions[changed], column] += coefficient
    return matrix.applyfunc(sp.cancel)


def first_copy_weight(b: sp.Expr, momentum: sp.Expr, n: int) -> sp.Expr:
    b1_squared = 2 * b**2 / (1 - b**2)
    q1_squared = b1_squared + 2 + 1 / b1_squared
    return sp.cancel(
        q1_squared / 4
        - (momentum + 2 * n * b) ** 2 / (2 - 2 * b**2)
    )


def solve_v1(b: sp.Expr, momentum: sp.Expr) -> tuple[ExactNSVermaModule, TensorVector]:
    """Solve the level-two highest-weight equations and fix the note's scale."""

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    h = q**2 / 8 - momentum**2 / 2
    module = ExactNSVermaModule(c=c, weight=h)
    source = tensor_basis(module, 4)
    equations: list[sp.Matrix] = []
    for copy in (1, 2):
        for mode in (1, 2):
            target = tensor_basis(module, 4 - 2 * mode)
            equations.append(action_matrix(module, b, copy, mode, source, target))
    l0_first = action_matrix(module, b, 1, 0, source, source)
    eigenvalue = first_copy_weight(b, momentum, 1)
    equations.append(l0_first - eigenvalue * sp.eye(len(source)))
    system = equations[0].col_join(equations[1]).col_join(equations[2]).col_join(
        equations[3]
    ).col_join(equations[4])
    kernel = system.nullspace()
    if len(kernel) != 1:
        raise AssertionError(f"expected one normalized n=1 eigendirection, got {len(kernel)}")
    raw = kernel[0]

    # The note orders w_1=chi_{-3/2}chi_{-1/2}|P>.  In the canonical
    # auxiliary basis f_{-1/2}f_{-3/2}|0>, its pure-fermion coefficient is
    # therefore -1.  Since v_1=2^{-2} ell(Q+2P,4) w_1, this fixes the sign as
    # well as the scale of the independently solved vector.
    pure_fermion = ((1, 2), ())
    target_coefficient = -sp.Rational(1, 4) * ell(q + 2 * momentum, 4, b, q)
    raw_coefficient = raw[source.index(pure_fermion)]
    scale = sp.cancel(target_coefficient / raw_coefficient)
    vector = {
        state: sp.cancel(scale * coefficient)
        for state, coefficient in zip(source, raw)
        if coefficient != 0
    }
    return module, vector


def v0() -> TensorVector:
    return {((), ()): sp.S.One}


def vhalf(q: sp.Expr, momentum: sp.Expr) -> TensorVector:
    return {
        ((), (("G", -1),)): sp.S.One,
        ((1,), ()): q / 2 + momentum,
    }


def fermion_bpz_inner(left: FermionState, right: FermionState) -> sp.Expr:
    if left != right:
        return sp.S.Zero
    return sp.Integer(-1 if len(left) % 2 else 1)


def tensor_inner(
    module: ExactNSVermaModule, left: TensorState, right: TensorState
) -> sp.Expr:
    """The ungraded tensor-product pairing used throughout this audit."""

    left_fermion, left_sca = left
    right_fermion, right_sca = right
    return (
        fermion_bpz_inner(left_fermion, right_fermion)
        * module.inner_product(left_sca, right_sca)
    )


def vector_norm(module: ExactNSVermaModule, vector: TensorVector) -> sp.Expr:
    result = sp.S.Zero
    for left, left_coefficient in vector.items():
        for right, right_coefficient in vector.items():
            result += (
                left_coefficient
                * right_coefficient
                * tensor_inner(module, left, right)
            )
    return simplify(result)


def current_sca_three_point(
    form: ExactNSDescendantThreeForm, states: tuple[State, State, State]
) -> sp.Expr:
    """Return the Ward engine's human-note fixed-parity three-form."""

    return form.value(*states)


def current_fermion_three_point(
    states: tuple[FermionState, FermionState, FermionState]
) -> sp.Expr:
    # psi_r^dagger=-psi_{-r} contributes one minus sign per bra fermion.
    return sp.Integer(-1 if len(states[0]) % 2 else 1) * majorana_three_point(
        *states
    )


def tensor_three_point(
    form: ExactNSDescendantThreeForm,
    vectors: tuple[TensorVector, TensorVector, TensorVector],
) -> sp.Expr:
    """Three-form defined by the note's ungraded product matrix element."""

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
        result += (
            coefficient
            * crossing
            * fermion_value
            * current_sca_three_point(form, sca_states)
        )
    return simplify(result)


def graded_tensor_three_point_diagnostic(
    form: ExactNSDescendantThreeForm,
    vectors: tuple[TensorVector, TensorVector, TensorVector],
) -> sp.Expr:
    """Use the standard graded tensor crossing as a pairing diagnostic.

    This is not the branching three-form.  It is retained only to demonstrate
    explicitly how the standard graded tensor BPZ convention differs from the
    ungraded product convention selected in the note.
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
        absolute_sca = tuple(
            sca_parity ^ primary_parity
            for sca_parity, primary_parity in zip(
                sca_parities, form.primary_parities
            )
        )
        crossing_exponent = absolute_sca[0] * (
            fermion_parities[1] + fermion_parities[2]
        )
        crossing_exponent += absolute_sca[1] * fermion_parities[2]
        result += (
            coefficient
            * (-1 if crossing_exponent % 2 else 1)
            * fermion_value
            * current_sca_three_point(form, sca_states)
        )
    return simplify(result)


def product_norm(k: int, momentum: sp.Expr, b: sp.Expr, q: sp.Expr) -> sp.Expr:
    return simplify(
        (-1) ** k
        * sp.Pow(2, -k)
        * ell(2 * momentum, 2 * k, b, q)
        * ell(q + 2 * momentum, 2 * k, b, q)
    )


def ell_numerator(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
) -> sp.Expr:
    k1, k2, k3 = labels
    p1, p2, p3 = momenta
    arguments = (
        q / 2 + p1 + p2 + p3,
        q / 2 - p1 + p2 + p3,
        q / 2 + p1 + p2 - p3,
        q / 2 - p1 + p2 - p3,
    )
    indices = (
        k1 + k2 + k3,
        -k1 + k2 + k3,
        k1 + k2 - k3,
        -k1 + k2 - k3,
    )
    return simplify(sp.prod(ell(x, m, b, q) for x, m in zip(arguments, indices)))


def product_rho(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
    *,
    third_slot_sign: bool,
) -> sp.Expr:
    label_sum = sum(labels)
    parity = label_sum % 2
    # The note's analytic branch is (-1)^(1/2)=-i.  Hence
    # (-1)^(n_1+n_2+n_3)i^a=(-i)^(sum k_i)i^a.
    phase = (-sp.I) ** label_sum * sp.I**parity
    slot_sign = (-1) ** labels[2] if third_slot_sign else 1
    return simplify(
        slot_sign
        * phase
        * sp.Pow(2, -sp.Rational(label_sum, 2))
        * ell_numerator(labels, momenta, b, q)
    )


def verify_v1_primary(
    module: ExactNSVermaModule,
    b: sp.Expr,
    momentum: sp.Expr,
    vector: TensorVector,
) -> int:
    """Recheck the five defining highest-weight/eigenvalue equations."""

    source = tensor_basis(module, 4)
    column = sp.Matrix([vector.get(state, sp.S.Zero) for state in source])
    checked = 0
    for copy in (1, 2):
        for mode in (1, 2):
            target = tensor_basis(module, 4 - 2 * mode)
            residual = action_matrix(
                module, b, copy, mode, source, target
            ) * column
            if any(simplify(entry) != 0 for entry in residual):
                raise AssertionError(
                    f"v_1 is not primary for copy={copy}, mode={mode}: "
                    f"{tuple(map(simplify, residual))}"
                )
            checked += 1

    l0_first = action_matrix(module, b, 1, 0, source, source)
    eigenvalue = first_copy_weight(b, momentum, 1)
    residual = (l0_first - eigenvalue * sp.eye(len(source))) * column
    if any(simplify(entry) != 0 for entry in residual):
        raise AssertionError(
            "v_1 has the wrong first-copy L_0 eigenvalue: "
            f"{tuple(map(simplify, residual))}"
        )
    return checked + 1


def run_sample(
    b: sp.Rational, momenta: tuple[sp.Rational, sp.Rational, sp.Rational]
) -> tuple[int, int, int, int, int, int]:
    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
    modules: list[ExactNSVermaModule] = []
    vectors_by_slot: list[dict[int, TensorVector]] = []
    direct_norms_by_slot: list[dict[int, sp.Expr]] = []
    primary_passes = 0
    norm_passes = 0
    for momentum in momenta:
        module, second = solve_v1(b, momentum)
        primary_passes += verify_v1_primary(module, b, momentum, second)
        vectors = {0: v0(), 1: vhalf(q, momentum), 2: second}
        direct_norms: dict[int, sp.Expr] = {}
        modules.append(module)
        vectors_by_slot.append(vectors)
        for label, vector in vectors.items():
            direct = vector_norm(module, vector)
            direct_norms[label] = direct
            expected = product_norm(label, momentum, b, q)
            residual = simplify(direct - expected)
            if residual != 0:
                raise AssertionError(
                    f"norm mismatch at b={b}, P={momentum}, k={label}: {residual}"
                )
            norm_passes += 1
        direct_norms_by_slot.append(direct_norms)

    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    reflected_form = ExactNSDescendantThreeForm(
        c=c, weights=(weights[2], weights[1], weights[0])
    )
    rho_passes = 0
    old_rho_passes = 0
    squared_passes = 0
    endpoint_consistency_passes = 0
    rho_mismatches: list[
        tuple[tuple[int, int, int], sp.Expr, sp.Expr, sp.Expr]
    ] = []
    b_squared_mismatches: list[
        tuple[tuple[int, int, int], sp.Expr, sp.Expr, sp.Expr]
    ] = []
    endpoint_mismatches: list[tuple[tuple[int, int, int], sp.Expr]] = []
    for labels in product((0, 1, 2), repeat=3):
        vectors = tuple(
            vectors_by_slot[slot][label] for slot, label in enumerate(labels)
        )
        direct = tensor_three_point(form, vectors)
        # Endpoint exchange is retained as a diagnostic, but the ordered
        # coefficient at (infinity,1,0) remains the definition.  In
        # particular, do not discard (2,2,2): it is the first case that can
        # expose an incompatibility between the ungraded tensor pairing and
        # the factorized three-point product.
        reflection_sign = (-1) ** ((labels[0] + labels[2]) % 2)
        reflected_direct = simplify(
            reflection_sign
            * tensor_three_point(
                reflected_form, (vectors[2], vectors[1], vectors[0])
            )
        )
        endpoint_residual = simplify(direct - reflected_direct)
        if endpoint_residual != 0:
            endpoint_mismatches.append((labels, endpoint_residual))
        else:
            endpoint_consistency_passes += 1
        expected = product_rho(
            labels, momenta, b, q, third_slot_sign=True
        )
        old_expected = product_rho(
            labels, momenta, b, q, third_slot_sign=False
        )
        residual = simplify(direct - expected)
        old_residual = simplify(direct - old_expected)
        if residual == 0:
            rho_passes += 1
        else:
            rho_mismatches.append((labels, direct, expected, residual))
        if old_residual == 0:
            old_rho_passes += 1

        direct_norm_product = sp.prod(
            direct_norms_by_slot[slot][label]
            for slot, label in enumerate(labels)
        )
        formula_norm_product = sp.prod(
            product_norm(label, momentum, b, q)
            for label, momentum in zip(labels, momenta)
        )
        direct_b_squared = simplify(direct**2 / direct_norm_product)
        # Compare the direct matrix element with the *identical* all-label
        # coefficient used by the NS c-recursion/double-Virasoro check.  Keep
        # the local product-rho reconstruction only as a dictionary audit so
        # the two implementations cannot silently drift apart.
        reconstructed_b_squared = simplify(expected**2 / formula_norm_product)
        formula_b_squared = simplify(
            paper_branching_candidate_squared(
                momenta=momenta,
                labels=labels,
                b=b,
            )
        )
        if simplify(reconstructed_b_squared - formula_b_squared) != 0:
            raise AssertionError(
                f"branching-formula dictionary mismatch at k={labels}"
            )
        if simplify(direct_b_squared - formula_b_squared) == 0:
            squared_passes += 1
        else:
            b_squared_mismatches.append(
                (
                    labels,
                    direct_b_squared,
                    formula_b_squared,
                    simplify(direct_b_squared - formula_b_squared),
                )
            )
    if rho_mismatches:
        print(
            "  third-slot-corrected rho mismatches: "
            + ", ".join(
                f"k={labels} (direct={sp.factor(direct)}, "
                f"product={sp.factor(expected)}, "
                f"ratio={sp.factor(direct / expected)}, "
                f"residual={sp.factor(residual)})"
                for labels, direct, expected, residual in rho_mismatches
            )
        )
    if b_squared_mismatches:
        print(
            "  normalized B^2 mismatches: "
            + ", ".join(
                f"k={labels} (ratio={sp.factor(direct / expected)}, "
                f"residual={sp.factor(residual)})"
                for labels, direct, expected, residual in b_squared_mismatches
            )
        )
    if endpoint_mismatches:
        print(
            "  endpoint-exchange diagnostic failures: "
            + ", ".join(
                f"k={labels} (residual={sp.factor(residual)})"
                for labels, residual in endpoint_mismatches
            )
        )
    return (
        primary_passes,
        norm_passes,
        endpoint_consistency_passes,
        rho_passes,
        old_rho_passes,
        squared_passes,
    )


def identity_specialization(
    b: sp.Rational, momentum: sp.Rational
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Compare the identity three-form in both tensor BPZ conventions.

    The middle weight is kept symbolic during Ward reduction and set to zero
    only afterward, avoiding inversion of the singular vacuum Verma Gram
    matrix.
    """

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    h = q**2 / 8 - momentum**2 / 2
    middle_weight = sp.symbols("h_identity_probe")
    module, vector = solve_v1(b, momentum)
    form = ExactNSDescendantThreeForm(
        c=c, weights=(h, middle_weight, h)
    )
    direct_identity = simplify(
        tensor_three_point(form, (vector, v0(), vector)).subs(
            middle_weight, 0
        )
    )
    direct_graded_identity = simplify(
        graded_tensor_three_point_diagnostic(
            form, (vector, v0(), vector)
        ).subs(middle_weight, 0)
    )

    diagnostic_graded_norm = sp.S.Zero
    for left, left_coefficient in vector.items():
        for right, right_coefficient in vector.items():
            left_fermion, left_sca = left
            right_fermion, right_sca = right
            graded_sign = -1 if (
                len(right_fermion) * state_parity(left_sca)
            ) % 2 else 1
            diagnostic_graded_norm += (
                left_coefficient
                * right_coefficient
                * graded_sign
                * fermion_bpz_inner(left_fermion, right_fermion)
                * module.inner_product(left_sca, right_sca)
            )
    diagnostic_graded_norm = simplify(diagnostic_graded_norm)
    ungraded_norm = vector_norm(module, vector)
    ell_identity = product_rho(
        (2, 0, 2),
        (momentum, q / 2, momentum),
        b,
        q,
        third_slot_sign=True,
    )
    if simplify(direct_identity - ungraded_norm) != 0:
        raise AssertionError("matrix-element three-form does not reproduce the ungraded norm")
    if simplify(direct_graded_identity - diagnostic_graded_norm) != 0:
        raise AssertionError("graded diagnostic does not reproduce the graded norm")
    if simplify(ell_identity - ungraded_norm) != 0:
        raise AssertionError("ell identity specialization is not the ungraded norm")
    if simplify(diagnostic_graded_norm - ungraded_norm) == 0:
        raise AssertionError("the level-two probe failed to distinguish the pairings")
    return direct_identity, ungraded_norm, diagnostic_graded_norm


def identity_basis_audit(
    b: sp.Rational, momentum: sp.Rational, *, max_twice_level: int = 6
) -> int:
    """Check that the standard graded crossing induces the graded pairing.

    The default cutoff covers all 370 pairs of equal-level tensor states
    through physical level three.  This diagnostic is kept separate from the
    ungraded matrix-element three-form used by the branching calculation.
    """

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    h = q**2 / 8 - momentum**2 / 2
    middle_weight = sp.symbols("h_identity_basis_audit")
    module = ExactNSVermaModule(c=c, weight=h)
    form = ExactNSDescendantThreeForm(
        c=c, weights=(h, middle_weight, h)
    )
    checked = 0
    for twice_level in range(max_twice_level + 1):
        basis = tensor_basis(module, twice_level)
        for left in basis:
            for right in basis:
                direct = graded_tensor_three_point_diagnostic(
                    form,
                    ({left: sp.S.One}, v0(), {right: sp.S.One}),
                ).subs(middle_weight, 0)
                right_fermion, _ = right
                _, left_sca = left
                graded_sign = -1 if (
                    len(right_fermion) * state_parity(left_sca)
                ) % 2 else 1
                expected = graded_sign * tensor_inner(module, left, right)
                residual = simplify(direct - expected)
                if residual != 0:
                    raise AssertionError(
                        "graded identity mismatch at "
                        f"twice_level={twice_level}, left={left}, "
                        f"right={right}: {residual}"
                    )
                checked += 1
    return checked


def main() -> None:
    samples = (
        (sp.Rational(3, 2), (sp.Rational(2, 5), sp.Rational(-1, 3), sp.Rational(3, 7))),
        (sp.Rational(5, 4), (sp.Rational(-2, 7), sp.Rational(1, 6), sp.Rational(4, 9))),
    )
    totals = [0, 0, 0, 0, 0, 0]
    identity_basis_checks = identity_basis_audit(
        samples[0][0], samples[0][1][0]
    )
    print(
        "standard-graded identity-on-basis diagnostic: "
        f"{identity_basis_checks}/{identity_basis_checks}"
    )
    for b, momenta in samples:
        counts = run_sample(b, momenta)
        totals = [left + right for left, right in zip(totals, counts)]
        print(
            f"sample b={b}, P={momenta}: v1 primary equations "
            f"{counts[0]}/15; norms {counts[1]}/9; "
            f"endpoint-consistent rho cases {counts[2]}/27; "
            f"ordered rho with (-1)^(2n_3) {counts[3]}/27; "
            f"rho without it {counts[4]}/27; B^2 {counts[5]}/27"
        )
        direct_identity, ungraded_norm, graded_norm = identity_specialization(
            b, momenta[0]
        )
        print(
            "  identity probe (k=(2,0,2)): "
            f"rho(v1,1,v1)={direct_identity}; ungraded/ell norm={ungraded_norm}; "
            f"diagnostic graded norm={graded_norm}"
        )
    print(
        "TOTAL: "
        f"v1 primary equations {totals[0]}/{15 * len(samples)}, "
        f"norms {totals[1]}/{9 * len(samples)}, "
        f"endpoint-consistent rho cases {totals[2]}/{27 * len(samples)}, "
        f"ordered rho with (-1)^(2n_3) "
        f"{totals[3]}/{27 * len(samples)}, "
        f"rho without it {totals[4]}/{27 * len(samples)}, "
        f"B^2 {totals[5]}/{27 * len(samples)}"
    )


if __name__ == "__main__":
    main()
