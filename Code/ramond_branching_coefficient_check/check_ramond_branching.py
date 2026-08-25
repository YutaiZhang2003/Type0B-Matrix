"""State-level checks of the Ramond two-Virasoro branching coefficient.

The code uses the conventions of the accompanying note.  It does not insert
the conjectured answer while constructing a state.  Instead it

1. builds the Ramond free-field transition matrix at each integer level;
2. expands the two-fermion strings which define the simultaneous
   Virasoro x Virasoro highest states;
3. transports every component to the abstract Ramond PBW basis;
4. constructs the abstract Ramond Gram matrix from the SCA commutators;
5. projects the even branch state on the Ramond Whittaker vector; and
6. compares the squared projection with the odd-screening product.

The labels n=1/4 and n=3/4 are checked symbolically for both signs and both
parity copies.  At n=5/4 the complete state is still constructed, but the
comparison is made at selected exact rational values of b and P so that the
test remains quick and independently reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)
P, Q = sp.symbols("P Q")
H, CENTRAL_CHARGE = sp.symbols("H CENTRAL_CHARGE")


def add_term(out, state, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    out[state] = sp.cancel(out.get(state, 0) + coefficient)
    if out[state] == 0:
        del out[state]


@lru_cache(None)
def partitions(total, largest=None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def strict_partitions(total, largest=None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in strict_partitions(total - first, first - 1):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def basis(level):
    """PBW/Fock basis (bosons, distinct positive fermion modes, ground sign)."""
    answer = []
    for fermion_level in range(level + 1):
        for bosons in partitions(level - fermion_level):
            for fermions in strict_partitions(fermion_level):
                for ground in (0, 1):
                    answer.append((bosons, fermions, ground))
    return tuple(answer)


def state_parity(state):
    _, fermions, ground = state
    return (len(fermions) + ground) % 2


def apply_c(mode, state):
    bosons, fermions, ground = state
    if mode < 0:
        created = -mode
        return (tuple(sorted(bosons + (created,), reverse=True)), fermions, ground), 1
    if mode == 0:
        raise AssertionError("The bosonic zero mode has already been evaluated.")
    count = bosons.count(mode)
    if count == 0:
        return None, 0
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining), fermions, ground), mode * count


def apply_fermion(mode, state, realization):
    """Apply the physical Ramond fermion.

    realization=-1 is the paper's psi^- realization and has
    psi_0|P,+> = +|P,->/sqrt(2).  realization=+1 is psi^+ and has the
    opposite zero-mode sign.
    """
    bosons, fermions, ground = state
    if mode < 0:
        created = -mode
        if created in fermions:
            return None, 0
        crossings = sum(existing > created for existing in fermions)
        new_fermions = tuple(sorted(fermions + (created,), reverse=True))
        return (bosons, new_fermions, ground), (-1) ** crossings
    if mode > 0:
        if mode not in fermions:
            return None, 0
        position = fermions.index(mode)
        remaining = fermions[:position] + fermions[position + 1 :]
        return (bosons, remaining, ground), (-1) ** position

    # psi_0 must cross every negative physical-fermion creator before it
    # reaches the ground doublet.  The two free-field realizations have
    # opposite zero-mode signs.
    zero_sign = 1 if realization == -1 else -1
    coefficient = (-1) ** len(fermions) * zero_sign / SQRT2
    return (bosons, fermions, 1 - ground), coefficient


def apply_two(first, second, state):
    middle, coefficient_second = second(state)
    if coefficient_second == 0:
        return None, 0
    final, coefficient_first = first(middle)
    if coefficient_first == 0:
        return None, 0
    return final, coefficient_second * coefficient_first


def apply_L_to_state(mode, state, realization, momentum=P):
    assert mode < 0
    bosons, fermions, _ = state
    out = {}

    bosonic_indices = set(range(mode + 1, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update(mode - existing for existing in bosons)
    for summation_mode in bosonic_indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = apply_two(
            lambda current, m=mode - summation_mode: apply_c(m, current),
            lambda current, m=summation_mode: apply_c(m, current),
            state,
        )
        if coefficient:
            add_term(out, final, sp.Rational(1, 2) * coefficient)

    fermionic_indices = set(range(mode, 1))
    fermionic_indices.update(fermions)
    fermionic_indices.update(mode - existing for existing in fermions)
    for summation_mode in fermionic_indices:
        final, coefficient = apply_two(
            lambda current, r=mode - summation_mode: apply_fermion(
                r, current, realization
            ),
            lambda current, r=summation_mode: apply_fermion(
                r, current, realization
            ),
            state,
        )
        if coefficient:
            add_term(out, final, sp.Rational(summation_mode, 2) * coefficient)

    final, coefficient = apply_c(mode, state)
    if coefficient:
        momentum_term = Q * mode - 2 * momentum if realization == -1 else Q * mode + 2 * momentum
        add_term(out, final, I * sp.Rational(1, 2) * momentum_term * coefficient)
    return out


def apply_G_to_state(mode, state, realization, momentum=P):
    assert mode < 0
    bosons, fermions, _ = state
    out = {}

    bosonic_indices = set(range(mode, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update(mode - existing for existing in fermions)
    for summation_mode in bosonic_indices:
        if summation_mode == 0:
            continue
        final, coefficient = apply_two(
            lambda current, m=summation_mode: apply_c(m, current),
            lambda current, r=mode - summation_mode: apply_fermion(
                r, current, realization
            ),
            state,
        )
        if coefficient:
            add_term(out, final, coefficient)

    final, coefficient = apply_fermion(mode, state, realization)
    if coefficient:
        momentum_term = Q * mode - momentum if realization == -1 else Q * mode + momentum
        add_term(out, final, I * momentum_term * coefficient)
    return out


def apply_to_expression(action, expression):
    out = {}
    for state, outer_coefficient in expression.items():
        for final, inner_coefficient in action(state).items():
            add_term(out, final, outer_coefficient * inner_coefficient)
    return out


def descendant_to_fock(descendant, realization, momentum=P):
    virasoro_modes, supercurrent_modes, ground = descendant
    expression = {((), (), ground): sp.Integer(1)}
    for mode in reversed(supercurrent_modes):
        expression = apply_to_expression(
            lambda state, r=-mode: apply_G_to_state(
                r, state, realization, momentum
            ),
            expression,
        )
    for mode in reversed(virasoro_modes):
        expression = apply_to_expression(
            lambda state, n=-mode: apply_L_to_state(
                n, state, realization, momentum
            ),
            expression,
        )
    return expression


@lru_cache(None)
def transition(level, realization):
    ordered_basis = basis(level)
    matrix = sp.zeros(len(ordered_basis), len(ordered_basis))
    row = {state: index for index, state in enumerate(ordered_basis)}
    for column, descendant in enumerate(ordered_basis):
        for state, coefficient in descendant_to_fock(descendant, realization).items():
            matrix[row[state], column] = coefficient
    return ordered_basis, matrix


def super_bracket(first, second):
    """Return the graded bracket of two standard-normalized Ramond modes."""
    first_kind, first_mode = first
    second_kind, second_mode = second
    answer = []
    if first_kind == "L" and second_kind == "L":
        answer.append((first_mode - second_mode, (("L", first_mode + second_mode),)))
        if first_mode + second_mode == 0:
            answer.append((CENTRAL_CHARGE * (first_mode**3 - first_mode) / 12, ()))
    elif first_kind == "L" and second_kind == "G":
        answer.append((sp.Rational(first_mode, 2) - second_mode, (("G", first_mode + second_mode),)))
    elif first_kind == "G" and second_kind == "L":
        answer.append((first_mode - sp.Rational(second_mode, 2), (("G", first_mode + second_mode),)))
    else:
        answer.append((sp.Integer(2), (("L", first_mode + second_mode),)))
        if first_mode + second_mode == 0:
            answer.append((CENTRAL_CHARGE * (first_mode**2 - sp.Rational(1, 4)) / 3, ()))
    return tuple((sp.expand(coefficient), word) for coefficient, word in answer)


@lru_cache(None)
def highest_weight_expectation(word, left_ground, right_ground):
    """Evaluate <left|word|right> from the Ramond SCA alone."""
    if not word:
        return sp.Integer(left_ground == right_ground)
    if word[0][1] < 0 or word[-1][1] > 0:
        return sp.Integer(0)

    # Push every negative mode to the left.  This also handles a zero mode
    # sitting immediately to the left of a negative mode.
    for position in range(len(word) - 1):
        first, second = word[position], word[position + 1]
        if first[1] >= 0 and second[1] < 0:
            exchange_sign = -1 if first[0] == second[0] == "G" else 1
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            answer = exchange_sign * highest_weight_expectation(
                exchanged, left_ground, right_ground
            )
            for coefficient, replacement in super_bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                answer += coefficient * highest_weight_expectation(
                    reduced, left_ground, right_ground
                )
            return sp.expand(answer)

    # Push positive modes to the right through zero modes.
    for position in range(len(word) - 1):
        first, second = word[position], word[position + 1]
        if first[1] > 0 and second[1] == 0:
            exchange_sign = -1 if first[0] == second[0] == "G" else 1
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            answer = exchange_sign * highest_weight_expectation(
                exchanged, left_ground, right_ground
            )
            for coefficient, replacement in super_bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                answer += coefficient * highest_weight_expectation(
                    reduced, left_ground, right_ground
                )
            return sp.expand(answer)

    if any(mode < 0 for _, mode in word) or any(mode > 0 for _, mode in word):
        raise AssertionError(f"Nonzero modes were not reduced: {word}")

    l_zero_count = sum(kind == "L" for kind, _ in word)
    g_zero_count = sum(kind == "G" for kind, _ in word)
    g0 = sp.Matrix([[0, -I * P / SQRT2], [-I * P / SQRT2, 0]])
    return sp.expand(H**l_zero_count * (g0**g_zero_count)[left_ground, right_ground])


def descendant_word(descendant, bra=False):
    virasoro_modes, supercurrent_modes, _ = descendant
    if not bra:
        return tuple(("L", -mode) for mode in virasoro_modes) + tuple(
            ("G", -mode) for mode in supercurrent_modes
        )
    return tuple(("G", mode) for mode in reversed(supercurrent_modes)) + tuple(
        ("L", mode) for mode in reversed(virasoro_modes)
    )


@lru_cache(None)
def abstract_gram(level):
    ordered_basis = basis(level)
    matrix = sp.zeros(len(ordered_basis), len(ordered_basis))
    for row, left in enumerate(ordered_basis):
        left_word = descendant_word(left, bra=True)
        for column, right in enumerate(ordered_basis):
            matrix[row, column] = highest_weight_expectation(
                left_word + descendant_word(right), left[2], right[2]
            )
    # In a Ramond module H and the G_0 matrix are constrained by
    # G_0^2=H-C/24=-P^2/2.  Symmetry is therefore tested on that locus.
    symmetry_difference = (matrix - matrix.T).subs(
        H, CENTRAL_CHARGE / 24 - P**2 / 2
    )
    if any(sp.expand(entry) != 0 for entry in symmetry_difference):
        raise AssertionError(f"Ramond Gram matrix at level {level} is not symmetric.")
    return ordered_basis, matrix


def apply_auxiliary(mode, state):
    """Apply f_mode to (negative modes, ground sign)."""
    modes, ground = state
    if mode < 0:
        created = -mode
        if created in modes:
            return None, 0
        crossings = sum(existing > created for existing in modes)
        return (tuple(sorted(modes + (created,), reverse=True)), ground), (-1) ** crossings
    if mode == 0:
        return (modes, 1 - ground), (-1) ** len(modes) / SQRT2
    if mode not in modes:
        return None, 0
    position = modes.index(mode)
    return (modes[:position] + modes[position + 1 :], ground), (-1) ** position


def expand_chi_string(branch_label, parity):
    """Expand one paper-normalized chi string in the two free fermions."""
    branch_label = sp.Rational(branch_label)
    if 4 * branch_label not in sp.S.Integers or int(4 * branch_label) % 2 == 0:
        raise ValueError("The Ramond branch label must lie in Z/2+1/4.")
    realization = -1 if branch_label > 0 else 1
    largest_mode = int(2 * abs(branch_label) - sp.Rational(1, 2))
    operators = [0] + [-mode for mode in range(1, largest_mode + 1)]
    raw_parity = len(operators) % 2
    if raw_parity != parity:
        operators.append(0)  # the opposite realization chi_0 is used below

    # State: auxiliary modes, auxiliary ground, physical Fock state.
    expression = {((), 0, (), (), 0): sp.Integer(1)}
    for position, mode in reversed(tuple(enumerate(operators))):
        next_expression = {}
        for full_state, outer in expression.items():
            aux_modes, aux_ground, bosons, phys_modes, phys_ground = full_state

            aux_final, aux_coefficient = apply_auxiliary(
                mode, (aux_modes, aux_ground)
            )
            if aux_coefficient:
                final = (
                    aux_final[0],
                    aux_final[1],
                    bosons,
                    phys_modes,
                    phys_ground,
                )
                add_term(next_expression, final, outer * aux_coefficient)

            physical_realization = realization
            if mode == 0 and position == len(operators) - 1 and raw_parity != parity:
                physical_realization = -realization
            phys_final, phys_coefficient = apply_fermion(
                mode,
                (bosons, phys_modes, phys_ground),
                physical_realization,
            )
            if phys_coefficient:
                aux_parity = (len(aux_modes) + aux_ground) % 2
                final = (
                    aux_modes,
                    aux_ground,
                    phys_final[0],
                    phys_final[1],
                    phys_final[2],
                )
                add_term(
                    next_expression,
                    final,
                    outer * (-I) * (-1) ** aux_parity * phys_coefficient,
                )
        expression = next_expression
    return realization, tuple(operators), expression


def branch_in_abstract_basis(branch_label, parity, substitutions=None):
    realization, operators, fock_expression = expand_chi_string(branch_label, parity)
    sectors = defaultdict(lambda: defaultdict(lambda: 0))
    for state, coefficient in fock_expression.items():
        aux_modes, aux_ground, bosons, phys_modes, phys_ground = state
        sectors[(aux_modes, aux_ground)][(bosons, phys_modes, phys_ground)] += coefficient

    abstract_sectors = {}
    for auxiliary_state, physical_expression in sectors.items():
        level = sum(next(iter(physical_expression))[0]) + sum(next(iter(physical_expression))[1])
        ordered_basis, matrix = transition(level, realization)
        vector = sp.zeros(len(ordered_basis), 1)
        row = {state: index for index, state in enumerate(ordered_basis)}
        for state, coefficient in physical_expression.items():
            vector[row[state]] += coefficient
        if substitutions:
            matrix = matrix.subs(substitutions, simultaneous=True)
            vector = vector.subs(substitutions, simultaneous=True)
        abstract_sectors[auxiliary_state] = (
            level,
            ordered_basis,
            matrix.inv() * vector,
        )
    return operators, abstract_sectors


def auxiliary_norm(auxiliary_state):
    modes, ground = auxiliary_state
    return (-1) ** len(modes) * (1 if ground == 0 else -1)


def substituted_gram(level):
    highest_weight = sp.Rational(1, 16) + Q**2 / 8 - P**2 / 2
    central_charge = sp.Rational(3, 2) + 3 * Q**2
    return abstract_gram(level)[1].subs(
        {H: highest_weight, CENTRAL_CHARGE: central_charge}, simultaneous=True
    )


def branch_norm(branch_label, parity, substitutions=None):
    operators, sectors = branch_in_abstract_basis(
        branch_label, parity, substitutions=substitutions
    )
    total = 0
    component_count = 0
    for auxiliary_state, (level, _, coefficients) in sectors.items():
        gram = substituted_gram(level)
        if substitutions:
            gram = gram.subs(substitutions, simultaneous=True)
        total += auxiliary_norm(auxiliary_state) * (coefficients.T * gram * coefficients)[0]
        component_count += sum(sp.cancel(value) != 0 for value in coefficients)
    return operators, sectors, component_count, sp.factor(sp.cancel(total))


def whittaker_projection_squared(branch_label, substitutions=None):
    """Squared reduced l_n^{+,+}: overlap^2 divided by branch norm."""
    operators, sectors, component_count, norm = branch_norm(
        branch_label, parity=0, substitutions=substitutions
    )
    auxiliary_ground = ((), 0)
    if auxiliary_ground not in sectors:
        return operators, component_count, norm, sp.Integer(0)
    level, ordered_basis, coefficients = sectors[auxiliary_ground]
    target = ((1,) * level, (), 0)
    target_index = ordered_basis.index(target)
    overlap = coefficients[target_index] * sp.Rational(1, 2) ** level
    return operators, component_count, norm, sp.factor(sp.cancel(overlap**2 / norm))


def s_odd(x, argument, b):
    argument = sp.Rational(argument)
    if argument < 0:
        return s_odd(b + 1 / b - x, -argument, b)
    threshold = int(2 * argument)
    answer = sp.Pow(2, sp.Rational(1, 8))
    for i in range(threshold):
        for j in range(threshold - i):
            if (i + j) % 2 == 1:
                answer *= x + i * b + j / b
    return sp.factor(answer)


def paper_reduced_projection(branch_label, b):
    branch_label = sp.Rational(branch_label)
    q_value = b + 1 / b
    return sp.factor(
        2 ** (4 * branch_label**2 - 1)
        / (
            s_odd(2 * P, 2 * branch_label, b)
            * s_odd(2 * P + q_value, 2 * branch_label, b)
        )
    )


def conjectured_reduced_projection(branch_label, b):
    """The paper's product after converting its Whittaker coordinate to ours.

    The Ramond Whittaker vectors in the present notes obey
    L_1|N>=(1/2)|N-1>.  At the branch onset
    N=2*n**2-1/8, so a highest-state projection acquires 2^{-N}, and
    its square acquires 4^{-N}.  The published product is written in the
    coordinate in which this factor is absent.
    """
    branch_label = sp.Rational(branch_label)
    onset_level = 2 * branch_label**2 - sp.Rational(1, 8)
    return sp.factor(
        4 ** (-onset_level) * paper_reduced_projection(branch_label, b)
    )


def check_transition_and_gram():
    level_zero_gram = abstract_gram(0)[1]
    if level_zero_gram != sp.eye(2):
        raise AssertionError("The Ramond ground Gram matrix is incorrect.")
    expected_level_one = sp.Matrix(
        [
            [2 * H, 0, 0, -3 * I * P / (2 * SQRT2)],
            [0, 2 * H, -3 * I * P / (2 * SQRT2), 0],
            [0, -3 * I * P / (2 * SQRT2), 2 * H + CENTRAL_CHARGE / 4, 0],
            [-3 * I * P / (2 * SQRT2), 0, 0, 2 * H + CENTRAL_CHARGE / 4],
        ]
    )
    calculated = abstract_gram(1)[1]
    if any(sp.expand(entry) != 0 for entry in calculated - expected_level_one):
        raise AssertionError(f"Unexpected level-one Gram matrix:\n{calculated}")

    # Both free-field realizations must represent the same G_0 action.
    for realization in (-1, 1):
        for ground in (0, 1):
            final, coefficient = apply_fermion(0, ((), (), ground), realization)
            free_g0 = (-I * P if realization == -1 else I * P) * coefficient
            if sp.simplify(free_g0 + I * P / SQRT2) != 0:
                raise AssertionError("The free-field G_0 convention is inconsistent.")


def symbolic_low_checks():
    b = sp.symbols("b", nonzero=True)
    results = []
    for absolute_label in (sp.Rational(1, 4), sp.Rational(3, 4)):
        for branch_label in (absolute_label, -absolute_label):
            for parity in (0, 1):
                operators, _, count, norm = branch_norm(branch_label, parity)
                if norm == 0:
                    raise AssertionError(f"Null branch state at n={branch_label}, parity={parity}.")
                results.append((branch_label, parity, tuple(operators), count, norm))

            operators, count, norm, calculated = whittaker_projection_squared(branch_label)
            expected = conjectured_reduced_projection(branch_label, b).subs(
                b + 1 / b, Q
            )
            # At these levels the expected product can be reduced using Q=b+b^{-1}.
            difference = sp.factor(
                sp.together(calculated.subs(Q, b + 1 / b) - expected)
            )
            if difference != 0:
                raise AssertionError(
                    f"Reduced projection mismatch at n={branch_label}: {difference}"
                )
    return results


def selected_level_five_quarters_checks():
    samples = (
        (sp.Rational(3, 2), sp.Rational(2, 5)),
        (sp.Rational(5, 3), sp.Rational(7, 10)),
    )
    reports = []
    for branch_label in (sp.Rational(5, 4), -sp.Rational(5, 4)):
        for b_value, p_value in samples:
            substitutions = {Q: b_value + 1 / b_value, P: p_value}
            operators, count, norm, calculated_value = whittaker_projection_squared(
                branch_label, substitutions=substitutions
            )
            calculated_value = sp.factor(calculated_value)
            expected_value = sp.factor(
                conjectured_reduced_projection(branch_label, b_value).subs(P, p_value)
            )
            difference = sp.factor(calculated_value - expected_value)
            if difference != 0:
                raise AssertionError(
                    f"n={branch_label}, b={b_value}, P={p_value}: {difference}"
                )
            reports.append(
                (branch_label, tuple(operators), count, b_value, p_value, calculated_value)
            )
    return reports


def main():
    check_transition_and_gram()
    print("Ramond ground and level-one Gram matrices: exact")
    print("Both free-field zero-mode realizations: exact")

    low_results = symbolic_low_checks()
    for branch_label, parity, operators, count, norm in low_results:
        print(
            f"n={branch_label}, parity={parity}, operators={operators}, "
            f"nonzero PBW components={count}, norm={norm}"
        )
    print("All n=+/-1/4,+/-3/4 states and reduced projections: exact symbolic match")

    high_results = selected_level_five_quarters_checks()
    for branch_label, operators, count, b_value, p_value, value in high_results:
        print(
            f"n={branch_label}, operators={operators}, nonzero PBW components={count}, "
            f"b={b_value}, P={p_value}, reduced projection^2={value}"
        )
    print("Selected n=+/-5/4 reduced projections: exact rational matches")


if __name__ == "__main__":
    main()
