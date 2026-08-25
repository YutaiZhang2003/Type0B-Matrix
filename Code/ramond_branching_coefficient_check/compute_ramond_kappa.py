"""Direct low-level computation of the discrete Ramond branching factor.

The one-leg Whittaker calculation fixes the momentum-dependent norm of a
Ramond Virasoro x Virasoro branch, but it cannot fix the finite NS--R--R
three-point phase.  This script computes that missing factor without using
the proposed three-leg answer.

We specialize to the boundary family

    (n_1,n_2,n_3) = (0, 1/4, n),  n = 1/4, 3/4, 5/4.

The NS branch is its primary.  The n_2=1/4 Ramond branch has only ground
components.  The complete branch at n_3=n is imported from
check_ramond_branching.py, including every auxiliary-fermion sector and every
abstract Ramond PBW component.  Its physical three-point matrix elements are
then reduced by the triangular NS--R--R Ward recursion, while the auxiliary
fermion matrix elements are extracted from the two-spin-field kernel.

For each allowed (f,g,eta,epsilon_2,epsilon_3), the program evaluates

    kappa^2 = rho_hat^2 D_2 D_3 / (||v_2||^2 ||v_3||^2 Pcal^2),

where D_i is the known ell-product.  The direct calculation shows that the
four-factor numerator must be evaluated on the chiral-structure-dependent
Ramond sheet

    P_2 -> eta (-1)^(2 n_3 - 1/2) P_2.

Thus no choice of square root is needed for the primary certificate.  The
program proves, by exact arithmetic at two independent momentum samples,

    kappa^2 = eta (-1)^(epsilon_3 + 2 n_3 - 1/2) i^(1-f) / 2.

The sign of kappa itself is changed by changing the sign of any one branch
highest state, so kappa^2 is the convention-independent statement.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ramond_branching as branch  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2
EIGHTH_PLUS = (1 + I) / SQRT2


def add_term(out, key, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    out[key] = sp.cancel(out.get(key, 0) + coefficient)
    if out[key] == 0:
        del out[key]


def physical_ground(form_parity, eta, external_sign, internal_sign):
    """SCblock NS--R--R ground form in the displayed slot order."""

    if form_parity == 0:
        return {
            (0, 0): sp.Integer(1),
            (1, 1): sp.Integer(eta),
        }.get((external_sign, internal_sign), sp.Integer(0))
    return {
        (0, 1): sp.Integer(1),
        (1, 0): I * eta,
    }.get((external_sign, internal_sign), sp.Integer(0))


def physical_g0(beta, sign):
    """Coefficient and flipped sign in G_0 w^sign."""

    coefficient = I * beta * (EIGHTH_MINUS if sign == 0 else EIGHTH_PLUS)
    return coefficient, 1 - sign


def word_level(word):
    return sum(-mode for _, mode in word)


def word_parity(word, ground):
    return (sum(kind == "G" for kind, _ in word) + ground) % 2


class PhysicalNRREvaluator:
    """Triangular evaluator of rho_f^eta(NS,R_ground,R_descendant)."""

    def __init__(
        self,
        form_parity,
        eta,
        h_ns,
        h_external,
        h_internal,
        beta_external,
        beta_internal,
        central_charge,
    ):
        self.form_parity = int(form_parity)
        self.eta = int(eta)
        self.h_ns = h_ns
        self.h_external = h_external
        self.h_internal = h_internal
        self.beta_external = beta_external
        self.beta_internal = beta_internal
        self.central_charge = central_charge

    def _bracket(self, first, second):
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
                    sp.Rational(first_mode, 2) - second_mode,
                    (("G", first_mode + second_mode),),
                )
            )
        elif first_kind == "G" and second_kind == "L":
            answer.append(
                (
                    first_mode - sp.Rational(second_mode, 2),
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

    @lru_cache(None)
    def act_mode(self, kind, mode, word, ground):
        """Act one nonnegative mode on a negative-mode word."""

        if mode < 0:
            return {(((kind, mode),) + word, ground): sp.Integer(1)}
        if not word:
            if kind == "L":
                if mode == 0:
                    return {((), ground): self.h_internal}
                return {}
            if mode == 0:
                coefficient, flipped = physical_g0(self.beta_internal, ground)
                return {((), flipped): coefficient}
            return {}

        first = word[0]
        rest = word[1:]
        if first[1] >= 0:
            raise AssertionError("The descendant word is not negative-mode ordered.")

        out = {}
        exchange_sign = -1 if kind == first[0] == "G" else 1
        for (reduced_word, reduced_ground), coefficient in self.act_mode(
            kind, mode, rest, ground
        ).items():
            add_term(
                out,
                ((first,) + reduced_word, reduced_ground),
                exchange_sign * coefficient,
            )

        for bracket_coefficient, replacement in self._bracket(
            (kind, mode), first
        ):
            if not replacement:
                add_term(out, (rest, ground), bracket_coefficient)
                continue
            replacement_kind, replacement_mode = replacement[0]
            for key, coefficient in self.act_mode(
                replacement_kind, replacement_mode, rest, ground
            ).items():
                add_term(out, key, bracket_coefficient * coefficient)
        return out

    @lru_cache(None)
    def value(self, external_sign, word, internal_ground):
        if not word:
            return physical_ground(
                self.form_parity,
                self.eta,
                external_sign,
                internal_ground,
            )

        kind, mode = word[0]
        if mode >= 0:
            raise AssertionError("The Ward evaluator expects a negative-mode word.")
        rest = word[1:]

        if kind == "L":
            n = -mode
            coefficient = (
                self.h_internal
                + word_level(rest)
                + n * self.h_external
                - self.h_ns
            )
            return sp.cancel(
                coefficient * self.value(external_sign, rest, internal_ground)
            )

        m = -mode
        external_coefficient, flipped_external = physical_g0(
            self.beta_external, external_sign
        )
        answer = (
            external_coefficient
            * self.value(flipped_external, rest, internal_ground)
            / (I * (-1) ** word_parity(rest, internal_ground))
        )

        # A mode G_k with k larger than the level of rest annihilates it.
        for p in range(1, m + word_level(rest) + 2):
            acted = self.act_mode("G", -m + p, rest, internal_ground)
            if not acted:
                continue
            ward_coefficient = sp.binomial(sp.Rational(1, 2), p) * (-1) ** p
            for (reduced_word, reduced_ground), coefficient in acted.items():
                answer -= ward_coefficient * coefficient * self.value(
                    external_sign, reduced_word, reduced_ground
                )
        return sp.cancel(answer)

    def check_level_one_anchor(self):
        """Reproduce the first Ramond null-vector Ward polynomial."""

        if self.form_parity != 0:
            return
        value = physical_g0(self.beta_internal, 0)[0] * self.value(
            0, (("G", -1),), 1
        )
        expected = -self.beta_internal**2 / 2 - self.eta * (
            self.beta_internal * self.beta_external
        )
        difference = sp.factor(sp.simplify(value - expected))
        if difference != 0:
            raise AssertionError(f"Ramond level-one Ward anchor failed: {difference}")


def fermion_ground(form_parity, external_sign, internal_sign):
    """The canonical auxiliary matrices Gamma_0=K and Gamma_1=J."""

    if form_parity == 0:
        return {
            (0, 0): sp.Integer(1),
            (1, 1): sp.Integer(-1),
        }.get((external_sign, internal_sign), sp.Integer(0))
    return {
        (0, 1): sp.Integer(1),
        (1, 0): sp.Integer(-1),
    }.get((external_sign, internal_sign), sp.Integer(0))


def half_binomial_series(index):
    """Coefficient of z^index in (1-z)^(-1/2)."""

    return sp.binomial(2 * index, index) / 4**index


def positive_half_series(index):
    """Coefficient of z^index in (1-z)^(1/2)."""

    return sp.binomial(sp.Rational(1, 2), index) * (-1) ** index


def fermion_two_mode_coefficient(first_mode, second_mode):
    """Coefficient from the two-spin-field kernel for first_mode>second_mode."""

    if first_mode <= second_mode:
        raise ValueError("The strict Ramond modes must be in decreasing order.")
    answer = sp.Integer(0)
    for k in range(second_mode + 1):
        answer += (
            half_binomial_series(first_mode + k)
            * positive_half_series(second_mode - k)
            / 2
        )
    for k in range(second_mode):
        answer += (
            positive_half_series(first_mode + k + 1)
            * half_binomial_series(second_mode - 1 - k)
            / 2
        )
    return sp.factor(answer)


def fermion_value(form_parity, external_sign, modes, internal_ground):
    """rho_g^F(1,u_external,Psi_-modes u_internal) through level three."""

    if not modes:
        return fermion_ground(form_parity, external_sign, internal_ground)
    if len(modes) == 1:
        mode = modes[0]
        coefficient = half_binomial_series(mode) / SQRT2
        return coefficient * fermion_ground(
            form_parity, external_sign, 1 - internal_ground
        )
    if len(modes) == 2:
        return fermion_two_mode_coefficient(*modes) * fermion_ground(
            form_parity, external_sign, internal_ground
        )
    raise ValueError("Only auxiliary Ramond levels through three are required.")


def branch_components(branch_label, parity, substitutions):
    """Return branch components in the SCblock w^+,w^- ground basis."""

    _, sectors = branch.branch_in_abstract_basis(
        branch_label, parity, substitutions=substitutions
    )
    answer = []
    for (auxiliary_modes, auxiliary_ground), (_, ordered_basis, coefficients) in (
        sectors.items()
    ):
        for index, (virasoro_modes, supercurrent_modes, physical_ground) in enumerate(
            ordered_basis
        ):
            coefficient = coefficients[index]
            if coefficient == 0:
                continue
            # The transition code uses |Delta,->=-e^{-i*pi/4}w^-.
            if physical_ground == 1:
                coefficient *= -EIGHTH_MINUS
            word = tuple(("L", -mode) for mode in virasoro_modes) + tuple(
                ("G", -mode) for mode in supercurrent_modes
            )
            answer.append(
                (
                    auxiliary_modes,
                    auxiliary_ground,
                    word,
                    physical_ground,
                    sp.cancel(coefficient),
                )
            )
    return tuple(answer)


def enlarged_raw_three_point(
    spectator_label,
    spectator_parity,
    internal_label,
    internal_parity,
    form_parity,
    fermion_form_parity,
    eta,
    b_value,
    p_ns,
    p_external,
    p_internal,
):
    q_value = b_value + 1 / b_value
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h_ns = (q_value**2 / 4 - p_ns**2) / 2
    h_external = sp.Rational(1, 16) + q_value**2 / 8 - p_external**2 / 2
    h_internal = sp.Rational(1, 16) + q_value**2 / 8 - p_internal**2 / 2

    evaluator = PhysicalNRREvaluator(
        form_parity,
        eta,
        h_ns,
        h_external,
        h_internal,
        p_external / SQRT2,
        p_internal / SQRT2,
        central_charge,
    )
    evaluator.check_level_one_anchor()

    spectator = branch_components(
        spectator_label,
        spectator_parity,
        {branch.P: p_external, branch.Q: q_value},
    )
    internal = branch_components(
        internal_label,
        internal_parity,
        {branch.P: p_internal, branch.Q: q_value},
    )

    answer = sp.Integer(0)
    for (
        external_aux_modes,
        external_aux_ground,
        external_word,
        external_physical_ground,
        external_coefficient,
    ) in spectator:
        if external_aux_modes or external_word:
            raise AssertionError("The spectator n=1/4 branch must be at ground level.")
        for (
            internal_aux_modes,
            internal_aux_ground,
            internal_word,
            internal_physical_ground,
            internal_coefficient,
        ) in internal:
            auxiliary = fermion_value(
                fermion_form_parity,
                external_aux_ground,
                internal_aux_modes,
                internal_aux_ground,
            )
            if auxiliary == 0:
                continue
            physical = evaluator.value(
                external_physical_ground,
                internal_word,
                internal_physical_ground,
            )
            tensor_sign = (-1) ** (
                external_physical_ground
                * ((len(internal_aux_modes) + internal_aux_ground) % 2)
            )
            answer += (
                tensor_sign
                * external_coefficient
                * internal_coefficient
                * auxiliary
                * physical
            )
    return sp.factor(sp.cancel(answer))


def ell(x, index, b_value):
    index = int(index)
    q_value = b_value + 1 / b_value
    if index == 0:
        return sp.Integer(1)
    if index < 0:
        if index % 2 == 0:
            return (-1) ** (-index // 2) * ell(q_value - x, -index, b_value)
        return ell(q_value - x, -index, b_value)

    answer = sp.Pow(2, sp.Rational(1, 8)) if index % 2 else sp.Integer(1)
    required_parity = index % 2
    for r in range(index):
        for s in range(index - r):
            if (r + s) % 2 == required_parity:
                answer *= x + r * b_value + s / b_value
    return sp.factor(answer)


def leg_product(momentum, branch_label, b_value):
    index = int(4 * sp.Rational(branch_label))
    q_value = b_value + 1 / b_value
    return sp.factor(
        ell(2 * momentum, index, b_value)
        * ell(q_value + 2 * momentum, index, b_value)
    )


def numerator_product(n1, n2, n3, p1, p2, p3, b_value):
    q_value = b_value + 1 / b_value
    data = (
        (q_value / 2 + p1 + p2 + p3, 2 * (n1 + n2 + n3)),
        (q_value / 2 - p1 + p2 + p3, 2 * (-n1 + n2 + n3)),
        (q_value / 2 + p1 - p2 + p3, 2 * (n1 - n2 + n3)),
        (q_value / 2 + p1 + p2 - p3, 2 * (n1 + n2 - n3)),
    )
    answer = sp.Integer(1)
    for argument, index in data:
        if not sp.Rational(index).is_integer:
            raise AssertionError(f"Nonintegral ell index {index}")
        answer *= ell(argument, int(index), b_value)
    return sp.factor(answer)


def oriented_second_momentum(eta, internal_label, p_external):
    """Ramond momentum in the numerator for the family (0,1/4,n)."""

    mode_count = 2 * sp.Rational(internal_label) - sp.Rational(1, 2)
    if not mode_count.is_integer or mode_count < 0:
        raise ValueError("This checker uses positive labels n=1/4,3/4,5/4.")
    return int(eta) * (-1) ** int(mode_count) * p_external


def expected_kappa_squared(internal_label, internal_parity, form_parity, eta):
    """Closed discrete factor in the raw-state and BPZ conventions above."""

    mode_count = int(
        2 * sp.Rational(internal_label) - sp.Rational(1, 2)
    )
    return (
        sp.Rational(1, 2)
        * int(eta)
        * (-1) ** (int(internal_parity) + mode_count)
        * I ** (1 - int(form_parity))
    )


def branch_norm_at(branch_label, parity, b_value, momentum):
    substitutions = {
        branch.Q: b_value + 1 / b_value,
        branch.P: momentum,
    }
    return branch.branch_norm(
        branch_label, parity, substitutions=substitutions
    )[3]


def kappa_certificate(
    internal_label,
    spectator_parity,
    internal_parity,
    form_parity,
    eta,
    b_value,
    p_ns,
    p_external,
    p_internal,
):
    spectator_label = sp.Rational(1, 4)
    fermion_form_parity = (
        spectator_parity + internal_parity - form_parity
    ) % 2
    raw = enlarged_raw_three_point(
        spectator_label,
        spectator_parity,
        internal_label,
        internal_parity,
        form_parity,
        fermion_form_parity,
        eta,
        b_value,
        p_ns,
        p_external,
        p_internal,
    )
    spectator_norm = branch_norm_at(
        spectator_label, spectator_parity, b_value, p_external
    )
    internal_norm = branch_norm_at(
        internal_label, internal_parity, b_value, p_internal
    )
    d_external = leg_product(p_external, spectator_label, b_value)
    d_internal = leg_product(p_internal, internal_label, b_value)
    effective_external_momentum = oriented_second_momentum(
        eta, internal_label, p_external
    )
    pcal = numerator_product(
        sp.Integer(0),
        spectator_label,
        internal_label,
        p_ns,
        effective_external_momentum,
        p_internal,
        b_value,
    )
    kappa_squared = sp.factor(
        sp.cancel(
            raw**2
            * d_external
            * d_internal
            / (spectator_norm * internal_norm * pcal**2)
        )
    )
    return {
        "g": fermion_form_parity,
        "raw": raw,
        "norm2": spectator_norm,
        "norm3": internal_norm,
        "pcal": pcal,
        "kappa_squared": kappa_squared,
    }


def run_samples():
    b_symbol, p1_symbol, p2_symbol, p3_symbol = sp.symbols(
        "b p_1 p_2 p_3", nonzero=True
    )
    exact_samples = (
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
    labels = (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4))
    reports = []
    for internal_label in labels:
        samples = (
            ((b_symbol, p1_symbol, p2_symbol, p3_symbol),)
            if internal_label < sp.Rational(5, 4)
            else exact_samples
        )
        for spectator_parity in (0, 1):
            for internal_parity in (0, 1):
                for form_parity in (0, 1):
                    for eta in (1, -1):
                        values = []
                        for sample in samples:
                            values.append(
                                kappa_certificate(
                                    internal_label,
                                    spectator_parity,
                                    internal_parity,
                                    form_parity,
                                    eta,
                                    *sample,
                                )
                            )
                        expected = expected_kappa_squared(
                            internal_label, internal_parity, form_parity, eta
                        )
                        differences = tuple(
                            sp.factor(
                                sp.cancel(value["kappa_squared"] - expected)
                            )
                            for value in values
                        )
                        if any(difference != 0 for difference in differences):
                            raise AssertionError(
                                "kappa^2 does not equal the discrete formula for "
                                f"n={internal_label}, eps=({spectator_parity},"
                                f"{internal_parity}), f={form_parity}, eta={eta}: "
                                f"{differences}"
                            )
                        reports.append(
                            (
                                internal_label,
                                spectator_parity,
                                internal_parity,
                                form_parity,
                                values[0]["g"],
                                eta,
                                values[0]["kappa_squared"],
                            )
                        )
    return reports


def main():
    coefficient = fermion_two_mode_coefficient(2, 1)
    if coefficient != sp.Rational(1, 32):
        raise AssertionError(f"Unexpected two-fermion coefficient: {coefficient}")
    print("auxiliary two-spin kernel coefficient [psi_-2 psi_-1] = 1/32")

    reports = run_samples()
    for n in (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4)):
        selected = [report for report in reports if report[0] == n]
        values = sorted({str(report[-1]) for report in selected})
        method = (
            "symbolically in b,P_1,P_2,P_3"
            if n < sp.Rational(5, 4)
            else "at two independent exact momentum samples"
        )
        print(
            f"n={n}: checked {len(selected)} discrete states {method}; "
            f"kappa^2 in {{{', '.join(values)}}}"
        )
    print(
        "All 48 discrete states satisfy "
        "kappa^2=eta*(-1)^(epsilon_3+2*n-1/2)*i^(1-f)/2."
    )


if __name__ == "__main__":
    main()
