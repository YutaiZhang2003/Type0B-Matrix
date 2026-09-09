"""Exact Ramond PBW and generalized NS--R--R Ward computations.

This module follows Section 5 of ``Human Notes/SCblock.tex``:

* ``G_0 w^+ = i beta exp(-i pi/4) w^-`` and
  ``G_0 w^- = i beta exp(+i pi/4) w^+``;
* the Ramond PBW basis excludes ``G_0``;
* all four component ground three-forms in (5.3) are normalized to one;
* the degeneracy, inverse-norm, and fusion formulas are (5.6)--(5.10).

There is deliberately no fixed-even-primary Ward implementation here.  The
intrinsic NS-primary parity ``p_phi`` is mandatory, and the generalized Ward
sign is always computed from the full states,

    epsilon(xi, eta) = -i (-1)^(|xi| + |eta| + 1),
    |xi| = p_phi + number of G modes in xi (mod 2).

The Ward recursion is the literal plane NS--R--R contour system recorded in
``Machine Notes/c-Recursion/ramond_channel_c_recursion.tex``.  Keeping it
literal is useful: the audit can expose, instead of hiding, any remaining
local-coordinate sign between that system and the factorization printed in
the human note.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

import sympy as sp
from sympy.polys.matrices import DomainMatrix


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_PLUS = (1 + I) / SQRT2

Mode = tuple[str, sp.Expr]
Word = tuple[Mode, ...]


def clean(value: sp.Expr) -> sp.Expr:
    """Return a stable exact rational form."""

    return sp.factor(sp.cancel(sp.together(value)))


def add_term(output: dict, key, coefficient: sp.Expr) -> None:
    coefficient = clean(coefficient)
    if coefficient == 0:
        return
    output[key] = clean(output.get(key, sp.S.Zero) + coefficient)
    if output[key] == 0:
        del output[key]


def word_level(word: Word) -> sp.Expr:
    return sum((-mode for _, mode in word), sp.S.Zero)


def word_parity(word: Word) -> int:
    return sum(kind == "G" for kind, _ in word) % 2


def state_parity(word: Word, ground: int, primary_parity: int = 0) -> int:
    return (int(primary_parity) + word_parity(word) + int(ground)) % 2


def _integer_partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    for first in range(maximum, 0, -1):
        for rest in _integer_partitions(total - first, first):
            yield (first,) + rest


def _strict_partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    for first in range(maximum, 0, -1):
        for rest in _strict_partitions(total - first, first - 1):
            yield (first,) + rest


@dataclass(frozen=True)
class RamondState:
    """A Ramond PBW word acting on ``w^+`` (ground=0) or ``w^-`` (1)."""

    word: Word
    ground: int

    @property
    def level(self) -> int:
        return int(word_level(self.word))

    @property
    def parity(self) -> int:
        return state_parity(self.word, self.ground)

    def label(self) -> str:
        suffix = "+" if self.ground == 0 else "-"
        if not self.word:
            return f"w^{suffix}"
        modes = " ".join(f"{kind}_{mode}" for kind, mode in self.word)
        return f"{modes} w^{suffix}"


class SuperVirasoroModule:
    """Exact NS or Ramond highest-weight module mode action."""

    def __init__(
        self,
        weight: sp.Expr,
        central_charge: sp.Expr,
        *,
        sector: str,
        beta: sp.Expr | None = None,
    ) -> None:
        if sector not in {"NS", "R"}:
            raise ValueError("sector must be 'NS' or 'R'")
        if sector == "R" and beta is None:
            raise ValueError("a Ramond module requires beta")
        self.weight = sp.sympify(weight)
        self.central_charge = sp.sympify(central_charge)
        self.sector = sector
        self.beta = None if beta is None else sp.sympify(beta)

    def bracket(self, first: Mode, second: Mode):
        """The supercommutator of two N=1 modes."""

        first_kind, first_mode = first
        second_kind, second_mode = second
        answer: list[tuple[sp.Expr, Word]] = []
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
    def _canonical_key(operator: Mode):
        kind, mode = operator
        return (0 if kind == "L" else 1, mode)

    @lru_cache(None)
    def canonicalize(self, word: Word) -> dict[Word, sp.Expr]:
        """Put negative modes in the human-note L-before-G PBW order."""

        if any(mode >= 0 for _, mode in word):
            raise ValueError(f"canonicalize received a nonnegative mode: {word}")
        for position in range(len(word) - 1):
            first, second = word[position], word[position + 1]
            if first == second and first[0] == "G":
                reduced = word[:position] + (("L", 2 * first[1]),) + word[position + 2 :]
                return self.canonicalize(reduced)
            if self._canonical_key(first) <= self._canonical_key(second):
                continue
            output: dict[Word, sp.Expr] = {}
            exchange_sign = -1 if first[0] == second[0] == "G" else 1
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            for canonical, coefficient in self.canonicalize(exchanged).items():
                add_term(output, canonical, exchange_sign * coefficient)
            for bracket_coefficient, replacement in self.bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                for canonical, coefficient in self.canonicalize(reduced).items():
                    add_term(output, canonical, bracket_coefficient * coefficient)
            return output
        return {word: sp.S.One}

    def g0_action(self, ground: int) -> tuple[sp.Expr, int]:
        """Equation (5.1), with ground 0/1 denoting w^+/w^-."""

        if self.sector != "R":
            raise ValueError("G_0 exists only in the Ramond module")
        if ground not in (0, 1):
            raise ValueError("ground must be 0 (w+) or 1 (w-)")
        if ground == 0:
            return self.beta * EIGHTH_PLUS, 1
        return I * self.beta * EIGHTH_PLUS, 0

    @lru_cache(None)
    def act(
        self, kind: str, mode: sp.Expr, word: Word, ground: int
    ) -> dict[tuple[Word, int], sp.Expr]:
        """Act one mode on a canonical negative-mode descendant."""

        mode = sp.sympify(mode)
        if mode < 0:
            output: dict[tuple[Word, int], sp.Expr] = {}
            for canonical, coefficient in self.canonicalize(((kind, mode),) + word).items():
                add_term(output, (canonical, ground), coefficient)
            return output

        if not word:
            if kind == "L":
                if mode == 0:
                    return {((), ground): self.weight}
                return {}
            if mode == 0:
                coefficient, flipped = self.g0_action(ground)
                return {((), flipped): coefficient}
            return {}

        first = word[0]
        rest = word[1:]
        output: dict[tuple[Word, int], sp.Expr] = {}
        exchange_sign = -1 if kind == first[0] == "G" else 1

        for (reduced_word, reduced_ground), coefficient in self.act(
            kind, mode, rest, ground
        ).items():
            for canonical, canonical_coefficient in self.canonicalize(
                (first,) + reduced_word
            ).items():
                add_term(
                    output,
                    (canonical, reduced_ground),
                    exchange_sign * coefficient * canonical_coefficient,
                )

        for bracket_coefficient, replacement in self.bracket((kind, mode), first):
            if not replacement:
                add_term(output, (rest, ground), bracket_coefficient)
                continue
            replacement_kind, replacement_mode = replacement[0]
            for key, coefficient in self.act(
                replacement_kind, replacement_mode, rest, ground
            ).items():
                add_term(output, key, bracket_coefficient * coefficient)
        return output


class RamondPBWModule(SuperVirasoroModule):
    """Ramond PBW bases and Gram matrices in the physical w^+/w^- basis."""

    def __init__(self, weight: sp.Expr, beta: sp.Expr, central_charge: sp.Expr):
        super().__init__(weight, central_charge, sector="R", beta=beta)

    @staticmethod
    @lru_cache(None)
    def basis(level: int, parity: int | None = None) -> tuple[RamondState, ...]:
        if isinstance(level, bool) or int(level) != level or level < 0:
            raise ValueError("Ramond level must be a nonnegative integer")
        level = int(level)
        if parity not in (None, 0, 1):
            raise ValueError("parity must be None, 0, or 1")
        states: list[RamondState] = []
        for fermion_level in range(level + 1):
            for fermions in _strict_partitions(fermion_level):
                for bosons in _integer_partitions(level - fermion_level):
                    word: Word = tuple(("L", -n) for n in bosons) + tuple(
                        ("G", -m) for m in fermions
                    )
                    for ground in (0, 1):
                        state = RamondState(word, ground)
                        if parity is None or state.parity == parity:
                            states.append(state)
        states.sort(
            key=lambda state: (
                state.ground,
                tuple((0 if kind == "L" else 1, mode) for kind, mode in state.word),
            )
        )
        return tuple(states)

    @staticmethod
    def ground_pairing(left: int, right: int) -> sp.Expr:
        """Bilinear BPZ ground metric implied by self-adjoint G_0."""

        if left != right:
            return sp.S.Zero
        return sp.S.One if left == 0 else I

    def inner_product(self, left: RamondState, right: RamondState) -> sp.Expr:
        if left.level != right.level or left.parity != right.parity:
            return sp.S.Zero
        states: dict[tuple[Word, int], sp.Expr] = {
            (right.word, right.ground): sp.S.One
        }
        # If left.word=A_1 ... A_k, then its adjoint is
        # A_k^dagger ... A_1^dagger; A_1^dagger acts first on the ket.
        for kind, mode in left.word:
            acted: dict[tuple[Word, int], sp.Expr] = {}
            for (word, ground), coefficient in states.items():
                for key, action_coefficient in self.act(kind, -mode, word, ground).items():
                    add_term(acted, key, coefficient * action_coefficient)
            states = acted
        answer = sp.S.Zero
        for (word, ground), coefficient in states.items():
            if not word:
                answer += coefficient * self.ground_pairing(left.ground, ground)
        return clean(answer)

    @lru_cache(None)
    def gram_matrix(
        self, level: int, parity: int
    ) -> tuple[tuple[RamondState, ...], sp.Matrix]:
        basis = self.basis(level, parity)
        matrix = sp.Matrix(
            [
                [self.inner_product(left, right) for right in basis]
                for left in basis
            ]
        )
        return basis, matrix.applyfunc(clean)

    @lru_cache(None)
    def gram_kernel(
        self, level: int, parity: int
    ) -> tuple[tuple[RamondState, ...], tuple[sp.Matrix, ...]]:
        """Return an exact kernel using fraction-free polynomial-domain RREF.

        SymPy's expression-level ``Matrix.nullspace`` becomes unnecessarily
        expensive from level three onward.  ``DomainMatrix`` performs the
        same calculation over the exact algebraic rational-function domain
        (for example ``QQ_I(b)``) and keeps the higher-level certificate
        practical.
        """

        basis, gram = self.gram_matrix(level, parity)
        rows = DomainMatrix.from_Matrix(gram, extension=True).nullspace().to_Matrix()
        return basis, tuple(rows.row(index).T for index in range(rows.rows))

    def normalized_null_vector(
        self, level: int, ground: int
    ) -> dict[RamondState, sp.Expr]:
        """Return chi^+ or chi^- with coefficient of L_-1^level w^± equal to 1."""

        basis, kernel = self.gram_kernel(level, ground)
        if len(kernel) != 1:
            raise AssertionError(
                f"expected one null vector in parity {ground}, got {len(kernel)}"
            )
        target = RamondState(tuple(("L", -1) for _ in range(level)), ground)
        target_index = basis.index(target)
        leading = clean(kernel[0][target_index, 0])
        if leading == 0:
            raise AssertionError("null vector has zero L_-1 leading coefficient")
        return {
            state: clean(kernel[0][index, 0] / leading)
            for index, state in enumerate(basis)
            if kernel[0][index, 0] != 0
        }


def central_charge_from_b(b: sp.Expr) -> sp.Expr:
    return clean(sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2)


def x_rs_from_beta(r: int, s: int, beta: sp.Expr) -> sp.Expr:
    """The selected algebraic branch ``x_rs^R(beta)`` in equation (5.7)."""

    return clean(
        (
            4 * beta**2
            - r * s
            + 2 * SQRT2 * beta * sp.sqrt(2 * beta**2 - r * s)
        )
        / r**2
    )


def c_rs_from_beta(r: int, s: int, beta: sp.Expr) -> sp.Expr:
    """The fixed-beta pole ``c_rs^R(beta)`` in equation (5.7)."""

    x = x_rs_from_beta(r, s, beta)
    return clean(sp.Rational(15, 2) + 3 * (x + 1 / x))


def ramond_degenerate_data(r: int, s: int, b: sp.Expr) -> dict[str, sp.Expr]:
    """Equations (5.6)--(5.8) on the b branch."""

    if r <= 0 or s <= 0 or (r + s) % 2 != 1:
        raise ValueError("Ramond labels require r,s>0 and r+s odd")
    beta = clean((r * b + s / b) / (2 * SQRT2))
    central_charge = central_charge_from_b(b)
    weight = clean(central_charge / 24 - beta**2)
    beta_shifted = clean(((-1) ** s) * (r * b - s / b) / (2 * SQRT2))
    return {
        "b": b,
        "x": b**2,
        "c": central_charge,
        "beta": beta,
        "h": weight,
        "level": sp.Rational(r * s, 2),
        "beta_shifted": beta_shifted,
        "h_shifted": clean(central_charge / 24 - beta_shifted**2),
    }


def ramond_labels_at_level(level: int) -> tuple[tuple[int, int], ...]:
    """All positive Ramond Kac labels satisfying ``rs/2 = level``."""

    if isinstance(level, bool) or int(level) != level or level <= 0:
        raise ValueError("level must be a positive integer")
    product = 2 * int(level)
    labels = []
    for r in range(1, product + 1):
        if product % r:
            continue
        s = product // r
        if (r + s) % 2 == 1:
            labels.append((r, s))
    return tuple(labels)


@lru_cache(None)
def _cached_degenerate_null_items(
    r: int, s: int, b: sp.Expr, ground: int
) -> tuple[tuple[RamondState, sp.Expr], ...]:
    data = ramond_degenerate_data(r, s, b)
    module = RamondPBWModule(data["h"], data["beta"], data["c"])
    vector = module.normalized_null_vector(int(data["level"]), ground)
    return tuple(vector.items())


def degenerate_null_vector(
    r: int, s: int, b: sp.Expr, ground: int
) -> dict[RamondState, sp.Expr]:
    """Return a cached normalized chi_rs^+ or chi_rs^- PBW vector."""

    if ground not in (0, 1):
        raise ValueError("ground must be 0 (chi+) or 1 (chi-)")
    return dict(_cached_degenerate_null_items(r, s, sp.sympify(b), ground))


def pole_equation_residual(r: int, s: int, b: sp.Expr) -> sp.Expr:
    data = ramond_degenerate_data(r, s, b)
    x = data["x"]
    beta = data["beta"]
    return clean(r**2 * x**2 + (2 * r * s - 8 * beta**2) * x + s**2)


def fusion_polynomial_510(
    r: int,
    s: int,
    lambda_i: sp.Expr,
    beta_j: sp.Expr,
    b: sp.Expr,
    eta: int,
) -> sp.Expr:
    """The Ramond fusion polynomial printed in equation (5.10)."""

    if eta not in (-1, 1):
        raise ValueError("eta must be +1 or -1")
    answer = sp.S.One
    for k in range(r):
        for ell in range(s):
            lattice = (1 - r + 2 * k) * b + (1 - s + 2 * ell) / b
            beta_term = 2 * SQRT2 * eta * beta_j
            if (k + ell) % 2 == 0:
                factor = (lambda_i - beta_term - lattice) / (2 * SQRT2)
            else:
                factor = (lambda_i + beta_term - lattice) / (2 * SQRT2)
            answer *= factor
    return clean(answer)


def inverse_null_product_59(
    r: int, s: int, b: sp.Expr, *, lattice: str = "literal"
) -> sp.Expr:
    """Evaluate (5.9), with its printed lattice ambiguity made explicit.

    ``literal`` implements the displayed condition ``p+q in Z`` (all integer
    pairs).  ``even`` and ``odd`` restrict to the corresponding parity
    sublattice.  The direct PBW calculation decides which one agrees.
    """

    if lattice not in {"literal", "even", "odd"}:
        raise ValueError("lattice must be 'literal', 'even', or 'odd'")
    denominator = sp.S.One
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p, q) == (0, 0):
                continue
            if lattice == "even" and (p + q) % 2:
                continue
            if lattice == "odd" and (p + q) % 2 == 0:
                continue
            denominator *= p * b + q / b
    return clean(2 ** (r * s - 2) * (r * b + s / b) / denominator)


class GeneralizedNRRWard:
    """Literal generalized plane Ward recursion for rho_f^eta(NS,R,R)."""

    def __init__(
        self,
        *,
        p_phi: int,
        form_parity: int,
        eta: int,
        h_ns: sp.Expr,
        h_second: sp.Expr,
        h_third: sp.Expr,
        beta_second: sp.Expr,
        beta_third: sp.Expr,
        central_charge: sp.Expr,
    ) -> None:
        if p_phi not in (0, 1):
            raise ValueError("p_phi is mandatory and must be 0 or 1")
        if form_parity not in (0, 1):
            raise ValueError("form_parity must be 0 or 1")
        if eta not in (-1, 1):
            raise ValueError("eta must be +1 or -1")
        self.p_phi = int(p_phi)
        self.form_parity = int(form_parity)
        self.eta = int(eta)
        self.modules = (
            SuperVirasoroModule(h_ns, central_charge, sector="NS"),
            SuperVirasoroModule(
                h_second, central_charge, sector="R", beta=beta_second
            ),
            SuperVirasoroModule(
                h_third, central_charge, sector="R", beta=beta_third
            ),
        )

    @staticmethod
    def component_normalization(second_ground: int, third_ground: int) -> sp.Expr:
        """Each of rho^{++}, rho^{+-}, rho^{-+}, rho^{--} in (5.3)."""

        if second_ground not in (0, 1) or third_ground not in (0, 1):
            raise ValueError("ground labels must be 0 or 1")
        return sp.S.One

    def ground_value(self, second_ground: int, third_ground: int) -> sp.Expr:
        """Build rho_f^eta from the four unit-normalized components (5.3)."""

        total = (self.p_phi + second_ground + third_ground) % 2
        if total != self.form_parity:
            return sp.S.Zero
        pair = (second_ground, third_ground)
        return {
            (0, 0): sp.S.One,
            (1, 1): sp.Integer(self.eta),
            (0, 1): sp.S.One,
            (1, 0): I * self.eta,
        }[pair]

    def epsilon(self, word1: Word, word3: Word, ground3: int) -> sp.Expr:
        """The generalized sign with the intrinsic primary parity included."""

        xi_parity = state_parity(word1, 0, self.p_phi)
        eta_parity = state_parity(word3, ground3)
        return -I * (-1) ** (xi_parity + eta_parity + 1)

    def _state_sum(
        self,
        word1: Word,
        states2: Mapping[tuple[Word, int], sp.Expr],
        states3: Mapping[tuple[Word, int], sp.Expr],
    ) -> sp.Expr:
        answer = sp.S.Zero
        for (word2, ground2), coefficient2 in states2.items():
            for (word3, ground3), coefficient3 in states3.items():
                answer += coefficient2 * coefficient3 * self.value(
                    word1, word2, ground2, word3, ground3
                )
        return clean(answer)

    def _first_state_sum(
        self,
        states1: Mapping[tuple[Word, int], sp.Expr],
        word2: Word,
        ground2: int,
        word3: Word,
        ground3: int,
    ) -> sp.Expr:
        answer = sp.S.Zero
        for (word1, _), coefficient in states1.items():
            answer += coefficient * self.value(
                word1, word2, ground2, word3, ground3
            )
        return clean(answer)

    @staticmethod
    def _cutoff(*levels: sp.Expr) -> int:
        return int(max([sp.S.Zero, *levels])) + 4

    @lru_cache(None)
    def value(
        self,
        word1: Word,
        word2: Word,
        ground2: int,
        word3: Word,
        ground3: int,
    ) -> sp.Expr:
        """Evaluate the fixed-form three-point function at z=1."""

        # Solve the first generalized Ward identity for an outer NS G_-r.
        if word1 and word1[0][0] == "G":
            _, mode = word1[0]
            rest1 = word1[1:]
            r = -mode
            epsilon = self.epsilon(rest1, word3, ground3)
            answer = sp.S.Zero
            maximum = self._cutoff(
                r + word_level(rest1), word_level(word2), word_level(word3)
            )
            for p in range(maximum + 1):
                acted2 = self.modules[1].act("G", p, word2, ground2)
                answer += sp.binomial(r, p) * self._state_sum(
                    rest1, acted2, {(word3, ground3): sp.S.One}
                )
                if p:
                    acted1 = self.modules[0].act("G", p - r, rest1, 0)
                    answer -= (
                        sp.binomial(sp.Rational(1, 2), p)
                        * (-1) ** p
                        * self._first_state_sum(
                            acted1, word2, ground2, word3, ground3
                        )
                    )
                acted3 = self.modules[2].act(
                    "G", r - sp.Rational(1, 2) + p, word3, ground3
                )
                answer -= (
                    epsilon
                    * sp.binomial(sp.Rational(1, 2), p)
                    * (-1) ** p
                    * self._state_sum(
                        rest1, {(word2, ground2): sp.S.One}, acted3
                    )
                )
            return clean(answer)

        # Remove the descendant at the middle puncture first.
        if word2:
            kind, mode = word2[0]
            rest2 = word2[1:]
            if kind == "L":
                n = int(-mode)
                if n == 1:
                    exponent = (
                        self.modules[0].weight
                        + word_level(word1)
                        - self.modules[1].weight
                        - word_level(rest2)
                        - self.modules[2].weight
                        - word_level(word3)
                    )
                    return clean(
                        exponent
                        * self.value(word1, rest2, ground2, word3, ground3)
                    )
                answer = sp.S.Zero
                maximum = self._cutoff(
                    word_level(word1) - n, word_level(word3) + 1
                )
                for p in range(maximum + 1):
                    ward = sp.binomial(n - 2 + p, n - 2)
                    acted1 = self.modules[0].act("L", n + p, word1, 0)
                    answer += ward * self._first_state_sum(
                        acted1, rest2, ground2, word3, ground3
                    )
                    acted3 = self.modules[2].act("L", p - 1, word3, ground3)
                    answer += ward * (-1) ** n * self._state_sum(
                        word1, {(rest2, ground2): sp.S.One}, acted3
                    )
                return clean(answer)

            # Solve the second generalized Ward identity for G_-n in slot 2.
            n = int(-mode)
            epsilon = self.epsilon(word1, word3, ground3)
            answer = sp.S.Zero
            maximum = self._cutoff(
                n + word_level(rest2),
                word_level(word1) - n + sp.Rational(1, 2),
                word_level(word3),
            )
            for p in range(maximum + 1):
                ward = sp.binomial(sp.Rational(1, 2) - n, p)
                acted1 = self.modules[0].act(
                    "G", sp.Rational(2 * p + 2 * n - 1, 2), word1, 0
                )
                answer += ward * (-1) ** p * self._first_state_sum(
                    acted1, rest2, ground2, word3, ground3
                )
                acted3 = self.modules[2].act("G", p, word3, ground3)
                answer += epsilon * ward * (-1) ** (n + p) * self._state_sum(
                    word1, {(rest2, ground2): sp.S.One}, acted3
                )
                if p:
                    acted2 = self.modules[1].act("G", p - n, rest2, ground2)
                    answer -= sp.binomial(sp.Rational(1, 2), p) * self._state_sum(
                        word1, acted2, {(word3, ground3): sp.S.One}
                    )
            return clean(answer)

        # Remove a BPZ descendant at infinity.
        if word1:
            kind, mode = word1[0]
            rest1 = word1[1:]
            if kind == "L":
                n = int(-mode)
                answer = self._state_sum(
                    rest1,
                    {((), ground2): sp.S.One},
                    self.modules[2].act("L", n, word3, ground3),
                )
                for m in range(-1, n + 1):
                    ward = sp.binomial(n + 1, m + 1)
                    answer += ward * self._state_sum(
                        rest1,
                        self.modules[1].act("L", m, (), ground2),
                        {(word3, ground3): sp.S.One},
                    )
                return clean(answer)
            raise AssertionError("outer G descendant should have been removed first")

        # Finally solve the first generalized identity for G_-m in slot 3.
        if word3:
            kind, mode = word3[0]
            rest3 = word3[1:]
            if kind == "L":
                n = int(-mode)
                coefficient = (
                    self.modules[2].weight
                    + word_level(rest3)
                    + n * self.modules[1].weight
                    - self.modules[0].weight
                )
                return clean(
                    coefficient * self.value((), (), ground2, rest3, ground3)
                )

            m = int(-mode)
            epsilon = self.epsilon((), rest3, ground3)
            answer = sp.S.Zero
            maximum = self._cutoff(m + word_level(rest3))
            for p in range(maximum + 1):
                acted2 = self.modules[1].act("G", p, (), ground2)
                answer += (
                    self._state_sum((), acted2, {(rest3, ground3): sp.S.One})
                    * sp.binomial(sp.Rational(1, 2) - m, p)
                    / epsilon
                )
                acted1 = self.modules[0].act(
                    "G", sp.Rational(2 * m - 1, 2) + p, (), 0
                )
                answer -= (
                    (-1) ** p
                    * sp.binomial(sp.Rational(1, 2), p)
                    * self._first_state_sum(
                        acted1, (), ground2, rest3, ground3
                    )
                    / epsilon
                )
                if p:
                    acted3 = self.modules[2].act("G", -m + p, rest3, ground3)
                    answer -= (
                        sp.binomial(sp.Rational(1, 2), p)
                        * (-1) ** p
                        * self._state_sum(
                            (), {((), ground2): sp.S.One}, acted3
                        )
                    )
            return clean(answer)

        return self.ground_value(ground2, ground3)

    def vector_value(
        self,
        second_vector: Mapping[RamondState, sp.Expr],
        *,
        word1: Word = (),
        word3: Word = (),
        ground3: int = 0,
    ) -> sp.Expr:
        answer = sp.S.Zero
        for state, coefficient in second_vector.items():
            answer += coefficient * self.value(
                word1, state.word, state.ground, word3, ground3
            )
        return clean(answer)

    def ward_first_residual(
        self,
        n: int,
        word1: Word,
        word2: Word,
        ground2: int,
        word3: Word,
        ground3: int,
    ) -> sp.Expr:
        """Residual of the first displayed generalized Ramond Ward identity."""

        epsilon = self.epsilon(word1, word3, ground3)
        maximum = self._cutoff(
            abs(n) + word_level(word1),
            abs(n) + word_level(word2),
            abs(n) + word_level(word3),
        )
        left = sp.S.Zero
        right = sp.S.Zero
        for p in range(maximum + 1):
            left += sp.binomial(n + sp.Rational(1, 2), p) * self._state_sum(
                word1,
                self.modules[1].act("G", p, word2, ground2),
                {(word3, ground3): sp.S.One},
            )
            right += (
                sp.binomial(sp.Rational(1, 2), p)
                * (-1) ** p
                * self._first_state_sum(
                    self.modules[0].act(
                        "G", p - n - sp.Rational(1, 2), word1, 0
                    ),
                    word2,
                    ground2,
                    word3,
                    ground3,
                )
            )
            right += (
                epsilon
                * sp.binomial(sp.Rational(1, 2), p)
                * (-1) ** p
                * self._state_sum(
                    word1,
                    {(word2, ground2): sp.S.One},
                    self.modules[2].act("G", n + p, word3, ground3),
                )
            )
        return clean(left - right)

    def ward_second_residual(
        self,
        n: int,
        word1: Word,
        word2: Word,
        ground2: int,
        word3: Word,
        ground3: int,
    ) -> sp.Expr:
        """Residual of the second displayed generalized Ramond Ward identity."""

        epsilon = self.epsilon(word1, word3, ground3)
        maximum = self._cutoff(
            abs(n) + word_level(word1),
            abs(n) + word_level(word2),
            abs(n) + word_level(word3),
        )
        left = sp.S.Zero
        right = sp.S.Zero
        for p in range(maximum + 1):
            left += sp.binomial(sp.Rational(1, 2), p) * self._state_sum(
                word1,
                self.modules[1].act("G", p - n, word2, ground2),
                {(word3, ground3): sp.S.One},
            )
            ward = sp.binomial(sp.Rational(1, 2) - n, p)
            right += ward * (-1) ** p * self._first_state_sum(
                self.modules[0].act(
                    "G", p + n - sp.Rational(1, 2), word1, 0
                ),
                word2,
                ground2,
                word3,
                ground3,
            )
            right += (
                epsilon
                * ward
                * (-1) ** (n + p)
                * self._state_sum(
                    word1,
                    {(word2, ground2): sp.S.One},
                    self.modules[2].act("G", p, word3, ground3),
                )
            )
        return clean(left - right)


def fixed_beta_inverse_null_norm(
    r: int,
    s: int,
    b: sp.Symbol,
    null_vector: Mapping[RamondState, sp.Expr],
) -> sp.Expr:
    """Compute the defining limit in (5.9) by differentiating at fixed beta."""

    pole = ramond_degenerate_data(r, s, b)
    parities = {state.parity for state, coefficient in null_vector.items() if coefficient != 0}
    if len(parities) != 1:
        raise ValueError("the null vector must have one definite Ramond parity")
    parity = parities.pop()
    central_charge_variable = sp.Dummy("central_charge")
    beta_fixed = pole["beta"]
    weight_variable = central_charge_variable / 24 - beta_fixed**2
    module_variable = RamondPBWModule(
        weight_variable, beta_fixed, central_charge_variable
    )
    basis = module_variable.basis(int(pole["level"]), parity)
    coefficients = sp.Matrix(
        [null_vector.get(state, sp.S.Zero) for state in basis]
    )
    gram = module_variable.gram_matrix(int(pole["level"]), parity)[1]
    norm = clean((coefficients.T * gram * coefficients)[0])
    norm_slope = sp.diff(norm, central_charge_variable).subs(
        central_charge_variable, pole["c"]
    )

    b_variable = sp.Dummy("b_variable", nonzero=True)
    beta_rs_variable = (r * b_variable + s / b_variable) / (2 * SQRT2)
    numerator_slope = (
        sp.diff(beta_rs_variable**2, b_variable)
        / sp.diff(central_charge_from_b(b_variable), b_variable)
    ).subs(b_variable, b)
    return clean(numerator_slope / norm_slope)


def contract_ramond_null(
    *,
    r: int,
    s: int,
    p_phi: int,
    eta: int,
    b: sp.Expr,
    lambda_i: sp.Expr,
    beta_j: sp.Expr,
    null_ground: int = 0,
    spectator_ground: int = 0,
    null_slot: int = 2,
) -> dict[str, sp.Expr]:
    """Compare a full PBW null-vector contraction with equation (5.10).

    ``null_ground`` selects chi^+ or chi^- and ``spectator_ground`` selects
    the other Ramond ground.  ``null_slot`` is 2 (the insertion at one) or 3
    (the ket at zero).  The returned ``generalized_residual`` tests the phase
    law found by the literal generalized plane Ward system,

        slot 2: rho_plane = (-1)^(rs/2) P_eff rho_shifted,
        slot 3: rho_plane =                 P_eff rho_shifted,

    where ``P_eff = P_rs^{R,(-1)^p_phi eta}``.
    """

    if null_ground not in (0, 1) or spectator_ground not in (0, 1):
        raise ValueError("ground labels must be 0 or 1")
    if null_slot not in (2, 3):
        raise ValueError("null_slot must be 2 (at one) or 3 (at zero)")
    pole = ramond_degenerate_data(r, s, b)
    level = int(pole["level"])
    central_charge = pole["c"]
    h_ns = clean(((b + 1 / b) ** 2 - lambda_i**2) / 8)
    h_third = clean(central_charge / 24 - beta_j**2)
    null_vector = degenerate_null_vector(r, s, b, null_ground)
    form_parity = (p_phi + null_ground + spectator_ground) % 2
    if null_slot == 2:
        left = GeneralizedNRRWard(
            p_phi=p_phi,
            form_parity=form_parity,
            eta=eta,
            h_ns=h_ns,
            h_second=pole["h"],
            h_third=h_third,
            beta_second=pole["beta"],
            beta_third=beta_j,
            central_charge=central_charge,
        ).vector_value(null_vector, ground3=spectator_ground)
        shifted = GeneralizedNRRWard(
            p_phi=p_phi,
            form_parity=form_parity,
            eta=eta,
            h_ns=h_ns,
            h_second=pole["h_shifted"],
            h_third=h_third,
            beta_second=pole["beta_shifted"],
            beta_third=beta_j,
            central_charge=central_charge,
        ).value((), (), null_ground, (), spectator_ground)
        coordinate_sign = (-1) ** level
    else:
        ward = GeneralizedNRRWard(
            p_phi=p_phi,
            form_parity=form_parity,
            eta=eta,
            h_ns=h_ns,
            h_second=h_third,
            h_third=pole["h"],
            beta_second=beta_j,
            beta_third=pole["beta"],
            central_charge=central_charge,
        )
        left = clean(
            sum(
                coefficient
                * ward.value(
                    (), (), spectator_ground, state.word, state.ground
                )
                for state, coefficient in null_vector.items()
            )
        )
        shifted = GeneralizedNRRWard(
            p_phi=p_phi,
            form_parity=form_parity,
            eta=eta,
            h_ns=h_ns,
            h_second=h_third,
            h_third=pole["h_shifted"],
            beta_second=beta_j,
            beta_third=pole["beta_shifted"],
            central_charge=central_charge,
        ).value((), (), spectator_ground, (), null_ground)
        coordinate_sign = 1
    polynomial = fusion_polynomial_510(r, s, lambda_i, beta_j, b, eta)
    effective_eta = (-1) ** p_phi * eta
    effective_polynomial = fusion_polynomial_510(
        r, s, lambda_i, beta_j, b, effective_eta
    )
    predicted = clean(polynomial * shifted)
    generalized_prediction = clean(
        coordinate_sign * effective_polynomial * shifted
    )
    return {
        "direct": clean(left),
        "shifted_ground": clean(shifted),
        "polynomial": polynomial,
        "effective_eta": sp.Integer(effective_eta),
        "effective_polynomial": effective_polynomial,
        "coordinate_sign": sp.Integer(coordinate_sign),
        "null_slot": sp.Integer(null_slot),
        "predicted": predicted,
        "generalized_prediction": generalized_prediction,
        "residual": clean(left - predicted),
        "generalized_residual": clean(left - generalized_prediction),
        "ratio_to_polynomial": clean(left / predicted),
    }


def contract_level_one_null(
    *,
    r: int,
    s: int,
    p_phi: int,
    eta: int,
    b: sp.Expr,
    lambda_i: sp.Expr,
    beta_j: sp.Expr,
) -> dict[str, sp.Expr]:
    """Compatibility wrapper for the original chi^+, w^+ level-one audit."""

    if r * s != 2:
        raise ValueError("this compatibility wrapper requires r*s=2")
    return contract_ramond_null(
        r=r,
        s=s,
        p_phi=p_phi,
        eta=eta,
        b=b,
        lambda_i=lambda_i,
        beta_j=beta_j,
    )
