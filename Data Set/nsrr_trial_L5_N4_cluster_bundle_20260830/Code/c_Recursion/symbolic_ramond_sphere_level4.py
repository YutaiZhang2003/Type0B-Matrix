"""Exact finite-c long-R sphere-block coefficients through level four.

This module implements the definition of the local four-point block: at
each level it constructs the full even Ramond PBW Gram matrix, constructs
the two RRNS Ward vectors, and sews them with the inverse Gram matrix.
It deliberately contains no pole recursion, large-c contraction, elliptic
change of variables, or global-block decomposition.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import argparse

import sympy as sp
from sympy.polys.matrices import DomainMatrix


Mode = tuple[str, int]
State = tuple[Mode, ...]


def integer_partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    maximum = total if maximum is None else min(maximum, total)
    for first in range(maximum, 0, -1):
        for tail in integer_partitions(total - first, first):
            yield (first,) + tail


def strict_partitions(total: int, maximum: int | None = None):
    if total == 0:
        yield ()
        return
    maximum = total if maximum is None else min(maximum, total)
    for first in range(maximum, 0, -1):
        for tail in strict_partitions(total - first, first - 1):
            yield (first,) + tail


def state_level(state: State) -> int:
    return sum(-index for _, index in state if index < 0)


def state_parity(state: State) -> int:
    return sum(kind == "G" for kind, _ in state) % 2


def mode_parity(mode: Mode) -> int:
    return int(mode[0] == "G")


def zone(mode: Mode) -> int:
    return 0 if mode[1] < 0 else (1 if mode[1] == 0 else 2)


class ExactRamondModule:
    def __init__(self, c, beta, h=None):
        self.c = c
        self.beta = beta
        self.h = c / 24 - beta**2 if h is None else h
        self.kappa_squared = -beta**2

    @lru_cache(maxsize=None)
    def basis(self, level: int, parity: int = 0) -> tuple[State, ...]:
        states = []
        for g_level in range(level + 1):
            for g_partition in strict_partitions(g_level):
                for l_partition in integer_partitions(level - g_level):
                    ground_g0 = (parity - len(g_partition)) % 2
                    states.append(
                        tuple(
                            [("L", -part) for part in l_partition]
                            + [("G", -part) for part in g_partition]
                            + ([("G", 0)] if ground_g0 else [])
                        )
                    )
        states.sort(
            key=lambda state: (
                sum(kind == "G" and index < 0 for kind, index in state),
                tuple((kind, -index) for kind, index in state),
            )
        )
        return tuple(states)

    @staticmethod
    def bpz(state: State) -> State:
        return tuple((kind, -index) for kind, index in reversed(state))

    def bracket(self, left: Mode, right: Mode):
        lk, m = left
        rk, n = right
        if lk == "L" and rk == "L":
            terms = [(m - n, ("L", m + n))]
            if m + n == 0:
                terms.append((self.c * (m**3 - m) / 12, None))
        elif lk == "L" and rk == "G":
            terms = [(sp.Rational(m, 2) - n, ("G", m + n))]
        elif lk == "G" and rk == "L":
            terms = [(m - sp.Rational(n, 2), ("G", m + n))]
        else:
            terms = [(sp.Integer(2), ("L", m + n))]
            if m + n == 0:
                terms.append((self.c * (m**2 - sp.Rational(1, 4)) / 3, None))
        return tuple((coefficient, mode) for coefficient, mode in terms if coefficient != 0)

    @lru_cache(maxsize=None)
    def expectation(self, word: State):
        for index in range(len(word) - 1):
            left, right = word[index], word[index + 1]
            if zone(left) <= zone(right):
                continue
            sign = -1 if mode_parity(left) and mode_parity(right) else 1
            swapped = word[:index] + (right, left) + word[index + 2 :]
            result = sign * self.expectation(swapped)
            for coefficient, replacement in self.bracket(left, right):
                reduced = (
                    word[:index]
                    + (() if replacement is None else (replacement,))
                    + word[index + 2 :]
                )
                result += coefficient * self.expectation(reduced)
            return sp.expand(result)
        if any(index != 0 for _, index in word):
            return sp.Integer(0)
        l0_count = sum(kind == "L" for kind, _ in word)
        g0_count = sum(kind == "G" for kind, _ in word)
        if g0_count % 2:
            return sp.Integer(0)
        return self.h**l0_count * self.kappa_squared ** (g0_count // 2)

    def inner_product(self, left: State, right: State):
        return self.expectation(self.bpz(left) + right)

    @lru_cache(maxsize=None)
    def gram(self, level: int, parity: int = 0) -> sp.Matrix:
        basis = self.basis(level, parity)
        return sp.Matrix(
            [[self.inner_product(left, right) for right in basis] for left in basis]
        )

    @lru_cache(maxsize=None)
    def mode_action(self, mode: Mode, state: State):
        target_level = state_level(state) - mode[1]
        target_parity = (state_parity(state) + mode_parity(mode)) % 2
        if target_level < 0:
            return ()
        target_basis = self.basis(target_level, target_parity)
        overlaps = sp.Matrix(
            [
                self.expectation(self.bpz(test_state) + (mode,) + state)
                for test_state in target_basis
            ]
        )
        coordinates = self.gram(target_level, target_parity).inv().multiply(overlaps)
        return tuple(
            (basis_state, sp.cancel(coefficient))
            for basis_state, coefficient in zip(target_basis, coordinates)
            if coefficient != 0
        )


class ExactRamondWardVector:
    def __init__(self, module, external_beta, external_r_weight, external_ns_weight, eta):
        self.module = module
        self.external_beta = external_beta
        self.external_r_weight = external_r_weight
        self.external_ns_weight = external_ns_weight
        self.eta = eta

    @staticmethod
    def g0_coefficient(beta, parity):
        return ((1 + sp.I) if parity == 0 else (-1 + sp.I)) * beta / sp.sqrt(2)

    @lru_cache(maxsize=None)
    def value(self, state: State):
        parity = state_parity(state)
        level = state_level(state)
        if level == 0:
            if state == ():
                return sp.Integer(1)
            if state == (("G", 0),):
                # Ground-fibre convention used in the c-recursion note:
                # rho(G_0 w) = + eta g_0(beta) rho(w).  It gives
                # rho(G_{-1}G_0w)=-beta^2/2-eta beta beta_ext.
                return self.eta * self.g0_coefficient(self.module.beta, 0)
            raise ValueError(state)

        first, rest = state[0], state[1:]
        if first[0] == "L":
            n = -first[1]
            return sp.expand(
                (
                    self.module.h
                    + state_level(rest)
                    + n * self.external_r_weight
                    - self.external_ns_weight
                )
                * self.value(rest)
            )

        n = -first[1]
        result = (
            self.g0_coefficient(self.external_beta, parity)
            * self.value(rest)
            / (sp.I * (-1) ** state_parity(rest))
        )
        for p in range(1, level + 1):
            lower = sum(
                coefficient * self.value(lower_state)
                for lower_state, coefficient in self.module.mode_action(("G", -n + p), rest)
            )
            result -= sp.binomial(sp.Rational(1, 2), p) * (-1) ** p * lower
        return sp.cancel(sp.expand_complex(result))

    def vector(self, level: int) -> sp.Matrix:
        return sp.Matrix([[self.value(state) for state in self.module.basis(level, 0)]])


def exact_coefficients(max_level: int = 4):
    c, beta, beta2, beta3, h1, h4, eta2, eta3 = sp.symbols(
        "c beta beta2 beta3 h1 h4 eta2 eta3", real=True
    )
    module = ExactRamondModule(c, beta)
    left = ExactRamondWardVector(
        module, beta3, c / 24 - beta3**2, h4, eta3
    )
    right = ExactRamondWardVector(
        module, beta2, c / 24 - beta2**2, h1, eta2
    )
    result = [sp.Integer(1)]
    for level in range(1, max_level + 1):
        gram = module.gram(level, 0)
        rhs = right.vector(level).T
        domain_gram, domain_rhs = DomainMatrix.from_Matrix(gram).unify(
            DomainMatrix.from_Matrix(rhs), fmt="sparse"
        )
        numerator_vector, denominator = domain_gram.solve_den(
            domain_rhs, method="charpoly"
        )
        numerator = (left.vector(level) * numerator_vector.to_Matrix())[0]
        result.append(numerator / denominator.as_expr())
    return (c, beta, beta2, beta3, h1, h4, eta2, eta3), tuple(result)


def exact_level_data(max_level: int = 4):
    """Return the exact matrices/vectors defining the finite-c block."""

    c, beta, beta2, beta3, h1, h4, eta2, eta3 = sp.symbols(
        "c beta beta2 beta3 h1 h4 eta2 eta3", real=True
    )
    module = ExactRamondModule(c, beta)
    left = ExactRamondWardVector(
        module, beta3, c / 24 - beta3**2, h4, eta3
    )
    right = ExactRamondWardVector(
        module, beta2, c / 24 - beta2**2, h1, eta2
    )
    data = []
    for level in range(1, max_level + 1):
        data.append(
            (
                module.basis(level, 0),
                module.gram(level, 0),
                left.vector(level),
                right.vector(level),
            )
        )
    return (c, beta, beta2, beta3, h1, h4, eta2, eta3), tuple(data)


def formal_coefficient(gram: sp.Matrix, left: sp.Matrix, right: sp.Matrix):
    """Return f_N as an unevaluated bordered-determinant quotient."""

    border = sp.Matrix.vstack(
        sp.Matrix.hstack(sp.zeros(1, 1), left),
        sp.Matrix.hstack(right.T, gram),
    )
    return -sp.Determinant(border) / sp.Determinant(gram)


def dump_exact_markdown(path: str | Path, max_level: int = 4) -> None:
    """Write all finite-c level data in exact, machine-readable SymPy form."""

    symbols, data = exact_level_data(max_level)
    names = "c, beta, beta2, beta3, h1, h4, eta2, eta3"
    lines = [
        "# Exact long-R sphere block through level four",
        "",
        "All entries below are exact.  Define",
        "",
        "```python",
        "import sympy as sp",
        f"{names} = sp.symbols('{names}', real=True)",
        "```",
        "",
        "The full local coefficient is `fN = (rhoL[N] * B[N].inv() * rhoR[N].T)[0]`.",
        "Equivalently it is the negative bordered determinant divided by `det(B[N])`.",
        "The full block is `z**(-beta**2 + beta2**2 - h1) * (1 + sum(fN*z**N, N=1..4) + O(z**5))`.",
        "",
    ]
    for level, (basis, gram, left, right) in enumerate(data, start=1):
        lines.extend(
            [
                f"## Level {level}",
                "",
                "```python",
                f"basis{level} = {basis!r}",
                f"B{level} = sp.Matrix({sp.sstr(gram.tolist())})",
                f"rhoL{level} = sp.Matrix({sp.sstr(left.tolist())})",
                f"rhoR{level} = sp.Matrix({sp.sstr(right.tolist())})",
                f"f{level} = (rhoL{level} * B{level}.inv() * rhoR{level}.T)[0]",
                "```",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-markdown")
    parser.add_argument("--max-level", type=int, default=4)
    args = parser.parse_args()
    if args.dump_markdown:
        dump_exact_markdown(args.dump_markdown, args.max_level)
    else:
        _, data = exact_level_data(args.max_level)
        for level, (_, gram, left, right) in enumerate(data, start=1):
            print(f"f{level} = {formal_coefficient(gram, left, right)}")
