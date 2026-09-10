"""Run the repository's algebraic Ward/oscillator rules over a prime field.

The selected definitions are compiled from the local source, with only
arithmetic operations changed. All matrix inverses are exact modulo PRIME.
This is a specialized identity check, not a complex numerical block value.
"""

import ast
from fractions import Fraction
from functools import lru_cache
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zero_mode_recovery import BRANCHING_RECURSION
import direct_state_check as direct
import ramond_pbw_generalized_ward as human
from modular_arithmetic import F, I, SQRT2, PRIME, array, divide, inverse, lu_factor, lu_solve, mm


def add_term(output, key, value):
    value = F(value)
    if value:
        output[key] = F(output.get(key, 0)) + value
        if not output[key]:
            del output[key]


@lru_cache(None)
def binomial(a, k):
    a = Fraction(a)
    value = Fraction(1)
    for j in range(k):
        value *= (a - j) / (j + 1)
    return value if k >= 0 else Fraction(0)


class ExactConstants(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, complex):
            return ast.Call(func=ast.Name(id="F",ctx=ast.Load()),args=[ast.Constant(node.value)],keywords=[])
        if isinstance(node.value, float):
            return ast.Call(func=ast.Name(id="Fraction", ctx=ast.Load()), args=[ast.Constant(str(node.value))], keywords=[])
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="divide", ctx=ast.Load()), args=[node.left, node.right], keywords=[])
        return node


def clone(definitions, namespace):
    source = "from __future__ import annotations\n" + "\n\n".join(inspect.getsource(x) for x in definitions)
    tree = ast.fix_missing_locations(ExactConstants().visit(ast.parse(source)))
    exec(compile(tree, "<exact specialization of repository rules>", "exec"), namespace)


def sqrt(value):
    import sympy as sp
    return F(sp.sqrt_mod(int(F(value)), PRIME))


import sympy as _sp
EIGHTH_ROOT_TWO = F(next(r for r in _sp.nthroot_mod(2, 8, PRIME, all_roots=True)
                        if pow(r, 4, PRIME) == int(SQRT2)))


namespace = dict(
    Fraction=Fraction, lru_cache=lru_cache, math=math, np=np,
    F=F, I=I, SQRT2=SQRT2, EIGHTH_PLUS=(1 + I) / SQRT2,
    divide=divide, clean=lambda x: x, add_term=add_term,
    float=F, complex=F, real_number=F, complex_number=F,
    scalar_sqrt=sqrt, scalar_power_of_two=lambda e: EIGHTH_ROOT_TWO ** (8 * Fraction(e)),
    arithmetic_tolerance=lambda: 0, TOLERANCE=0,
    sp=SimpleNamespace(S=SimpleNamespace(Zero=0, One=1), sympify=lambda x: x,
                       Rational=Fraction, Integer=int, binomial=binomial),
)
clone([human.word_level, human.word_parity, human.state_parity, human.SuperVirasoroModule,
       human.GeneralizedNRRWard], namespace)
clone([BRANCHING_RECURSION.partitions, BRANCHING_RECURSION.strict_partitions,
       BRANCHING_RECURSION.strict_odd_partitions, BRANCHING_RECURSION.apply_expression,
       BRANCHING_RECURSION.combine, BRANCHING_RECURSION.ell, BRANCHING_RECURSION.FreeFieldModule,
       BRANCHING_RECURSION.canonicalize_word, BRANCHING_RECURSION.VirasoroThreePoint,
       BRANCHING_RECURSION.ns_norm_squared, BRANCHING_RECURSION.ramond_norm_squared], namespace)

partitions = namespace["partitions"]
strict_partitions = namespace["strict_partitions"]
strict_odd_partitions = namespace["strict_odd_partitions"]
FreeFieldModule = namespace["FreeFieldModule"]
SuperModule = namespace["SuperVirasoroModule"]
HumanForm = namespace["GeneralizedNRRWard"]
VirasoroForm = namespace["VirasoroThreePoint"]


@lru_cache(None)
def metadata(sector, level):
    return tuple(
        (ls, gs) + (() if sector == "NS" else (g,))
        for a in range(level // 2 + 1 if sector == "NS" else level + 1)
        for ls in partitions(a)
        for gs in (strict_odd_partitions(level - 2*a) if sector == "NS" else strict_partitions(level-a))
        for g in ((0,) if sector == "NS" else (0, 1))
    )


@lru_cache(None)
def transition(sector, b, momentum, realization, level):
    module = FreeFieldModule(sector, b, momentum, realization)
    meta = metadata(sector, level)
    columns = []
    for state in meta:
        value = {((), ()) + (() if sector == "NS" else (state[2],)): F(1)}
        for kind, modes in (("G", state[1]), ("L", state[0])):
            for n in reversed(modes):
                acted = {}
                for current, outer in value.items():
                    for final, inner in (module.physical_g_on_state(-n, current) if kind == "G"
                                         else module.physical_l_on_state(-n, current)):
                        add_term(acted, final, outer * inner)
                value = acted
        columns.append(value)
    rows = tuple(sorted(set().union(*(c.keys() for c in columns)), key=repr))
    matrix = array([[column.get(row, 0) for column in columns] for row in rows])
    assert matrix.shape == (len(meta), len(meta))
    return rows, matrix, lu_factor(matrix), meta


def r_branch(self, label, parity):
    native, raw = self._raw_r_branch(Fraction(label), parity)
    if native == self.realization:
        return raw
    groups = {}
    for state, coefficient in raw.items():
        aux, physical = self.split_state(state)
        groups.setdefault(aux, {})[physical] = coefficient
    answer = {}
    for aux, value in groups.items():
        level = self.physical_level_units(next(iter(value)))
        rows, _, lu, _ = transition("R", self.b, self.momentum, native, level)
        target_rows, target_matrix, _, _ = transition("R", self.b, self.momentum, self.realization, level)
        converted = mm(target_matrix, lu_solve(lu, array([value.get(r, 0) for r in rows])))
        for row, coefficient in zip(target_rows, converted):
            add_term(answer, self.join_state(aux, row), F(int(coefficient)))
    return answer


FreeFieldModule.r_branch = r_branch


class NativeModule(SuperModule):
    def g0_action(self, ground):
        return -I * self.beta, 1 - ground


class PBWModule:
    def __init__(self, module):
        self.module = module
        self.sector = module.sector
        self.central_charge = F(Fraction(3, 2)) + 3 * module.q**2
        self.weight = (module.q**2 / 4 - module.momentum**2) / 2 + (F(Fraction(1, 16)) if self.sector == "R" else 0)
        self.algebra = NativeModule(self.weight, self.central_charge, sector=self.sector, beta=module.momentum / SQRT2)

    def level_units(self, state):
        return (2 * sum(state[0]) + sum(state[1])) if self.sector == "NS" else sum(state[0]) + sum(state[1])

    def parity(self, state):
        return (len(state[1]) + (state[2] if self.sector == "R" else 0)) % 2

    def ground(self, state):
        return state[2] if self.sector == "R" else 0

    def l0_weight(self, state):
        return self.weight + F(Fraction(self.level_units(state), 2 if self.sector == "NS" else 1))

    def has_oscillators(self, state):
        return bool(state[0] or state[1])

    def word(self, state):
        return tuple(("L", -n) for n in state[0]) + tuple(("G", -Fraction(n, 2 if self.sector == "NS" else 1)) for n in state[1])

    def from_word(self, word, ground):
        return (tuple(int(-n) for k, n in word if k == "L"),
                tuple(int(-n * (2 if self.sector == "NS" else 1)) for k, n in word if k == "G")) + (() if self.sector == "NS" else (ground,))

    @lru_cache(None)
    def act(self, kind, mode, state):
        if kind == "G" and self.sector == "NS":
            mode = Fraction(mode, 2)
        return tuple((self.from_word(w, g), F(c)) for (w, g), c in self.algebra.act(kind, mode, self.word(state), self.ground(state)).items())

    def from_fock(self, expression):
        if not expression:
            return {}
        level = self.module.physical_level_units(next(iter(expression)))
        rows, _, lu, meta = transition(self.sector, self.module.b, self.module.momentum, self.module.realization, level)
        values = lu_solve(lu, array([expression.get(row, 0) for row in rows]))
        return {state: F(int(c)) for state, c in zip(meta, values) if c}

    def to_fock(self, state):
        rows, matrix, _, meta = transition(self.sector, self.module.b, self.module.momentum, self.module.realization, self.level_units(state))
        return {row: F(int(c)) for row, c in zip(rows, matrix[:, meta.index(state)]) if c}

    @lru_cache(None)
    def inner(self, left, right):
        value = {right: F(1)}
        for kind, modes in (("L", left[0]), ("G", left[1])):
            for n in modes:
                acted = {}
                for state, outer in value.items():
                    for final, inner in self.act(kind, n, state):
                        add_term(acted, final, outer * inner)
                value = acted
        return value.get(((), ()) + (() if self.sector == "NS" else (left[2],)), F(0))


namespace["PBWModule"] = PBWModule
clone([direct.generalized_binomial, direct.branch_in_pbw, direct.add_acted_triples,
       direct.PhysicalThreePoint, direct.AuxiliaryThreePoint, direct.DirectBranchingCoefficient], namespace)


def physical_init(self, modules, form_parity, eta, primary_parity=0):
    self.modules = tuple(modules)
    self.form_parity, self.eta, self.primary_parity = int(form_parity), int(eta), int(primary_parity)
    self.infinity_phase, self.zero_phase = -I, I
    self._cache, self._active = {}, set()
    self.ramond_odd_phase = (-1 + I) / SQRT2


namespace["PhysicalThreePoint"].__init__ = physical_init
AuxiliaryForm = namespace["AuxiliaryThreePoint"]
AuxiliaryForm.base_value = lambda self, states: F(0) if states[1][1] != states[2][1] else (F(1) if states[1][1] == 0 else I)
Direct = namespace["DirectBranchingCoefficient"]


class Weights:
    def __init__(self, b, momenta):
        self.b, self.momenta = F(b), tuple(map(F, momenta))
        q = self.b + 1 / self.b
        self.central_charges = (1 + 3*q*q/(1-self.b**2), 1 + 3*q*q/(1-self.b**-2))

    def triple(self, labels, copy):
        b = self.b if copy == 0 else 1 / self.b
        q = b + 1 / b
        return tuple((q*q/4 - (p + 2 * F(n) * b)**2) / (2*(1-b*b)) for p, n in zip(self.momenta, labels))
