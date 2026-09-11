"""Exact plumbing expansion of CCY (1703.09805), equation (5.5).

Primitive closed walks in the sewing graph represent primitive Schottky
classes. Walks are identified with their inverses, so the oscillator product
has exponent -1. All truncations follow from plumbing degree, not a numerical
word-length guess. No oscillator-state sum or finite-c Gram matrix is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb


class SeriesRing:
    """Sparse rational series truncated by total plumbing degree."""

    def __init__(self, variables, cutoff):
        self.variables, self.cutoff = int(variables), int(cutoff)
        self.zero_key = (0,)*self.variables
        self.one = {self.zero_key: Fraction(1)}

    def constant(self, value):
        return {self.zero_key: Fraction(value)} if value else {}

    def variable(self, index):
        key = tuple(int(i == index) for i in range(self.variables))
        return {key: Fraction(1)} if self.cutoff >= 1 else {}

    def add(self, *terms):
        result = {}
        for term in terms:
            for key, value in term.items():
                result[key] = result.get(key, 0)+value
        return {key: value for key, value in result.items() if value}

    def scale(self, term, factor):
        return {key: value*factor for key, value in term.items() if value*factor}

    def multiply(self, first, second):
        result = {}
        for a, x in first.items():
            for b, y in second.items():
                key = tuple(i+j for i, j in zip(a, b))
                if sum(key) <= self.cutoff:
                    result[key] = result.get(key, 0)+x*y
        return {key: value for key, value in result.items() if value}

    def order(self, term):
        return min(map(sum, term), default=self.cutoff+1)

    def inverse_unit(self, term):
        constant = term.get(self.zero_key, 0)
        if not constant:
            raise ArithmeticError("A reduced sewing cycle must have unit trace at the cusp")
        remainder = self.add(self.one, self.scale(term, -1/constant))
        result, power = dict(self.one), dict(self.one)
        for _ in range(self.cutoff//self.order(remainder)):
            power = self.multiply(power, remainder)
            result = self.add(result, power)
        return self.scale(result, 1/constant)

    def exponential(self, term):
        if term.get(self.zero_key, 0):
            raise ValueError("The vacuum logarithm must have zero constant term")
        result, power = dict(self.one), dict(self.one)
        for j in range(1, self.cutoff//self.order(term)+1):
            power = self.scale(self.multiply(power, term), Fraction(1, j))
            result = self.add(result, power)
        return result

    def matrix_multiply(self, first, second):
        a, b, c, d = first
        e, f, g, h = second
        return (self.add(self.multiply(a, e), self.multiply(b, g)),
                self.add(self.multiply(a, f), self.multiply(b, h)),
                self.add(self.multiply(c, e), self.multiply(d, g)),
                self.add(self.multiply(c, f), self.multiply(d, h)))

    def determinant(self, matrix):
        a, b, c, d = matrix
        return self.add(self.multiply(a, d), self.scale(self.multiply(b, c), -1))


@dataclass(frozen=True)
class SewingMap:
    source: int
    target: int
    inverse: int
    matrix: tuple


def primitive_cycles(edges, costs, maximum_cost):
    """Unoriented primitive free homotopy classes of a finite sewing graph.

    Arc identifiers retain their vertices and orientations. In particular,
    a theta walk with odd period in its *unoriented edge labels* need not be
    a proper power: only repetition of the full directed arc word counts.
    """
    outgoing = {}
    for index, edge in enumerate(edges):
        outgoing.setdefault(edge.source, []).append(index)
        other = edges[edge.inverse]
        if (other.source, other.target, other.inverse) != (edge.target, edge.source, index):
            raise ValueError("Sewing maps must have consistent inverse arcs")
    classes = set()

    def visit(start, vertex, word, cost):
        if word and vertex == start and word[-1] != edges[word[0]].inverse:
            length = len(word)
            if not any(length % p == 0 and word == word[:p]*(length//p)
                       for p in range(1, length)):
                inverse = tuple(edges[index].inverse for index in reversed(word))
                classes.add(min(w[i:]+w[:i] for w in (word, inverse)
                                for i in range(length)))
        for index in outgoing.get(vertex, ()):
            if word and index == edges[word[-1]].inverse:
                continue
            next_cost = cost+costs[index]
            if next_cost <= maximum_cost:
                visit(start, edges[index].target, word+(index,), next_cost)

    for vertex in outgoing:
        visit(vertex, vertex, (), 0)
    return tuple(sorted(classes, key=lambda word: (sum(costs[i] for i in word), word)))


def vacuum_series(edges, ring):
    """CCY Schottky product for supplied plumbing maps on a sewing graph.

    Assumptions: each edge determinant has positive plumbing order and every
    reduced closed walk has nonzero constant trace in the chosen local
    projective matrices. These hold for the standard 0,1,infinity plumbing
    maps. The inverse-arc matrices must represent inverse Mobius maps.
    """
    determinants = tuple(ring.determinant(edge.matrix) for edge in edges)
    costs = tuple(ring.order(determinant) for determinant in determinants)
    if min(costs, default=1) <= 0:
        raise ValueError("Every sewing determinant must have positive plumbing order")
    classes = primitive_cycles(edges, costs, ring.cutoff//2)
    logarithm, diagnostics = {}, []
    for word in classes:
        matrix = (ring.one, {}, {}, ring.one)
        determinant = dict(ring.one)
        for index in word:
            matrix = ring.matrix_multiply(edges[index].matrix, matrix)
            determinant = ring.multiply(determinant, determinants[index])
        inverse_trace = ring.inverse_unit(ring.add(matrix[0], matrix[3]))
        ratio = ring.multiply(determinant, ring.multiply(inverse_trace, inverse_trace))
        order = ring.order(ratio)
        # If k is the small/large eigenvalue ratio, t=det(M)/tr(M)^2
        # equals k/(1+k)^2. Thus k=sum_{j>=1}Catalan_j*t^j.
        multiplier, power = {}, dict(ring.one)
        for j in range(1, ring.cutoff//order+1):
            power = ring.multiply(power, ratio)
            multiplier = ring.add(multiplier, ring.scale(power, comb(2*j, j)//(j+1)))
        # -sum_{m>=2}log(1-k^m): the coefficient of k^s is
        # sum_{m|s,m>=2}m/s. Each inverse pair of classes occurs once.
        power = dict(multiplier)
        for s in range(2, ring.cutoff//order+1):
            power = ring.multiply(power, multiplier)
            coefficient = Fraction(sum(m for m in range(2, s+1) if s % m == 0), s)
            logarithm = ring.add(logarithm, ring.scale(power, coefficient))
        diagnostics.append({"directed_arc_word": word, "multiplier_order": order})
    return ring.exponential(logarithm), diagnostics


@lru_cache(maxsize=16)
def _theta_seed(cutoff):
    if cutoff < 0:
        raise ValueError("The plumbing cutoff must be nonnegative")
    ring = SeriesRing(3, cutoff)
    if cutoff < 4:
        return tuple(ring.one.items()), ()
    q1, q2, q3 = (ring.variable(i) for i in range(3))
    # Edges 1,2,3 meet infinity,one,zero respectively on both spheres.
    # Their projective crossing maps are z' = 1/(q1*z),
    # z' = 1+q2/(z-1), and z' = q3/z. Each matrix squares to q_i I.
    matrices = (({}, ring.one, q1, {}),
                (ring.one, ring.add(q2, ring.constant(-1)), ring.one, ring.constant(-1)),
                ({}, q3, ring.one, {}))
    edges = tuple(SewingMap(source, 1-source, 2*i+1-source, matrix)
                  for i, matrix in enumerate(matrices) for source in (0, 1))
    series, diagnostics = vacuum_series(edges, ring)
    return tuple(sorted(series.items())), tuple(diagnostics)


def theta_vacuum_seed(cutoff):
    """Exact rational coefficients in (q1,q2,q3), with total degree <= cutoff."""
    return dict(_theta_seed(int(cutoff))[0])


def theta_seed_provenance(cutoff):
    records = _theta_seed(int(cutoff))[1]
    return {"backend": "CCY Eq. (5.5), exact Schottky primitive product",
            "arithmetic": "rational formal power series",
            "edge_order": ["infinity", "one", "zero"],
            "primitive_classes": len(records),
            "classes": list(records),
            "inverse_class_identified": True, "oscillator_product_exponent": -1,
            "cutoff_rule": "2*multiplier plumbing order <= total cutoff"}
