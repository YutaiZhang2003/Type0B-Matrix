#!/usr/bin/env python3
r"""Fast meromorphic cube IBP for products of polynomial powers.

The five-point Hepp integrands have considerably more structure than a
generic symbolic expression.  Their smooth parts are finite sums of terms

.. math::

   c_r\prod_s P_{rs}(u)^{\lambda_{rs}},

where every ``P`` is a low-degree polynomial (in practice a sum of a few
monomials).  This module keeps that representation factored.  Mixed
derivatives required by Mellin integration by parts are evaluated with
truncated multivariate Taylor jets at quadrature nodes; no symbolic
differentation or expression expansion is performed.

Only fixed tensor-product Gauss--Jacobi terminal rules are supported.  This
is intentional: the API is a small, deterministic numerical backend for
the three-dimensional Hepp cubes, not a replacement for a computer algebra
system.  As in :func:`spin23_twisted_periods.meromorphic_cube_integral`, an
exact Mellin pole is an error.  The caller must use one common regulator and
sum the regulated sectors before extracting a Laurent coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Mapping, Sequence

import numpy as np


MultiIndex = tuple[int, ...]


class FactoredPoleError(ArithmeticError):
    """An exact integration-by-parts denominator was encountered."""


@dataclass(frozen=True, init=False)
class SparsePolynomial:
    """A small multivariate polynomial stored as nonzero monomials.

    ``coefficients`` maps exponent tuples to coefficients.  For example,
    ``{(0, 0): 1, (1, 0): 2, (1, 1): 3}`` represents
    ``1 + 2*u + 3*u*v``.
    """

    dimension: int
    terms: tuple[tuple[MultiIndex, complex], ...]

    def __init__(
        self,
        coefficients: Mapping[Sequence[int], complex],
        *,
        dimension: int | None = None,
    ) -> None:
        raw = tuple((tuple(int(power) for power in powers), complex(value))
                    for powers, value in coefficients.items())
        if dimension is None:
            if not raw:
                raise ValueError("dimension is required for the zero polynomial")
            inferred_dimension = len(raw[0][0])
        else:
            inferred_dimension = int(dimension)
        if inferred_dimension < 0:
            raise ValueError("polynomial dimension must be nonnegative")
        combined: dict[MultiIndex, complex] = {}
        for powers, coefficient in raw:
            if len(powers) != inferred_dimension:
                raise ValueError("all polynomial multi-indices need one common dimension")
            if any(power < 0 for power in powers):
                raise ValueError("polynomial powers must be nonnegative integers")
            combined[powers] = combined.get(powers, 0.0j) + coefficient
        normalized = tuple(
            sorted(
                (
                    (powers, coefficient)
                    for powers, coefficient in combined.items()
                    if coefficient != 0
                ),
                key=lambda item: item[0],
            )
        )
        object.__setattr__(self, "dimension", inferred_dimension)
        object.__setattr__(self, "terms", normalized)

    @classmethod
    def constant(cls, dimension: int, value: complex) -> "SparsePolynomial":
        """Construct a constant polynomial."""

        return cls({(0,) * int(dimension): value}, dimension=dimension)

    @classmethod
    def coordinate(cls, dimension: int, index: int) -> "SparsePolynomial":
        """Construct the coordinate polynomial ``u[index]``."""

        normalized_dimension = int(dimension)
        normalized_index = int(index)
        if normalized_index < 0 or normalized_index >= normalized_dimension:
            raise ValueError("coordinate index lies outside the polynomial dimension")
        powers = [0] * normalized_dimension
        powers[normalized_index] = 1
        return cls({tuple(powers): 1}, dimension=normalized_dimension)


@dataclass(frozen=True)
class PolynomialPower:
    """One factor ``polynomial**exponent`` in a smooth term."""

    polynomial: SparsePolynomial
    exponent: complex


@dataclass(frozen=True)
class FactoredTerm:
    """A coefficient times a product of polynomial powers."""

    factors: tuple[PolynomialPower, ...] = ()
    coefficient: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if self.factors:
            dimension = self.factors[0].polynomial.dimension
            if any(
                factor.polynomial.dimension != dimension for factor in self.factors
            ):
                raise ValueError("all factors in a term need one common dimension")


@dataclass(frozen=True)
class FactoredPowerSum:
    """A finite sum of :class:`FactoredTerm` objects."""

    terms: tuple[FactoredTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("a factored power sum must contain at least one term")


@dataclass(frozen=True)
class FactoredMeromorphicCubeIntegral:
    """Value and fixed-rule error estimate from factored Mellin IBP."""

    value: complex | np.ndarray
    absolute_error: float | np.ndarray
    quadrature_calls: int
    converged: bool


def _rising(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= value + offset
    return result


def _default_ibp_order(value: complex, *, margin: float) -> int:
    order = 0
    while (value + order).real <= margin:
        order += 1
    return order


def _multi_indices(orders: Sequence[int]) -> tuple[MultiIndex, ...]:
    return tuple(product(*(range(order + 1) for order in orders)))


def _jet_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two rectangular Taylor jets, truncating componentwise."""

    orders = tuple(length - 1 for length in left.shape[:-1])
    result = np.zeros_like(left, dtype=np.complex128)
    indices = _multi_indices(orders)
    for alpha in indices:
        value = np.zeros(left.shape[-1], dtype=np.complex128)
        beta_ranges = tuple(range(entry + 1) for entry in alpha)
        for beta in product(*beta_ranges):
            complement = tuple(a - b for a, b in zip(alpha, beta))
            value += left[beta] * right[complement]
        result[alpha] = value
    return result


def _polynomial_jet(
    polynomial: SparsePolynomial,
    points: tuple[np.ndarray, ...],
    orders: MultiIndex,
) -> np.ndarray:
    """Return Taylor coefficients ``partial**alpha P / alpha!``."""

    point_count = points[0].size if points else 1
    shape = tuple(order + 1 for order in orders) + (point_count,)
    result = np.zeros(shape, dtype=np.complex128)
    for alpha in _multi_indices(orders):
        value = np.zeros(point_count, dtype=np.complex128)
        for powers, coefficient in polynomial.terms:
            if any(power < derivative for power, derivative in zip(powers, alpha)):
                continue
            monomial = np.full(point_count, coefficient, dtype=np.complex128)
            for coordinate, power, derivative in zip(points, powers, alpha):
                monomial *= (
                    math.comb(power, derivative)
                    * coordinate ** (power - derivative)
                )
            value += monomial
        result[alpha] = value
    return result


def _power_jet(base: np.ndarray, exponent: complex) -> np.ndarray:
    """Raise a Taylor jet to a complex power by a finite binomial series."""

    orders = tuple(length - 1 for length in base.shape[:-1])
    constant_index = (0,) * len(orders)
    constant = base[constant_index]
    if np.any(np.abs(constant) < 1.0e-290):
        raise ZeroDivisionError(
            "a polynomial factor vanished at a quadrature node; "
            "the factored Taylor expansion is not valid there"
        )
    normalized = base / constant.reshape((1,) * len(orders) + (-1,))
    normalized[constant_index] = 0.0

    one = np.zeros_like(base, dtype=np.complex128)
    one[constant_index] = 1.0
    result = one.copy()
    power = one.copy()
    binomial = 1.0 + 0.0j
    # A zero-constant jet is nilpotent above this total degree.
    for degree in range(1, sum(orders) + 1):
        power = _jet_multiply(power, normalized)
        binomial *= (complex(exponent) - degree + 1) / degree
        result += binomial * power
    result *= np.power(constant, complex(exponent)).reshape(
        (1,) * len(orders) + (-1,)
    )
    return result


def _term_jet(
    term: FactoredTerm,
    points: tuple[np.ndarray, ...],
    orders: MultiIndex,
) -> np.ndarray:
    point_count = points[0].size if points else 1
    shape = tuple(order + 1 for order in orders) + (point_count,)
    result = np.zeros(shape, dtype=np.complex128)
    result[(0,) * len(orders)] = complex(term.coefficient)
    for factor in term.factors:
        if complex(factor.exponent) == 0:
            continue
        base = _polynomial_jet(factor.polynomial, points, orders)
        result = _jet_multiply(result, _power_jet(base, complex(factor.exponent)))
    return result


def _evaluate_derivatives(
    expressions: tuple[FactoredPowerSum, ...],
    points: tuple[np.ndarray, ...],
    derivative: MultiIndex,
) -> np.ndarray:
    """Evaluate one mixed derivative of every power sum at many points."""

    factorial = math.prod(math.factorial(order) for order in derivative)
    values = np.zeros((len(expressions), points[0].size if points else 1), dtype=np.complex128)
    for expression_index, expression in enumerate(expressions):
        for term in expression.terms:
            if term.factors and term.factors[0].polynomial.dimension != len(points):
                raise ValueError("factor dimension does not match Mellin dimension")
            values[expression_index] += factorial * _term_jet(
                term, points, derivative
            )[derivative]
    return values


@lru_cache(maxsize=256)
def _unit_gauss_jacobi(
    order: int,
    exponent: complex,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights for ``integral_0^1 u**(A-1) f(u) du``."""

    if order < 4:
        raise ValueError("Gauss--Jacobi order must be at least four")
    exponent = complex(exponent)
    if exponent.real <= 0:
        raise ValueError("Gauss--Jacobi exponents need positive real part")
    if abs(exponent.imag) < 1.0e-14:
        from scipy.special import roots_jacobi

        raw_nodes, raw_weights = roots_jacobi(order, 0.0, exponent.real - 1.0)
        nodes = np.asarray((raw_nodes + 1.0) / 2.0, dtype=np.complex128)
        weights = np.asarray(
            raw_weights * 2.0 ** (-exponent.real), dtype=np.complex128
        )
    else:
        # Complex-symmetric Golub--Welsch construction for Jacobi(0,A-1).
        beta = exponent - 1.0
        matrix = np.zeros((order, order), dtype=np.complex128)
        for degree in range(order):
            if degree == 0 and abs(beta) < 1.0e-14:
                diagonal = 0.0j
            else:
                diagonal = beta**2 / (
                    (2 * degree + beta) * (2 * degree + beta + 2)
                )
            matrix[degree, degree] = diagonal
        for degree in range(1, order):
            off_diagonal = 2.0 / (2 * degree + beta) * np.sqrt(
                degree**2 * (degree + beta) ** 2
                / ((2 * degree + beta - 1) * (2 * degree + beta + 1))
            )
            matrix[degree - 1, degree] = off_diagonal
            matrix[degree, degree - 1] = off_diagonal
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        nodes = (eigenvalues + 1.0) / 2.0
        weights = np.empty(order, dtype=np.complex128)
        for index in range(order):
            vector = eigenvectors[:, index]
            bilinear_norm = np.sum(vector * vector)
            if abs(bilinear_norm) < 1.0e-13:
                raise ArithmeticError(
                    "complex Gauss--Jacobi eigenvector is self-orthogonal"
                )
            weights[index] = vector[0] ** 2 / (exponent * bilinear_norm)
    # Cached arrays are read-only so callers cannot corrupt later rules.
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _terminal_quadrature(
    expressions: tuple[FactoredPowerSum, ...],
    derivative: MultiIndex,
    fixed: tuple[bool, ...],
    remaining_exponents: tuple[complex | None, ...],
    order: int,
) -> np.ndarray:
    dimension = len(derivative)
    active = tuple(index for index in range(dimension) if not fixed[index])
    if not active:
        points = tuple(np.ones(1, dtype=np.complex128) for _ in range(dimension))
        return _evaluate_derivatives(expressions, points, derivative)[:, 0]

    nodes_by_axis: list[np.ndarray] = []
    weights_by_axis: list[np.ndarray] = []
    for index in active:
        exponent = remaining_exponents[index]
        if exponent is None:
            raise RuntimeError("an integrated coordinate is missing its exponent")
        nodes, weights = _unit_gauss_jacobi(order, exponent)
        nodes_by_axis.append(nodes)
        weights_by_axis.append(weights)
    node_mesh = np.meshgrid(*nodes_by_axis, indexing="ij")
    weight_mesh = np.meshgrid(*weights_by_axis, indexing="ij")
    point_count = node_mesh[0].size
    points_list: list[np.ndarray] = []
    active_position = {axis: position for position, axis in enumerate(active)}
    for axis in range(dimension):
        if fixed[axis]:
            points_list.append(np.ones(point_count, dtype=np.complex128))
        else:
            points_list.append(node_mesh[active_position[axis]].reshape(-1))
    total_weight = np.ones(point_count, dtype=np.complex128)
    for mesh in weight_mesh:
        total_weight *= mesh.reshape(-1)
    values = _evaluate_derivatives(expressions, tuple(points_list), derivative)
    return np.sum(values * total_weight[None, :], axis=1)


def _terminal_branches(
    exponents: tuple[complex, ...],
    orders: tuple[int, ...],
    *,
    pole_tolerance: float,
) -> tuple[
    tuple[complex, MultiIndex, tuple[bool, ...], tuple[complex | None, ...]], ...
]:
    """Expand the commuting one-coordinate IBP recursions."""

    choices: list[list[tuple[complex, int, bool, complex | None]]] = []
    for exponent, order in zip(exponents, orders):
        if order == 0:
            choices.append([(1.0 + 0.0j, 0, False, exponent)])
            continue
        axis_choices: list[tuple[complex, int, bool, complex | None]] = []
        for derivative in range(order):
            denominator = _rising(exponent, derivative + 1)
            if abs(denominator) <= pole_tolerance:
                raise FactoredPoleError(
                    "an exact IBP pole was encountered; introduce one common "
                    "complex regulator and sum sectors before Laurent extraction"
                )
            axis_choices.append(
                (((-1) ** derivative) / denominator, derivative, True, None)
            )
        denominator = _rising(exponent, order)
        if abs(denominator) <= pole_tolerance:
            raise FactoredPoleError(
                "an exact IBP pole was encountered; introduce one common "
                "complex regulator and sum sectors before Laurent extraction"
            )
        axis_choices.append(
            (
                ((-1) ** order) / denominator,
                order,
                False,
                exponent + order,
            )
        )
        choices.append(axis_choices)

    branches = []
    for selection in product(*choices):
        branches.append(
            (
                math.prod(item[0] for item in selection),
                tuple(item[1] for item in selection),
                tuple(item[2] for item in selection),
                tuple(item[3] for item in selection),
            )
        )
    return tuple(branches)


def factored_meromorphic_cube_integral(
    expression: FactoredPowerSum | Sequence[FactoredPowerSum],
    exponents: Sequence[complex],
    *,
    ibp_orders: Sequence[int] | None = None,
    convergence_margin: float = 1.0e-10,
    pole_tolerance: float = 1.0e-13,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-11,
    rule: str = "gauss-jacobi-12",
) -> FactoredMeromorphicCubeIntegral:
    r"""Meromorphically integrate factored power sums on a Mellin cube.

    The represented integral is

    .. math::

       \int_{[0,1]^d}\prod_i u_i^{A_i-1}H(u)\,d^du.

    The integration-by-parts identity and default order selection agree with
    :func:`spin23_twisted_periods.meromorphic_cube_integral`.  ``rule`` must
    be ``"gauss-jacobi-N"``; order ``N+4`` supplies the returned error
    estimate.  A sequence of power sums is evaluated as one vector request.
    """

    if isinstance(expression, FactoredPowerSum):
        expressions = (expression,)
        scalar_output = True
    else:
        expressions = tuple(expression)
        scalar_output = False
        if not expressions:
            raise ValueError("at least one factored power sum is required")
        if not all(isinstance(item, FactoredPowerSum) for item in expressions):
            raise TypeError("expression entries must be FactoredPowerSum objects")

    if not rule.startswith("gauss-jacobi-"):
        raise ValueError("the factored backend supports only gauss-jacobi-N rules")
    try:
        quadrature_order = int(rule.rsplit("-", 1)[1])
    except ValueError as error:
        raise ValueError("a fixed rule must be named gauss-jacobi-N") from error
    if quadrature_order < 4:
        raise ValueError("Gauss--Jacobi order must be at least four")

    mellin_exponents = tuple(complex(value) for value in exponents)
    dimension = len(mellin_exponents)
    for power_sum in expressions:
        for term in power_sum.terms:
            if term.factors and term.factors[0].polynomial.dimension != dimension:
                raise ValueError("factor dimension does not match Mellin dimension")

    if ibp_orders is None:
        orders = tuple(
            _default_ibp_order(value, margin=convergence_margin)
            for value in mellin_exponents
        )
    else:
        orders = tuple(int(value) for value in ibp_orders)
        if len(orders) != dimension or any(value < 0 for value in orders):
            raise ValueError("ibp_orders must give one nonnegative integer per axis")
        if any(
            (exponent + order).real <= convergence_margin
            for exponent, order in zip(mellin_exponents, orders)
        ):
            raise ValueError("the requested IBP order does not reach convergence")

    branches = _terminal_branches(
        mellin_exponents, orders, pole_tolerance=pole_tolerance
    )
    values = np.zeros(len(expressions), dtype=np.complex128)
    errors = np.zeros(len(expressions), dtype=float)
    converged = True
    quadrature_calls = 0
    for coefficient, derivative, fixed, remaining in branches:
        low = _terminal_quadrature(
            expressions, derivative, fixed, remaining, quadrature_order
        )
        high = _terminal_quadrature(
            expressions, derivative, fixed, remaining, quadrature_order + 4
        )
        branch_error = np.abs(high - low)
        values += coefficient * high
        errors += abs(coefficient) * branch_error
        tolerances = atol + rtol * np.abs(high)
        converged = converged and bool(np.all(branch_error <= tolerances))
        if any(not entry for entry in fixed):
            quadrature_calls += 2

    if scalar_output:
        return FactoredMeromorphicCubeIntegral(
            value=complex(values[0]),
            absolute_error=float(errors[0]),
            quadrature_calls=quadrature_calls,
            converged=converged,
        )
    return FactoredMeromorphicCubeIntegral(
        value=values,
        absolute_error=errors,
        quadrature_calls=quadrature_calls,
        converged=converged,
    )


def hepp_factored_term(
    permutation: Sequence[int],
    divisor_powers: Mapping[tuple[int, ...], complex],
    *,
    monomial_shifts: Sequence[int] = (0, 0, 0),
    coefficient: complex = 1.0 + 0.0j,
) -> FactoredTerm:
    r"""Build one smooth Hepp term without constructing a symbolic formula.

    The result represents

    ``coefficient * u**monomial_shifts * T**(-(d+1)-sum(p_D)) * prod H_D**p_D``

    in the same gap convention as :func:`spin23_twisted_periods.hepp_gap_map`.
    Mellin monomials common to a lattice group should remain in the external
    ``exponents`` argument; only nonnegative integer shifts belong here.
    """

    normalized_permutation = tuple(int(item) for item in permutation)
    dimension = len(normalized_permutation) - 1
    if dimension < 1 or set(normalized_permutation) != set(range(dimension + 1)):
        raise ValueError("a d-dimensional Hepp permutation must permute 0,...,d")
    shifts = tuple(int(value) for value in monomial_shifts)
    if len(shifts) != dimension or any(value < 0 for value in shifts):
        raise ValueError(
            "monomial_shifts must give one nonnegative integer per cube axis"
        )

    total = SparsePolynomial(
        {
            tuple(int(index < rank) for index in range(dimension)): 1
            for rank in range(dimension + 1)
        },
        dimension=dimension,
    )
    factors: list[PolynomialPower] = [
        PolynomialPower(
            total,
            -(dimension + 1)
            - sum(complex(value) for value in divisor_powers.values()),
        )
    ]
    rank_of = {gap: rank for rank, gap in enumerate(normalized_permutation)}
    for divisor, exponent in divisor_powers.items():
        normalized_divisor = tuple(sorted({int(item) for item in divisor}))
        if not normalized_divisor or any(
            item not in range(dimension + 1) for item in normalized_divisor
        ):
            raise ValueError(
                "divisor indices must form a nonempty subset of 0,...,d"
            )
        ranks = tuple(sorted(rank_of[item] for item in normalized_divisor))
        leading = ranks[0]
        coefficients: dict[MultiIndex, complex] = {}
        for rank in ranks:
            powers = tuple(
                int(leading <= variable < rank)
                for variable in range(dimension)
            )
            coefficients[powers] = coefficients.get(powers, 0.0j) + 1
        factors.append(PolynomialPower(SparsePolynomial(coefficients), exponent))
    for index, shift in enumerate(shifts):
        if shift:
            factors.append(
                PolynomialPower(
                    SparsePolynomial.coordinate(dimension, index), shift
                )
            )
    return FactoredTerm(tuple(factors), coefficient=coefficient)


__all__ = [
    "FactoredMeromorphicCubeIntegral",
    "FactoredPoleError",
    "FactoredPowerSum",
    "FactoredTerm",
    "PolynomialPower",
    "SparsePolynomial",
    "factored_meromorphic_cube_integral",
    "hepp_factored_term",
]
