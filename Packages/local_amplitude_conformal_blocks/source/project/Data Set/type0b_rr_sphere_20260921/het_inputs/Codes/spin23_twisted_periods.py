#!/usr/bin/env python3
r"""Twisted-cycle and Hepp-sector utilities for screened sphere integrals.

This module contains no SO(23)-specific CFT data.  It implements the small
piece of generic integration machinery needed by the first screened
five-point resonance:

* the sine momentum kernel in the convention

  .. math::

     S[\gamma|\sigma]_1=\prod_t\sin\!\left[\pi\left(
       \kappa_{1\gamma_t}+
       \sum_{q>t\atop \sigma^{-1}(\gamma_q)<\sigma^{-1}(\gamma_t)}
       \kappa_{\gamma_t\gamma_q}\right)\right],

  with raw-area KLT pairing ``-L S R``;

* all 24 Hepp sectors of the ordered three-simplex, expressed in terms of
  its four gaps; and

* meromorphic integration of ``prod(u_j**(A_j - 1))*H(u)`` on a cube by
  repeated integration by parts.  Adaptive requests use SymPy derivatives;
  fixed Gauss--Jacobi requests default to the faster factored-polynomial jet
  backend in :mod:`spin23_factored_ibp`.

The measure convention is the literal planar measure ``d(Re z)d(Im z)``.
In particular there is no hidden factor of two in :func:`raw_area_beta_klt`.
At an exact pole of an integration-by-parts denominator the individual terms
are not defined.  The caller must introduce one common complex regulator,
sum all sectors/components, and only then extract the desired Laurent
coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import sys
from typing import Callable, Hashable, Mapping, Sequence

import mpmath as mp
import numpy as np
import sympy as sp


Label = Hashable
Permutation = tuple[Label, ...]
GapPermutation = tuple[int, int, int, int]


def _pair_value(
    kappa: Mapping[tuple[Label, Label], complex],
    first: Label,
    second: Label,
) -> complex:
    """Read a symmetric pair exponent without requiring duplicate keys."""

    direct = (first, second)
    reverse = (second, first)
    if direct in kappa:
        return kappa[direct]
    if reverse in kappa:
        return kappa[reverse]
    raise KeyError(f"missing pair exponent for {first!r}, {second!r}")


def sine_momentum_kernel(
    gamma: Sequence[Label],
    sigma: Sequence[Label],
    kappa: Mapping[tuple[Label, Label], complex],
    *,
    root: Label = 1,
) -> mp.mpc:
    r"""Return the sine momentum kernel ``S[gamma|sigma]_root``.

    ``gamma`` and ``sigma`` must be permutations of the same moving labels.
    Pair exponents can be stored in either orientation in ``kappa``; they are
    treated as symmetric.
    """

    gamma_tuple = tuple(gamma)
    sigma_tuple = tuple(sigma)
    if len(set(gamma_tuple)) != len(gamma_tuple):
        raise ValueError("gamma must not contain repeated labels")
    if len(set(sigma_tuple)) != len(sigma_tuple):
        raise ValueError("sigma must not contain repeated labels")
    if set(gamma_tuple) != set(sigma_tuple):
        raise ValueError("gamma and sigma must permute the same labels")
    if root in gamma_tuple:
        raise ValueError("the KLT root must not be a moving label")

    sigma_position = {label: index for index, label in enumerate(sigma_tuple)}
    result = mp.mpc(1)
    for index, current in enumerate(gamma_tuple):
        argument = mp.mpc(_pair_value(kappa, root, current))
        for later in gamma_tuple[index + 1 :]:
            if sigma_position[later] < sigma_position[current]:
                argument += mp.mpc(_pair_value(kappa, current, later))
        result *= mp.sin(mp.pi * argument)
    return result


def klt_bilinear(
    left_periods: Mapping[Permutation, complex],
    right_periods: Mapping[Permutation, complex],
    kappa: Mapping[tuple[Label, Label], complex],
    *,
    root: Label = 1,
) -> mp.mpc:
    r"""Pair two ordered-period bases with the raw-area KLT convention.

    The returned value is

    .. math::

       -\sum_{\sigma,\gamma}L_\sigma
       S[\gamma|\sigma]_1 R_\gamma.
    """

    if not left_periods or not right_periods:
        raise ValueError("both period bases must be nonempty")
    result = mp.mpc(0)
    for sigma, left in left_periods.items():
        for gamma, right in right_periods.items():
            result -= (
                mp.mpc(left)
                * sine_momentum_kernel(gamma, sigma, kappa, root=root)
                * mp.mpc(right)
            )
    return result


@dataclass(frozen=True)
class RawAreaBetaKLT:
    """The two periods and sine factor in the one-variable KLT identity."""

    left_period: mp.mpc
    right_period: mp.mpc
    sine_kernel: mp.mpc
    antiholomorphic_integer_shift: int

    @property
    def value(self) -> mp.mpc:
        """Return ``-sine_kernel * left_period * right_period``."""

        return -self.sine_kernel * self.left_period * self.right_period


def _integer_shift(value: complex, *, tolerance: float) -> int:
    candidate = mp.mpc(value)
    nearest = int(mp.nint(candidate.real))
    if abs(candidate.imag) > tolerance or abs(candidate.real - nearest) > tolerance:
        raise ValueError(
            "the holomorphic/antiholomorphic power mismatch must be an integer"
        )
    return nearest


def raw_area_beta_klt_components(
    a: complex,
    abar: complex,
    b: complex,
    bbar: complex,
    *,
    antiholomorphic_integer_shift: int | None = None,
    tolerance: float = 1.0e-10,
) -> RawAreaBetaKLT:
    r"""Return the one-moving-point KLT decomposition.

    The holomorphic period is ``B(a,b)``.  If
    ``n = a - abar`` is the integer monodromy mismatch, the right period is
    ``(-1)**n B(abar, 1-abar-bbar)``.  The pair exponent entering the sine
    kernel is the actual power at zero, ``kappa_0=a-1``.
    """

    inferred_shift = _integer_shift(mp.mpc(a) - mp.mpc(abar), tolerance=tolerance)
    if antiholomorphic_integer_shift is None:
        shift = inferred_shift
    else:
        shift = int(antiholomorphic_integer_shift)
        if shift != inferred_shift:
            raise ValueError(
                "antiholomorphic_integer_shift disagrees with a-abar"
            )
    return RawAreaBetaKLT(
        left_period=mp.mpc(mp.beta(a, b)),
        right_period=mp.mpc(((-1) ** shift) * mp.beta(abar, 1 - abar - bbar)),
        sine_kernel=mp.mpc(mp.sin(mp.pi * (mp.mpc(a) - 1))),
        antiholomorphic_integer_shift=shift,
    )


def raw_area_beta_klt(
    a: complex,
    abar: complex,
    b: complex,
    bbar: complex,
    *,
    antiholomorphic_integer_shift: int | None = None,
    tolerance: float = 1.0e-10,
) -> mp.mpc:
    r"""Evaluate the single-valued complex beta integral by raw-area KLT.

    For integer left/right power mismatches this equals

    .. math::

       \pi\frac{\Gamma(a)\Gamma(b)\Gamma(1-\bar a-\bar b)}
       {\Gamma(1-\bar a)\Gamma(1-\bar b)\Gamma(a+b)}.
    """

    return raw_area_beta_klt_components(
        a,
        abar,
        b,
        bbar,
        antiholomorphic_integer_shift=antiholomorphic_integer_shift,
        tolerance=tolerance,
    ).value


HEPP_GAP_PERMUTATIONS: tuple[GapPermutation, ...] = tuple(
    permutations(range(4))
)


def _validate_gap_permutation(permutation: Sequence[int]) -> GapPermutation:
    normalized = tuple(int(item) for item in permutation)
    if len(normalized) != 4 or set(normalized) != set(range(4)):
        raise ValueError("a Hepp gap permutation must permute (0,1,2,3)")
    return normalized  # type: ignore[return-value]


def _validate_cube_point(u: Sequence[float]) -> tuple[float, float, float]:
    normalized = tuple(float(value) for value in u)
    if len(normalized) != 3:
        raise ValueError("the three-simplex Hepp map needs three cube variables")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("cube variables must be finite")
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError("cube variables must lie in [0,1]")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class HeppGapEvaluation:
    r"""One point in a Hepp sector of the ordered three-simplex.

    The gap convention is
    ``g=(X1, X2-X1, X3-X2, 1-X3)``.
    """

    permutation: GapPermutation
    cube_point: tuple[float, float, float]
    q: tuple[float, float, float, float]
    total: float
    gaps: tuple[float, float, float, float]
    simplex_point: tuple[float, float, float]
    jacobian: float


def hepp_gap_map(
    permutation: Sequence[int],
    u: Sequence[float],
) -> HeppGapEvaluation:
    r"""Map ``[0,1]^3`` to one of the 24 ordered-gap Hepp sectors.

    With ``pi=permutation`` and
    ``q=(1,u1,u1*u2,u1*u2*u3)``, the definition is
    ``g[pi[r]]=q[r]/sum(q)``.  Its positive Jacobian is
    ``u1**2*u2/sum(q)**4``.
    """

    pi = _validate_gap_permutation(permutation)
    u1, u2, u3 = _validate_cube_point(u)
    q = (1.0, u1, u1 * u2, u1 * u2 * u3)
    total = sum(q)
    gaps_list = [0.0, 0.0, 0.0, 0.0]
    for rank, gap_index in enumerate(pi):
        gaps_list[gap_index] = q[rank] / total
    gaps = tuple(gaps_list)
    simplex_point = (
        gaps[0],
        gaps[0] + gaps[1],
        gaps[0] + gaps[1] + gaps[2],
    )
    jacobian = u1**2 * u2 / total**4
    return HeppGapEvaluation(
        permutation=pi,
        cube_point=(u1, u2, u3),
        q=q,
        total=total,
        gaps=gaps,  # type: ignore[arg-type]
        simplex_point=simplex_point,
        jacobian=jacobian,
    )


@dataclass(frozen=True)
class HeppDivisorFactorization:
    r"""Monomial times smooth factor for ``L_D=sum_{i in D}g_i``."""

    permutation: GapPermutation
    divisor: tuple[int, ...]
    leading_rank: int
    monomial_exponents: tuple[int, int, int]
    divisor_ranks: tuple[int, ...]

    def smooth_factor(self, u: Sequence[float]) -> float:
        r"""Return ``H_D`` in ``L_D=u**m H_D/T``.

        ``H_D`` is evaluated without division, so it remains regular on all
        cube faces and is at least one on the real unit cube.
        """

        u1, u2, u3 = _validate_cube_point(u)
        variables = (u1, u2, u3)
        total = 0.0
        for rank in self.divisor_ranks:
            term = 1.0
            for variable_index in range(self.leading_rank, rank):
                term *= variables[variable_index]
            total += term
        return total

    def smooth_expression(
        self,
        variables: Sequence[sp.Symbol] | None = None,
    ) -> sp.Expr:
        """Return the exact SymPy expression for ``H_D``."""

        if variables is None:
            normalized = sp.symbols("u1 u2 u3", positive=True)
        else:
            normalized = tuple(variables)
            if len(normalized) != 3:
                raise ValueError("exactly three cube symbols are required")
        result = sp.Integer(0)
        for rank in self.divisor_ranks:
            term = sp.Integer(1)
            for variable_index in range(self.leading_rank, rank):
                term *= normalized[variable_index]
            result += term
        return sp.expand(result)

    def reconstruct(self, u: Sequence[float]) -> float:
        """Evaluate ``u**m H_D/T``, which must equal ``sum_D g``."""

        evaluation = hepp_gap_map(self.permutation, u)
        monomial = 1.0
        for variable, power in zip(evaluation.cube_point, self.monomial_exponents):
            monomial *= variable**power
        return monomial * self.smooth_factor(u) / evaluation.total


def hepp_divisor_factorization(
    permutation: Sequence[int],
    divisor: Sequence[int],
) -> HeppDivisorFactorization:
    r"""Factor ``sum_{i in divisor}g_i`` on a declared Hepp sector.

    If ``ell`` is the first rank whose gap belongs to ``divisor``, then the
    monomial exponents are ``(ell>=1, ell>=2, ell>=3)``.
    """

    pi = _validate_gap_permutation(permutation)
    normalized_divisor = tuple(sorted({int(item) for item in divisor}))
    if not normalized_divisor:
        raise ValueError("a divisor must contain at least one gap")
    if any(item not in range(4) for item in normalized_divisor):
        raise ValueError("divisor gap indices must lie in {0,1,2,3}")
    rank_of = {gap: rank for rank, gap in enumerate(pi)}
    ranks = tuple(sorted(rank_of[gap] for gap in normalized_divisor))
    leading = ranks[0]
    return HeppDivisorFactorization(
        permutation=pi,
        divisor=normalized_divisor,
        leading_rank=leading,
        monomial_exponents=tuple(
            int(leading >= variable_index) for variable_index in (1, 2, 3)
        ),
        divisor_ranks=ranks,
    )


def hepp_monomial_parameters(
    permutation: Sequence[int],
    divisor_powers: Mapping[tuple[int, ...], complex],
) -> tuple[complex, complex, complex]:
    r"""Return the three Mellin parameters ``A_j`` after a Hepp map.

    For ``j=1,2,3``,

    .. math::

       A_j=4-j+\sum_{D:\ell_D\ge j}p_D,

    so the mapped measure has monomial part
    ``prod(u_j**(A_j-1))``.
    """

    pi = _validate_gap_permutation(permutation)
    parameters: list[complex] = [3, 2, 1]
    for divisor, power in divisor_powers.items():
        factorization = hepp_divisor_factorization(pi, divisor)
        for index, monomial_power in enumerate(
            factorization.monomial_exponents
        ):
            if monomial_power:
                parameters[index] += power
    return tuple(parameters)  # type: ignore[return-value]


@dataclass(frozen=True)
class HeppSectorFactorization:
    """Symbolic ``prod(u**(A-1))*smooth_part`` for one Hepp sector."""

    permutation: GapPermutation
    variables: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    mellin_parameters: tuple[sp.Expr, sp.Expr, sp.Expr]
    smooth_part: sp.Expr

    @property
    def expression(self) -> sp.Expr:
        result = self.smooth_part
        for variable, parameter in zip(self.variables, self.mellin_parameters):
            result *= variable ** (parameter - 1)
        return result


def hepp_sector_factorization(
    permutation: Sequence[int],
    divisor_powers: Mapping[tuple[int, ...], complex],
    *,
    variables: Sequence[sp.Symbol] | None = None,
) -> HeppSectorFactorization:
    r"""Factor a product of gap divisors together with the simplex measure.

    For ``prod_D (sum_{i in D}g_i)**p_D`` the returned smooth factor is

    ``T**(-4-sum(p_D))*prod_D H_D**p_D``.
    """

    pi = _validate_gap_permutation(permutation)
    if variables is None:
        cube_variables = sp.symbols("u1 u2 u3", positive=True)
    else:
        cube_variables = tuple(variables)
        if len(cube_variables) != 3:
            raise ValueError("exactly three cube symbols are required")
    u1, u2, u3 = cube_variables
    total = 1 + u1 + u1 * u2 + u1 * u2 * u3
    total_power = sum((sp.sympify(power) for power in divisor_powers.values()), sp.Integer(0))
    smooth = total ** (-4 - total_power)
    for divisor, power in divisor_powers.items():
        factorization = hepp_divisor_factorization(pi, divisor)
        smooth *= factorization.smooth_expression(cube_variables) ** sp.sympify(power)
    parameters = tuple(
        sp.sympify(parameter)
        for parameter in hepp_monomial_parameters(pi, divisor_powers)
    )
    return HeppSectorFactorization(
        permutation=pi,
        variables=cube_variables,  # type: ignore[arg-type]
        mellin_parameters=parameters,  # type: ignore[arg-type]
        smooth_part=sp.factor(smooth),
    )


@dataclass(frozen=True)
class HeppSimplexIntegral:
    """Numerical sum and individual values of all 24 Hepp sectors."""

    value: complex
    absolute_error: float
    sector_values: tuple[complex, ...]
    sector_errors: tuple[float, ...]
    status: str
    subdivisions: int


def integrate_simplex_by_hepp(
    integrand: Callable[[tuple[float, float, float, float]], complex],
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-11,
    rule: str = "genz-malik",
    max_subdivisions: int = 10_000,
) -> HeppSimplexIntegral:
    r"""Integrate a function of the four gaps over ``0<X1<X2<X3<1``.

    All sectors are evaluated as one vector-valued SciPy cubature, which
    forces a common adaptive subdivision and is considerably cheaper than 24
    independent integrations.
    """

    from scipy.integrate import cubature

    sector_count = len(HEPP_GAP_PERMUTATIONS)

    def packed(points: np.ndarray) -> np.ndarray:
        values = np.empty((points.shape[0], sector_count), dtype=np.complex128)
        u1, u2, u3 = points.T
        q = np.column_stack(
            (np.ones(points.shape[0]), u1, u1 * u2, u1 * u2 * u3)
        )
        total = np.sum(q, axis=1)
        normalized_q = q / total[:, None]
        jacobian = u1**2 * u2 / total**4
        for sector_index, pi in enumerate(HEPP_GAP_PERMUTATIONS):
            gaps = np.empty_like(normalized_q)
            gaps[:, np.asarray(pi)] = normalized_q
            values[:, sector_index] = np.fromiter(
                (complex(integrand(tuple(row))) for row in gaps),
                dtype=np.complex128,
                count=points.shape[0],
            ) * jacobian
        return np.concatenate((values.real, values.imag), axis=1)

    result = cubature(
        packed,
        np.zeros(3),
        np.ones(3),
        rule=rule,
        rtol=rtol,
        atol=atol,
        max_subdivisions=max_subdivisions,
    )
    estimate = np.asarray(result.estimate, dtype=float)
    error = np.asarray(result.error, dtype=float)
    sector_values_array = estimate[:sector_count] + 1j * estimate[sector_count:]
    sector_errors_array = np.hypot(
        error[:sector_count], error[sector_count:]
    )
    return HeppSimplexIntegral(
        value=complex(np.sum(sector_values_array)),
        absolute_error=float(np.sum(sector_errors_array)),
        sector_values=tuple(complex(value) for value in sector_values_array),
        sector_errors=tuple(float(value) for value in sector_errors_array),
        status=str(result.status),
        subdivisions=int(result.subdivisions),
    )


class MeromorphicPoleError(ArithmeticError):
    """An IBP coefficient hit an exact Mellin pole before sector summation."""


@dataclass(frozen=True)
class MeromorphicCubeIntegral:
    """Value and conservative numerical error from meromorphic cube IBP."""

    value: complex | np.ndarray
    absolute_error: float | np.ndarray
    cubature_calls: int
    subdivisions: int
    converged: bool


@dataclass(frozen=True)
class OrderedSimplexTerm:
    r"""One monomial in gap divisors on ``0<X1<X2<X3<1``.

    The represented function is

    ``coefficient * prod_D (sum_{i in D} g_i)**divisor_powers[D]``.

    Every difference among ``0,X1,X2,X3,1`` is a sum of consecutive gaps,
    so this form directly represents Koba--Nielsen powers as well as the
    integer shifts supplied by fermion Wick contractions.
    """

    divisor_powers: Mapping[tuple[int, ...], complex]
    coefficient: complex = 1.0 + 0.0j
    label: str = ""

    def __post_init__(self) -> None:
        for divisor in self.divisor_powers:
            normalized = tuple(sorted({int(item) for item in divisor}))
            if not normalized or any(item not in range(4) for item in normalized):
                raise ValueError(
                    "each divisor must be a nonempty subset of gap indices 0..3"
                )


@dataclass(frozen=True)
class OrderedSimplexIntegral:
    """Value, error budget, and per-Hepp-sector period diagnostics."""

    value: complex
    absolute_error: float
    sector_values: tuple[complex, ...]
    sector_errors: tuple[float, ...]
    cubature_calls: int
    subdivisions: int
    converged: bool


@dataclass(frozen=True)
class OrderedSimplexVectorIntegral:
    """Several simplex integrals evaluated with common Hepp cubatures."""

    values: tuple[complex, ...]
    absolute_errors: tuple[float, ...]
    sector_values: tuple[tuple[complex, ...], ...]
    sector_errors: tuple[tuple[float, ...], ...]
    cubature_calls: int
    subdivisions: int
    converged: bool


@dataclass
class _CubeAccumulator:
    value: np.ndarray
    error: np.ndarray
    cubature_calls: int = 0
    subdivisions: int = 0
    converged: bool = True

    @classmethod
    def zeros(cls, output_dimension: int) -> "_CubeAccumulator":
        return cls(
            value=np.zeros(output_dimension, dtype=np.complex128),
            error=np.zeros(output_dimension, dtype=float),
        )

    def add_scaled(self, other: "_CubeAccumulator", coefficient: complex) -> None:
        self.value += coefficient * other.value
        self.error += abs(coefficient) * other.error
        self.cubature_calls += other.cubature_calls
        self.subdivisions += other.subdivisions
        self.converged = self.converged and other.converged


def _normalize_expressions(
    expression: sp.Expr | sp.MatrixBase | Sequence[sp.Expr],
) -> tuple[tuple[sp.Expr, ...], bool]:
    if isinstance(expression, sp.MatrixBase):
        return tuple(sp.sympify(item) for item in expression), False
    if isinstance(expression, Sequence) and not isinstance(expression, (str, bytes)):
        return tuple(sp.sympify(item) for item in expression), False
    return (sp.sympify(expression),), True


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


def _base_cube_cubature(
    expressions: tuple[sp.Expr, ...],
    variables: tuple[sp.Symbol, ...],
    exponents: tuple[complex, ...],
    *,
    rtol: float,
    atol: float,
    rule: str,
    max_subdivisions: int,
) -> _CubeAccumulator:
    output_dimension = len(expressions)
    if not variables:
        values = np.array(
            [complex(sp.N(expression, 17)) for expression in expressions],
            dtype=np.complex128,
        )
        return _CubeAccumulator(values, np.zeros(output_dimension, dtype=float))

    # Pfaffian-derived five-point components can contain deeply nested
    # rational expressions.  Common-subexpression elimination prevents
    # reevaluating their repeated divisor combinations.  The generated AST
    # for the all-singlet component is nevertheless deeper than Python's
    # conservative default recursion limit, so temporarily raise that limit
    # for compilation only and restore it before numerical evaluation.
    old_recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(old_recursion_limit, 10_000))
        functions = tuple(
            sp.lambdify(
                variables,
                expression,
                modules="numpy",
                cse=True,
                docstring_limit=0,
            )
            for expression in expressions
        )
    finally:
        sys.setrecursionlimit(old_recursion_limit)

    if rule.startswith("gauss-jacobi-"):
        try:
            quadrature_order = int(rule.rsplit("-", 1)[1])
        except ValueError as error:
            raise ValueError(
                "a fixed Gauss--Jacobi rule must be named gauss-jacobi-N"
            ) from error
        if quadrature_order < 4:
            raise ValueError("Gauss--Jacobi order must be at least four")

        def fixed(order: int) -> np.ndarray:
            nodes_by_axis: list[np.ndarray] = []
            weights_by_axis: list[np.ndarray] = []
            for exponent in exponents:
                if exponent.real <= 0:
                    raise ValueError(
                        "Gauss--Jacobi terminal exponents must have positive real part"
                    )
                if abs(exponent.imag) < 1.0e-14:
                    from scipy.special import roots_jacobi

                    real_nodes, real_weights = roots_jacobi(
                        order, 0.0, exponent.real - 1.0
                    )
                    nodes_by_axis.append((real_nodes + 1.0) / 2.0)
                    weights_by_axis.append(
                        real_weights * 2.0 ** (-exponent.real)
                    )
                    continue

                # Golub--Welsch for the complex Jacobi weight u**(A-1).
                # A complex-symmetric Jacobi matrix has transpose-orthogonal
                # eigenvectors.  Its first components therefore give the
                # Gaussian weights without discarding Im(A), unlike a real
                # Jacobi rule followed by the slowly convergent factor
                # u**(i Im(A)).
                beta = exponent - 1.0
                matrix = np.zeros((order, order), dtype=np.complex128)
                for degree in range(order):
                    if degree == 0 and abs(beta) < 1.0e-14:
                        diagonal = 0.0j
                    else:
                        diagonal = beta**2 / (
                            (2 * degree + beta)
                            * (2 * degree + beta + 2)
                        )
                    matrix[degree, degree] = diagonal
                for degree in range(1, order):
                    off_diagonal = 2.0 / (2 * degree + beta) * np.sqrt(
                        degree
                        * degree
                        * (degree + beta)
                        * (degree + beta)
                        / (
                            (2 * degree + beta - 1)
                            * (2 * degree + beta + 1)
                        )
                    )
                    matrix[degree - 1, degree] = off_diagonal
                    matrix[degree, degree - 1] = off_diagonal
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                unit_nodes = (eigenvalues + 1.0) / 2.0
                unit_weights = np.empty(order, dtype=np.complex128)
                for node_index in range(order):
                    vector = eigenvectors[:, node_index]
                    bilinear_norm = np.sum(vector * vector)
                    if abs(bilinear_norm) < 1.0e-13:
                        raise ArithmeticError(
                            "complex Gauss--Jacobi eigenvector is self-orthogonal"
                        )
                    unit_weights[node_index] = (
                        vector[0] ** 2 / (exponent * bilinear_norm)
                    )
                nodes_by_axis.append(unit_nodes)
                weights_by_axis.append(unit_weights)

            node_mesh = np.meshgrid(*nodes_by_axis, indexing="ij")
            weight_mesh = np.meshgrid(*weights_by_axis, indexing="ij")
            arguments = tuple(axis.reshape(-1) for axis in node_mesh)
            total_weight = np.ones(arguments[0].shape, dtype=np.complex128)
            for argument, axis_weight in zip(
                arguments,
                weight_mesh,
            ):
                total_weight *= axis_weight.reshape(-1)

            values = np.empty(output_dimension, dtype=np.complex128)
            for output_index, function in enumerate(functions):
                raw = np.asarray(function(*arguments), dtype=np.complex128)
                if raw.ndim == 0:
                    raw = np.full(arguments[0].shape, raw, dtype=np.complex128)
                values[output_index] = np.sum(
                    total_weight * np.broadcast_to(raw, arguments[0].shape)
                )
            return values

        low = fixed(quadrature_order)
        high = fixed(quadrature_order + 4)
        errors = np.abs(high - low)
        tolerances = atol + rtol * np.abs(high)
        return _CubeAccumulator(
            value=high,
            error=errors,
            cubature_calls=2,
            subdivisions=0,
            converged=bool(np.all(errors <= tolerances)),
        )

    from scipy.integrate import cubature

    endpoint_smoothing_power = 4.0

    def packed(points: np.ndarray) -> np.ndarray:
        integration_variables = tuple(
            points[:, index] for index in range(points.shape[1])
        )
        arguments = tuple(
            variable ** (endpoint_smoothing_power / exponent.real)
            for variable, exponent in zip(integration_variables, exponents)
        )
        weight = np.ones(points.shape[0], dtype=np.complex128)
        for variable, exponent in zip(integration_variables, exponents):
            real_part = exponent.real
            weight *= (
                np.exp(
                    (
                        endpoint_smoothing_power * exponent / real_part
                        - 1.0
                    )
                    * np.log(variable)
                )
                * endpoint_smoothing_power
                / real_part
            )
        values = np.empty((points.shape[0], output_dimension), dtype=np.complex128)
        for output_index, function in enumerate(functions):
            raw = np.asarray(function(*arguments), dtype=np.complex128)
            if raw.ndim == 0:
                raw = np.full(points.shape[0], raw, dtype=np.complex128)
            values[:, output_index] = weight * np.broadcast_to(raw, (points.shape[0],))
        return np.concatenate((values.real, values.imag), axis=1)

    effective_rule = "gk21" if len(variables) == 1 and rule == "genz-malik" else rule
    result = cubature(
        packed,
        np.zeros(len(variables)),
        np.ones(len(variables)),
        rule=effective_rule,
        rtol=rtol,
        atol=atol,
        max_subdivisions=max_subdivisions,
    )
    estimate = np.asarray(result.estimate, dtype=float)
    error = np.asarray(result.error, dtype=float)
    values = estimate[:output_dimension] + 1j * estimate[output_dimension:]
    errors = np.hypot(error[:output_dimension], error[output_dimension:])
    return _CubeAccumulator(
        value=values,
        error=errors,
        cubature_calls=1,
        subdivisions=int(result.subdivisions),
        converged=str(result.status) == "converged",
    )


def meromorphic_cube_integral(
    expression: sp.Expr | sp.MatrixBase | Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    exponents: Sequence[complex],
    *,
    substitutions: Mapping[sp.Symbol, complex] | None = None,
    ibp_orders: Sequence[int] | None = None,
    convergence_margin: float = 1.0e-10,
    pole_tolerance: float = 1.0e-13,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-11,
    rule: str = "genz-malik",
    max_subdivisions: int = 10_000,
) -> MeromorphicCubeIntegral:
    r"""Meromorphically continue a Mellin-weighted cube integral by IBP.

    The represented integral is

    .. math::

       J(A,H)=\int_{[0,1]^d}\prod_i u_i^{A_i-1}H(u)\,d^du.

    For a troublesome coordinate and an integer ``N`` with
    ``Re(A_i+N)>0``, the implementation recursively applies

    .. math::

       J=\sum_{k=0}^{N-1}\frac{(-1)^k}{(A_i)_{k+1}}
       J_{d-1}[(\partial_i^kH)_{u_i=1}]
       +\frac{(-1)^N}{(A_i)_N}J[A_i+N,\partial_i^NH].

    A scalar expression returns scalar diagnostics.  A SymPy matrix or a
    sequence of expressions is differentiated together and integrated by
    one vector cubature per terminal term.
    """

    expressions, scalar_output = _normalize_expressions(expression)
    cube_variables = tuple(variables)
    mellin_exponents = tuple(complex(value) for value in exponents)
    if len(cube_variables) != len(mellin_exponents):
        raise ValueError("variables and exponents must have equal length")
    if len(set(cube_variables)) != len(cube_variables):
        raise ValueError("cube variables must be distinct")
    if substitutions:
        expressions = tuple(item.subs(substitutions) for item in expressions)
    unresolved = set().union(*(item.free_symbols for item in expressions)) - set(cube_variables)
    if unresolved:
        names = ", ".join(sorted(str(symbol) for symbol in unresolved))
        raise ValueError(f"smooth expression has unsubstituted symbols: {names}")

    if ibp_orders is None:
        orders = tuple(
            _default_ibp_order(value, margin=convergence_margin)
            for value in mellin_exponents
        )
    else:
        orders = tuple(int(value) for value in ibp_orders)
        if len(orders) != len(cube_variables) or any(value < 0 for value in orders):
            raise ValueError("ibp_orders must give one nonnegative integer per variable")
        if any(
            (exponent + order).real <= convergence_margin
            for exponent, order in zip(mellin_exponents, orders)
        ):
            raise ValueError("the requested IBP order does not reach a convergent exponent")

    output_dimension = len(expressions)

    def recurse(
        current_expressions: tuple[sp.Expr, ...],
        current_variables: tuple[sp.Symbol, ...],
        current_exponents: tuple[complex, ...],
        current_orders: tuple[int, ...],
    ) -> _CubeAccumulator:
        try:
            coordinate = next(
                index for index, order in enumerate(current_orders) if order > 0
            )
        except StopIteration:
            if any(
                exponent.real <= convergence_margin
                for exponent in current_exponents
            ):
                raise ValueError("a terminal cubature still has a nonconvergent exponent")
            return _base_cube_cubature(
                current_expressions,
                current_variables,
                current_exponents,
                rtol=rtol,
                atol=atol,
                rule=rule,
                max_subdivisions=max_subdivisions,
            )

        variable = current_variables[coordinate]
        exponent = current_exponents[coordinate]
        order = current_orders[coordinate]
        accumulator = _CubeAccumulator.zeros(output_dimension)
        derivatives = current_expressions
        for derivative_order in range(order):
            denominator = _rising(exponent, derivative_order + 1)
            if abs(denominator) <= pole_tolerance:
                raise MeromorphicPoleError(
                    "an exact IBP pole was encountered; introduce one common "
                    "complex regulator and sum sectors before Laurent extraction"
                )
            boundary_expressions = tuple(
                item.subs(variable, 1) for item in derivatives
            )
            boundary = recurse(
                boundary_expressions,
                current_variables[:coordinate] + current_variables[coordinate + 1 :],
                current_exponents[:coordinate] + current_exponents[coordinate + 1 :],
                current_orders[:coordinate] + current_orders[coordinate + 1 :],
            )
            accumulator.add_scaled(
                boundary, ((-1) ** derivative_order) / denominator
            )
            derivatives = tuple(sp.diff(item, variable) for item in derivatives)

        denominator = _rising(exponent, order)
        if abs(denominator) <= pole_tolerance:
            raise MeromorphicPoleError(
                "an exact IBP pole was encountered; introduce one common "
                "complex regulator and sum sectors before Laurent extraction"
            )
        shifted_exponents = list(current_exponents)
        shifted_exponents[coordinate] += order
        residual_orders = list(current_orders)
        residual_orders[coordinate] = 0
        residual = recurse(
            derivatives,
            current_variables,
            tuple(shifted_exponents),
            tuple(residual_orders),
        )
        accumulator.add_scaled(residual, ((-1) ** order) / denominator)
        return accumulator

    result = recurse(expressions, cube_variables, mellin_exponents, orders)
    if scalar_output:
        return MeromorphicCubeIntegral(
            value=complex(result.value[0]),
            absolute_error=float(result.error[0]),
            cubature_calls=result.cubature_calls,
            subdivisions=result.subdivisions,
            converged=result.converged,
        )
    return MeromorphicCubeIntegral(
        value=result.value.copy(),
        absolute_error=result.error.copy(),
        cubature_calls=result.cubature_calls,
        subdivisions=result.subdivisions,
        converged=result.converged,
    )


def _integer_shift_vector(
    parameters: Sequence[complex],
    reference: Sequence[complex],
    *,
    tolerance: float,
) -> tuple[int, ...] | None:
    shifts: list[int] = []
    for parameter, base in zip(parameters, reference):
        difference = complex(parameter) - complex(base)
        nearest = int(round(difference.real))
        if abs(difference.imag) > tolerance or abs(difference.real - nearest) > tolerance:
            return None
        shifts.append(nearest)
    return tuple(shifts)


def _select_period_backend(backend: str, rule: str) -> bool:
    """Return whether an ordered-period request should use factored IBP."""

    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "symbolic", "factored"}:
        raise ValueError("backend must be one of 'auto', 'symbolic', or 'factored'")
    if normalized == "symbolic":
        return False
    fixed_rule = rule.startswith("gauss-jacobi-")
    if normalized == "factored" and not fixed_rule:
        raise ValueError(
            "the factored period backend currently requires a gauss-jacobi-N rule"
        )
    return fixed_rule


def _integrate_term_components_factored(
    components: tuple[tuple[OrderedSimplexTerm, ...], ...],
    *,
    integer_shift_tolerance: float,
    convergence_margin: float,
    pole_tolerance: float,
    rtol: float,
    atol: float,
    rule: str,
) -> OrderedSimplexVectorIntegral:
    """Non-symbolic Hepp integration for the fixed Gauss--Jacobi path."""

    from spin23_factored_ibp import (
        FactoredPoleError,
        FactoredPowerSum,
        FactoredTerm as NumericalFactoredTerm,
        factored_meromorphic_cube_integral,
        hepp_factored_term,
    )

    output_dimension = len(components)
    sector_values: list[tuple[complex, ...]] = []
    sector_errors: list[tuple[float, ...]] = []
    cubature_calls = 0
    converged = True

    for permutation in HEPP_GAP_PERMUTATIONS:
        factorized: list[
            tuple[int, OrderedSimplexTerm, tuple[complex, complex, complex]]
        ] = []
        for component_index, component in enumerate(components):
            for term in component:
                parameters = tuple(
                    complex(value)
                    for value in hepp_monomial_parameters(
                        permutation, term.divisor_powers
                    )
                )
                factorized.append((component_index, term, parameters))

        groups: list[
            list[tuple[int, OrderedSimplexTerm, tuple[complex, complex, complex]]]
        ] = []
        for item in factorized:
            for group in groups:
                if _integer_shift_vector(
                    item[2],
                    group[0][2],
                    tolerance=integer_shift_tolerance,
                ) is not None:
                    group.append(item)
                    break
            else:
                groups.append([item])

        value = np.zeros(output_dimension, dtype=np.complex128)
        error = np.zeros(output_dimension, dtype=float)
        for group in groups:
            base_parameters = tuple(
                min(
                    (item[2][index] for item in group),
                    key=lambda entry: entry.real,
                )
                for index in range(3)
            )
            expression_terms: list[list[NumericalFactoredTerm]] = [
                [] for _ in range(output_dimension)
            ]
            for component_index, term, parameters in group:
                shifts = _integer_shift_vector(
                    parameters,
                    base_parameters,
                    tolerance=integer_shift_tolerance,
                )
                if shifts is None or any(shift < 0 for shift in shifts):
                    raise RuntimeError("failed to choose a common Mellin base")
                expression_terms[component_index].append(
                    hepp_factored_term(
                        permutation,
                        term.divisor_powers,
                        monomial_shifts=shifts,
                        coefficient=term.coefficient,
                    )
                )
            # Keep a literal zero component in the vector request.  This lets
            # all projections share the same nodes even if a Mellin lattice
            # group occurs in only some of them.
            expressions = tuple(
                FactoredPowerSum(
                    tuple(terms)
                    if terms
                    else (NumericalFactoredTerm(coefficient=0.0j),)
                )
                for terms in expression_terms
            )
            try:
                result = factored_meromorphic_cube_integral(
                    expressions,
                    base_parameters,
                    convergence_margin=convergence_margin,
                    pole_tolerance=pole_tolerance,
                    rtol=rtol,
                    atol=atol / len(HEPP_GAP_PERMUTATIONS),
                    rule=rule,
                )
            except FactoredPoleError as error_at_pole:
                raise MeromorphicPoleError(str(error_at_pole)) from error_at_pole
            value += np.asarray(result.value, dtype=np.complex128)
            error += np.asarray(result.absolute_error, dtype=float)
            cubature_calls += result.quadrature_calls
            converged = converged and result.converged
        sector_values.append(tuple(complex(entry) for entry in value))
        sector_errors.append(tuple(float(entry) for entry in error))

    values = tuple(
        sum((sector[component] for sector in sector_values), 0.0j)
        for component in range(output_dimension)
    )
    errors = tuple(
        sum(sector[component] for sector in sector_errors)
        for component in range(output_dimension)
    )
    return OrderedSimplexVectorIntegral(
        values=values,
        absolute_errors=errors,
        sector_values=tuple(sector_values),
        sector_errors=tuple(sector_errors),
        cubature_calls=cubature_calls,
        subdivisions=0,
        converged=converged,
    )


def integrate_terms(
    terms: Sequence[OrderedSimplexTerm],
    *,
    variables: Sequence[sp.Symbol] | None = None,
    integer_shift_tolerance: float = 1.0e-10,
    convergence_margin: float = 1.0e-10,
    pole_tolerance: float = 1.0e-13,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-10,
    rule: str = "gk15",
    max_subdivisions: int = 10_000,
    backend: str = "auto",
) -> OrderedSimplexIntegral:
    r"""Integrate a sum of gap-divisor terms over the ordered simplex.

    The function performs all 24 Hepp maps and meromorphically integrates
    every resulting cube.  Within a sector, terms whose Mellin parameters
    differ only by integer vectors are combined *before* integration by
    parts: the componentwise-minimum parameter is factored out and the
    nonnegative integer shifts are absorbed into one smooth SymPy
    expression.  This preserves cancellations among Wick-contraction terms.

    Exact Mellin poles deliberately raise :class:`MeromorphicPoleError`.
    Regulate all input powers coherently, call this function on the regulated
    sum, and take the Laurent coefficient only after sector summation.

    ``backend="auto"`` uses the factored numerical-jet implementation for a
    fixed ``gauss-jacobi-N`` rule and the generic symbolic implementation for
    adaptive rules.  Pass ``"symbolic"`` to force the reference path or
    ``"factored"`` to require the fast fixed-rule path.
    """

    normalized_terms = tuple(terms)
    if not normalized_terms:
        raise ValueError("at least one ordered-simplex term is required")
    if variables is None:
        cube_variables = sp.symbols("u1 u2 u3", positive=True)
    else:
        cube_variables = tuple(variables)
        if len(cube_variables) != 3:
            raise ValueError("exactly three cube symbols are required")

    if _select_period_backend(backend, rule):
        vector_result = _integrate_term_components_factored(
            (normalized_terms,),
            integer_shift_tolerance=integer_shift_tolerance,
            convergence_margin=convergence_margin,
            pole_tolerance=pole_tolerance,
            rtol=rtol,
            atol=atol,
            rule=rule,
        )
        return OrderedSimplexIntegral(
            value=vector_result.values[0],
            absolute_error=vector_result.absolute_errors[0],
            sector_values=tuple(sector[0] for sector in vector_result.sector_values),
            sector_errors=tuple(sector[0] for sector in vector_result.sector_errors),
            cubature_calls=vector_result.cubature_calls,
            subdivisions=vector_result.subdivisions,
            converged=vector_result.converged,
        )

    sector_values: list[complex] = []
    sector_errors: list[float] = []
    cubature_calls = 0
    subdivisions = 0
    converged = True

    for permutation in HEPP_GAP_PERMUTATIONS:
        factorized = [
            (
                term,
                hepp_sector_factorization(
                    permutation,
                    term.divisor_powers,
                    variables=cube_variables,
                ),
            )
            for term in normalized_terms
        ]

        groups: list[list[tuple[OrderedSimplexTerm, HeppSectorFactorization]]] = []
        for item in factorized:
            parameters = tuple(complex(value) for value in item[1].mellin_parameters)
            for group in groups:
                reference = tuple(
                    complex(value) for value in group[0][1].mellin_parameters
                )
                if _integer_shift_vector(
                    parameters,
                    reference,
                    tolerance=integer_shift_tolerance,
                ) is not None:
                    group.append(item)
                    break
            else:
                groups.append([item])

        sector_value = 0.0j
        sector_error = 0.0
        for group in groups:
            all_parameters = [
                tuple(complex(value) for value in factor.mellin_parameters)
                for _, factor in group
            ]
            base_parameters = tuple(
                min(
                    (parameters[index] for parameters in all_parameters),
                    key=lambda value: value.real,
                )
                for index in range(3)
            )
            combined_smooth = sp.Integer(0)
            for (term, factor), parameters in zip(group, all_parameters):
                shifts = _integer_shift_vector(
                    parameters,
                    base_parameters,
                    tolerance=integer_shift_tolerance,
                )
                if shifts is None or any(shift < 0 for shift in shifts):
                    raise RuntimeError("failed to choose a common Mellin base")
                shifted_smooth = factor.smooth_part
                for variable, shift in zip(cube_variables, shifts):
                    shifted_smooth *= variable**shift
                combined_smooth += sp.sympify(term.coefficient) * shifted_smooth
            combined_smooth = sp.factor_terms(combined_smooth)
            if combined_smooth == 0:
                continue
            result = meromorphic_cube_integral(
                combined_smooth,
                cube_variables,
                base_parameters,
                convergence_margin=convergence_margin,
                pole_tolerance=pole_tolerance,
                rtol=rtol,
                atol=atol / len(HEPP_GAP_PERMUTATIONS),
                rule=rule,
                max_subdivisions=max_subdivisions,
            )
            sector_value += complex(result.value)
            sector_error += float(result.absolute_error)
            cubature_calls += result.cubature_calls
            subdivisions += result.subdivisions
            converged = converged and result.converged
        sector_values.append(sector_value)
        sector_errors.append(sector_error)

    return OrderedSimplexIntegral(
        value=sum(sector_values, 0.0j),
        absolute_error=sum(sector_errors),
        sector_values=tuple(sector_values),
        sector_errors=tuple(sector_errors),
        cubature_calls=cubature_calls,
        subdivisions=subdivisions,
        converged=converged,
    )


def integrate_term_components(
    components: Sequence[Sequence[OrderedSimplexTerm]],
    *,
    variables: Sequence[sp.Symbol] | None = None,
    integer_shift_tolerance: float = 1.0e-10,
    convergence_margin: float = 1.0e-10,
    pole_tolerance: float = 1.0e-13,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-10,
    rule: str = "gk15",
    max_subdivisions: int = 10_000,
    backend: str = "auto",
) -> OrderedSimplexVectorIntegral:
    r"""Integrate several term sums with shared symbolic IBP and cubature.

    Each entry of ``components`` is one independent sum of
    :class:`OrderedSimplexTerm` objects.  Terms on the same integer-shifted
    Mellin lattice are combined before IBP, while all requested component
    expressions are differentiated and sampled as one vector.  This is the
    efficient path for evaluating many SO(23) tensor projections with the
    same Koba--Nielsen twist.  With ``backend="auto"``, fixed
    ``gauss-jacobi-N`` rules use numerical jets of factored polynomial powers;
    adaptive rules retain the generic symbolic implementation.
    """

    normalized = tuple(tuple(component) for component in components)
    if not normalized or any(not component for component in normalized):
        raise ValueError("every requested simplex component must contain a term")
    if variables is None:
        cube_variables = sp.symbols("u1 u2 u3", positive=True)
    else:
        cube_variables = tuple(variables)
        if len(cube_variables) != 3:
            raise ValueError("exactly three cube symbols are required")
    if _select_period_backend(backend, rule):
        return _integrate_term_components_factored(
            normalized,
            integer_shift_tolerance=integer_shift_tolerance,
            convergence_margin=convergence_margin,
            pole_tolerance=pole_tolerance,
            rtol=rtol,
            atol=atol,
            rule=rule,
        )
    output_dimension = len(normalized)
    sector_values: list[tuple[complex, ...]] = []
    sector_errors: list[tuple[float, ...]] = []
    cubature_calls = 0
    subdivisions = 0
    converged = True

    for permutation in HEPP_GAP_PERMUTATIONS:
        factorized: list[
            tuple[int, OrderedSimplexTerm, HeppSectorFactorization]
        ] = []
        for component_index, component in enumerate(normalized):
            for term in component:
                factorized.append(
                    (
                        component_index,
                        term,
                        hepp_sector_factorization(
                            permutation,
                            term.divisor_powers,
                            variables=cube_variables,
                        ),
                    )
                )

        groups: list[
            list[tuple[int, OrderedSimplexTerm, HeppSectorFactorization]]
        ] = []
        for item in factorized:
            parameters = tuple(
                complex(value) for value in item[2].mellin_parameters
            )
            for group in groups:
                reference = tuple(
                    complex(value) for value in group[0][2].mellin_parameters
                )
                if _integer_shift_vector(
                    parameters,
                    reference,
                    tolerance=integer_shift_tolerance,
                ) is not None:
                    group.append(item)
                    break
            else:
                groups.append([item])

        value = np.zeros(output_dimension, dtype=np.complex128)
        error = np.zeros(output_dimension, dtype=float)
        for group in groups:
            all_parameters = [
                tuple(complex(entry) for entry in factor.mellin_parameters)
                for _, _, factor in group
            ]
            base_parameters = tuple(
                min(
                    (parameters[index] for parameters in all_parameters),
                    key=lambda entry: entry.real,
                )
                for index in range(3)
            )
            expressions = [sp.Integer(0) for _ in range(output_dimension)]
            for (component_index, term, factor), parameters in zip(
                group, all_parameters
            ):
                shifts = _integer_shift_vector(
                    parameters,
                    base_parameters,
                    tolerance=integer_shift_tolerance,
                )
                if shifts is None or any(shift < 0 for shift in shifts):
                    raise RuntimeError("failed to choose a common Mellin base")
                shifted = factor.smooth_part
                for variable, shift in zip(cube_variables, shifts):
                    shifted *= variable**shift
                expressions[component_index] += (
                    sp.sympify(term.coefficient) * shifted
                )
            expressions = [sp.factor_terms(expression) for expression in expressions]
            result = meromorphic_cube_integral(
                expressions,
                cube_variables,
                base_parameters,
                convergence_margin=convergence_margin,
                pole_tolerance=pole_tolerance,
                rtol=rtol,
                atol=atol / len(HEPP_GAP_PERMUTATIONS),
                rule=rule,
                max_subdivisions=max_subdivisions,
            )
            value += np.asarray(result.value, dtype=np.complex128)
            error += np.asarray(result.absolute_error, dtype=float)
            cubature_calls += result.cubature_calls
            subdivisions += result.subdivisions
            converged = converged and result.converged
        sector_values.append(tuple(complex(entry) for entry in value))
        sector_errors.append(tuple(float(entry) for entry in error))

    values = tuple(
        sum((sector[component] for sector in sector_values), 0.0j)
        for component in range(output_dimension)
    )
    errors = tuple(
        sum(sector[component] for sector in sector_errors)
        for component in range(output_dimension)
    )
    return OrderedSimplexVectorIntegral(
        values=values,
        absolute_errors=errors,
        sector_values=tuple(sector_values),
        sector_errors=tuple(sector_errors),
        cubature_calls=cubature_calls,
        subdivisions=subdivisions,
        converged=converged,
    )


__all__ = [
    "HEPP_GAP_PERMUTATIONS",
    "HeppDivisorFactorization",
    "HeppGapEvaluation",
    "HeppSectorFactorization",
    "HeppSimplexIntegral",
    "MeromorphicCubeIntegral",
    "MeromorphicPoleError",
    "OrderedSimplexIntegral",
    "OrderedSimplexTerm",
    "OrderedSimplexVectorIntegral",
    "RawAreaBetaKLT",
    "hepp_divisor_factorization",
    "hepp_gap_map",
    "hepp_monomial_parameters",
    "hepp_sector_factorization",
    "integrate_simplex_by_hepp",
    "integrate_terms",
    "integrate_term_components",
    "klt_bilinear",
    "meromorphic_cube_integral",
    "raw_area_beta_klt",
    "raw_area_beta_klt_components",
    "sine_momentum_kernel",
]
