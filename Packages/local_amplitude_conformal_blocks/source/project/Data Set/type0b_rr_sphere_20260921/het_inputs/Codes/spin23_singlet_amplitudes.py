#!/usr/bin/env python3
r"""Genuine genus-zero amplitudes containing Spin(23) singlets.

This module evaluates the singlet vertex as the anti-holomorphic
super-Liouville descendant

.. math::

   \mathcal S_{-1}(\omega)
   =c\widetilde c e^{-\varphi}e^{\pm i\omega X^0}
     \widetilde G_{-1/2}V_\omega .

It therefore does *not* identify the singlet with a fictitious 24th free
fermion.  With insertions ``(4,3,2,1)=(infinity,1,z,0)`` and picture-zero
vertices on legs 3 and 2, the two reduced integrands are

.. math::

   \mathcal I_{SSVV}^{ab}
   &=\delta^{ab}T_X\frac{1}{1-\bar z}
     \left[\mathcal C(M,O)
       +\frac{\omega_2\omega_3}{1-z}\mathcal C(P,O)\right],\\
   \mathcal I_{SSSS}
   &=T_X\left[\mathcal C(M,A)
       +\frac{\omega_2\omega_3}{1-z}\mathcal C(P,A)\right].

With vectors on the incoming and first outgoing legs, the same component
bookkeeping gives the additional channel

.. math::

   \mathcal I_{V\to VSS}
   =T_X\left[\mathcal C(M,M)
       +\frac{\omega_2\omega_3}{1-z}\mathcal C(P,M)\right].

Here ``P=(0,0,0,0)``, ``M=(0,1,1,0)``, ``O=(1,0,0,1)``, and
``A=(1,1,1,1)`` record external ``G_-1/2`` descendants.  ``C`` is the
diagonal super-Liouville spectral contraction.  Its component signs are not
fitted: converting each fixed-parity trilinear form to the physical
superfield component contributes

.. math::

   (-1)^{p_3(1-p_4)+p_2(1-r)},

where ``p_i`` is the external fermion parity and ``r`` is the exchanged-state
parity.  This reproduces the established primary and doubly-middle-starred
BRY conventions.

The moduli plane is folded to the unit disk and split into a bulk annulus
and analytically continued OPE disks at ``z=0`` and ``z=1``.  Blocks are
computed in sewing coordinates and algebraically converted to the elliptic
nome, where the production truncation is made. The full plane prefactor is
restored; neither the correlator nor the d^2z measure moves to a pillow. The
second meromorphic disk integral is essential here: singlet descendants make
the continued ``z=1`` integral non-absolutely-convergent in the scan domain,
so direct radial quadrature would measure a cutoff divergence.  All production
descendant coefficients use fixed-h c-recursion with their own osp(1|2)
large-c seeds, fusion parities, and odd-null transport signs.  The inverse-Gram
implementation is retained only as an explicitly selected validation oracle.

Returned amplitudes omit the common sphere normalization, coupling, energy
delta function, and asymptotic in/out reflection phases.  Both the raw vertex
normalization and unit-normalized external descendant convention are exposed.
"""

from __future__ import annotations

import cmath
import math
import sys
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import mpmath as mp
import sympy as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import roots_jacobi


BUNDLE_DIR = Path(__file__).resolve().parent
FIT_DIR = BUNDLE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

import heterotic_so23_1to3 as ref  # noqa: E402
import heterotic_so23_1to3_fast as fast  # noqa: E402
import ns_elliptic_conversion as elliptic_conversion  # noqa: E402
from ns_algebra.ns_sca import Word, fermion_parity, gram_matrix, twice_level  # noqa: E402
from ns_algebra.ns_three_point_tensor import ns_three_point  # noqa: E402
from spin23_ns_sphere_blocks import G_MINUS_HALF  # noqa: E402
from spin23_ns_c_recursion import block_coefficients  # noqa: E402
from spin23_three_point_amplitudes import singlet_descendant_norm  # noqa: E402


ExternalWords = tuple[Word, Word, Word, Word]

P_WORDS: ExternalWords = ((), (), (), ())
M_WORDS: ExternalWords = ((), G_MINUS_HALF, G_MINUS_HALF, ())
O_WORDS: ExternalWords = (G_MINUS_HALF, (), (), G_MINUS_HALF)
A_WORDS: ExternalWords = (G_MINUS_HALF,) * 4
L_WORDS: ExternalWords = (G_MINUS_HALF, G_MINUS_HALF, (), ())
R_WORDS: ExternalWords = ((), (), G_MINUS_HALF, G_MINUS_HALF)

WORD_PATTERNS: Mapping[str, ExternalWords] = {
    "P": P_WORDS,
    "M": M_WORDS,
    "O": O_WORDS,
    "A": A_WORDS,
    "L": L_WORDS,
    "R": R_WORDS,
}

_C, _H, _H1, _H2, _H3, _H4 = sp.symbols("c h h1 h2 h3 h4")
_TEMPLATE_ARGUMENTS = (_H, _H1, _H2, _H3, _H4, _C)


def _parities(words: ExternalWords) -> tuple[int, int, int, int]:
    return tuple(fermion_parity(word) for word in words)  # type: ignore[return-value]


def component_phase(words: ExternalWords, internal_parity: int) -> int:
    r"""Return the fixed-parity-to-superfield component phase.

    The left trinion has slots ``(4,3,r)`` and the right trinion has slots
    ``(r,2,1)``.  Applying ``(-1)**(b*(1-a))`` at both trinions gives the
    phase returned here.
    """

    if internal_parity not in (0, 1):
        raise ValueError("internal_parity must be 0 or 1")
    p1, p2, p3, p4 = _parities(words)
    exponent = p3 * (1 - p4) + p2 * (1 - internal_parity)
    return -1 if exponent % 2 else 1


def internal_parity(words: ExternalWords, structure_parity: int) -> int:
    """Return the exchanged-state parity selected by one trilinear form."""

    if structure_parity not in (0, 1):
        raise ValueError("structure_parity must be 0 or 1")
    p1, p2, p3, p4 = _parities(words)
    if (p1 + p2 + p3 + p4) % 2:
        raise ValueError("the four-point component must have even total parity")
    result = structure_parity ^ p1 ^ p2
    if result != (structure_parity ^ p3 ^ p4):
        raise AssertionError("left and right trinion parity routing disagree")
    return result


def _effective_weights(
    weights: Sequence[complex],
    words: ExternalWords,
) -> tuple[complex, complex, complex, complex]:
    return tuple(
        complex(weight) + twice_level(word) / 2
        for weight, word in zip(weights, words)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class _LevelTemplate:
    """Lambdified exact Ward/Gram data at one twice-level."""

    twice_descendant_level: int
    basis_size: int
    gram: Any
    vectors: Any
    slices: Mapping[str, tuple[slice, slice]]
    gram_expression: sp.Matrix
    vector_expressions: tuple[sp.Expr, ...]


@lru_cache(maxsize=None)
def _level_template(twice_descendant_level: int) -> _LevelTemplate:
    """Construct a symbolic template shared by all kinematic points."""

    basis, gram = gram_matrix(twice_descendant_level, h=_H, c=_C)
    expressions: list[sp.Expr] = []
    slices: dict[str, tuple[slice, slice]] = {}
    for name, words in WORD_PATTERNS.items():
        start = len(expressions)
        expressions.extend(
            ns_three_point(
                words[3],
                words[2],
                internal_word,
                h_infinity=_H4,
                h_middle=_H3,
                h_zero=_H,
                c=_C,
            )
            for internal_word in basis
        )
        middle = len(expressions)
        expressions.extend(
            ns_three_point(
                internal_word,
                words[1],
                words[0],
                h_infinity=_H,
                h_middle=_H2,
                h_zero=_H1,
                c=_C,
            )
            for internal_word in basis
        )
        stop = len(expressions)
        slices[name] = (slice(start, middle), slice(middle, stop))

    return _LevelTemplate(
        twice_descendant_level=twice_descendant_level,
        basis_size=len(basis),
        gram=sp.lambdify((_H, _C), gram, modules="numpy", cse=True),
        vectors=sp.lambdify(
            _TEMPLATE_ARGUMENTS,
            tuple(expressions),
            modules="numpy",
            cse=True,
        ),
        slices=slices,
        gram_expression=gram,
        vector_expressions=tuple(expressions),
    )


@dataclass(frozen=True)
class _HighPrecisionLevelTemplate:
    """Mpmath evaluators generated from one exact Ward/Gram template."""

    gram: Any
    vectors: Any


@lru_cache(maxsize=None)
def _high_precision_level_template(
    twice_descendant_level: int,
) -> _HighPrecisionLevelTemplate:
    """Compile arbitrary-precision evaluators only for requested levels."""

    template = _level_template(twice_descendant_level)
    return _HighPrecisionLevelTemplate(
        gram=sp.lambdify(
            (_H, _C),
            template.gram_expression,
            modules="mpmath",
            cse=True,
        ),
        vectors=sp.lambdify(
            _TEMPLATE_ARGUMENTS,
            template.vector_expressions,
            modules="mpmath",
            cse=True,
        ),
    )


@dataclass(frozen=True)
class _GramSolver:
    """One cached equilibrated Gram solve at fixed level and momentum."""

    cholesky: np.ndarray
    lower: bool
    equilibrated_matrix: np.ndarray
    diagonal_scale: np.ndarray
    condition: float
    equilibrated_condition: float


def _factor_gram_matrix(
    matrix: np.ndarray,
    *,
    twice_descendant_level: int,
    momentum: float,
    condition_limit: float,
) -> _GramSolver:
    r"""Equilibrate and factor one positive NS Gram matrix.

    In the PBW basis the diagonal norms span many orders of magnitude.  Write
    ``B = D E D``, where ``D_ii = sqrt(B_ii)``.  The block coefficient is then

    ``rho_L^T B^-1 rho_R = (D^-1 rho_L)^T E^-1 (D^-1 rho_R)``.

    This exact change of basis substantially lowers the floating-point
    condition number without changing the finite-level conformal block.
    """

    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram matrix must be square")
    imaginary_scale = float(np.max(np.abs(matrix.imag), initial=0.0))
    real_scale = float(np.max(np.abs(matrix.real), initial=1.0))
    if imaginary_scale > 1.0e-13 * max(1.0, real_scale):
        raise np.linalg.LinAlgError(
            "the physical NS Gram matrix should be real at real momentum"
        )
    real_matrix = np.asarray(matrix.real, dtype=float)
    # Remove tiny evaluation asymmetries before using a symmetric factorization.
    real_matrix = 0.5 * (real_matrix + real_matrix.T)
    diagonal = np.diag(real_matrix)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise np.linalg.LinAlgError(
            f"NS Gram matrix at level {twice_descendant_level}/2 and "
            f"P={momentum:.16g} is not positive definite on its diagonal"
        )
    diagonal_scale = np.sqrt(diagonal)
    equilibrated = (
        real_matrix / diagonal_scale[:, np.newaxis] / diagonal_scale[np.newaxis, :]
    )
    condition = float(np.linalg.cond(real_matrix))
    equilibrated_condition = float(np.linalg.cond(equilibrated))
    if (
        not math.isfinite(condition)
        or not math.isfinite(equilibrated_condition)
        or equilibrated_condition > condition_limit
    ):
        raise np.linalg.LinAlgError(
            f"equilibrated NS Gram matrix at level "
            f"{twice_descendant_level}/2 and P={momentum:.16g} has "
            f"condition number {equilibrated_condition:.3e} "
            f"(raw {condition:.3e}), above {condition_limit:.3e}"
        )
    try:
        cholesky, lower = cho_factor(
            equilibrated,
            lower=True,
            overwrite_a=False,
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            f"equilibrated NS Gram matrix at level "
            f"{twice_descendant_level}/2 and P={momentum:.16g} is not "
            "numerically positive definite"
        ) from error
    return _GramSolver(
        cholesky=cholesky,
        lower=bool(lower),
        equilibrated_matrix=equilibrated,
        diagonal_scale=diagonal_scale,
        condition=condition,
        equilibrated_condition=equilibrated_condition,
    )


def _gram_bilinear_coefficients(
    solver: _GramSolver,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
) -> np.ndarray:
    """Evaluate several ``left.T @ B^-1 @ right`` pairings stably."""

    left = np.asarray(left_vectors, dtype=complex)
    right = np.asarray(right_vectors, dtype=complex)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.T.shape:
        raise ValueError("left and right vector arrays have incompatible shapes")
    left_scaled = left / solver.diagonal_scale[np.newaxis, :]
    right_scaled = right / solver.diagonal_scale[:, np.newaxis]
    solutions = cho_solve(
        (solver.cholesky, solver.lower),
        right_scaled,
        overwrite_b=False,
        check_finite=False,
    )
    if solver.equilibrated_condition >= 1.0e6:
        matrix_extended = np.asarray(
            solver.equilibrated_matrix,
            dtype=np.longdouble,
        )
        right_real = np.asarray(right_scaled.real, dtype=np.longdouble)
        right_imag = np.asarray(right_scaled.imag, dtype=np.longdouble)
        for _ in range(2):
            residual = np.asarray(
                right_real
                - matrix_extended
                @ np.asarray(solutions.real, dtype=np.longdouble),
                dtype=float,
            ) + 1j * np.asarray(
                right_imag
                - matrix_extended
                @ np.asarray(solutions.imag, dtype=np.longdouble),
                dtype=float,
            )
            correction = cho_solve(
                (solver.cholesky, solver.lower),
                residual,
                overwrite_b=False,
                check_finite=False,
            )
            solutions += correction
            if np.max(np.abs(correction), initial=0.0) <= (
                4 * np.finfo(float).eps
                * max(1.0, float(np.max(np.abs(solutions), initial=0.0)))
            ):
                break

    products = left_scaled * solutions.T
    return np.asarray(
        [
            complex(
                float(np.sum(row.real, dtype=np.longdouble)),
                float(np.sum(row.imag, dtype=np.longdouble)),
            )
            for row in products
        ],
        dtype=complex,
    )


def _mp_complex(value: complex) -> mp.mpc:
    """Convert a binary complex value without decimal-string truncation."""

    normalized = complex(value)
    return mp.mpc(mp.mpf(normalized.real), mp.mpf(normalized.imag))


def _high_precision_level_coefficients(
    *,
    twice_descendant_level: int,
    h_internal: complex,
    weights: tuple[complex, complex, complex, complex],
    pattern_names: Sequence[str],
    digits: int,
) -> np.ndarray:
    """Evaluate one inverse-Gram coefficient vector with mpmath.

    The Gram matrix and both Ward vectors are reevaluated from their exact
    SymPy expressions at ``digits`` decimal digits.  This is a selective
    oracle for ill-conditioned levels, rather than a binary64 iterative
    refinement whose residual would still use rounded matrix entries.
    """

    if digits < 30:
        raise ValueError("high-precision Gram solves require at least 30 digits")
    level = _level_template(twice_descendant_level)
    high_precision = _high_precision_level_template(twice_descendant_level)
    with mp.workdps(int(digits)):
        h_mp = _mp_complex(h_internal)
        weights_mp = tuple(_mp_complex(value) for value in weights)
        matrix = high_precision.gram(h_mp, mp.mpf("13.5"))
        raw_vectors = high_precision.vectors(
            h_mp,
            *weights_mp,
            mp.mpf("13.5"),
        )
        values = tuple(mp.mpc(value) for value in raw_vectors)
        factor, pivots = mp.mp.LU_decomp(matrix.copy(), use_cache=False)
        coefficients: list[complex] = []
        for name in pattern_names:
            left_slice, right_slice = level.slices[name]
            left = mp.matrix(values[left_slice])
            right = mp.matrix(values[right_slice])
            intermediate = mp.mp.L_solve(factor, right, pivots)
            solution = mp.mp.U_solve(factor, intermediate)
            coefficient = (left.T * solution)[0]
            coefficients.append(complex(coefficient))
    return np.asarray(coefficients, dtype=complex)


@lru_cache(maxsize=None)
def _gram_solvers_cached(
    momentum_tuple: tuple[float, ...],
    maximum_twice_level: int,
    condition_limit: float,
) -> tuple[dict[int, tuple[_GramSolver, ...]], float]:
    momenta = np.asarray(momentum_tuple, dtype=float)
    solvers: dict[int, tuple[_GramSolver, ...]] = {}
    maximum_condition = 1.0
    for level in range(maximum_twice_level + 1):
        template = _level_template(level)
        level_solvers: list[_GramSolver] = []
        for momentum in momenta:
            h_internal = fast.h_of_p(float(momentum))
            matrix = np.asarray(template.gram(h_internal, 13.5), dtype=complex)
            matrix = np.reshape(matrix, (template.basis_size, template.basis_size))
            solver = _factor_gram_matrix(
                matrix,
                twice_descendant_level=level,
                momentum=float(momentum),
                condition_limit=condition_limit,
            )
            level_solvers.append(solver)
            maximum_condition = max(maximum_condition, solver.condition)
        solvers[level] = tuple(level_solvers)
    return solvers, maximum_condition


def _gram_solvers(
    momenta: np.ndarray,
    maximum_twice_level: int,
    *,
    condition_limit: float,
) -> tuple[dict[int, tuple[_GramSolver, ...]], float]:
    """Return cached equilibrated factors for one momentum quadrature."""

    momentum_tuple = tuple(float(value) for value in np.asarray(momenta).reshape(-1))
    return _gram_solvers_cached(
        momentum_tuple,
        int(maximum_twice_level),
        float(condition_limit),
    )


def _momentum_quadrature(
    p_nodes: int | str | Sequence[int],
    p_max: float,
    p_cut: float,
    *,
    scheme: str = "cutoff",
    infinite_gauss_scale: float = 1.0,
    momentum_threshold_options=None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the internal-momentum quadrature used by the singlet integral.

    ``scheme="infinite_gauss"`` maps a Gauss--Legendre rule on ``[-1,1]``
    directly onto the full half-line via

        P = -scale log((1-x)/2),   dP = scale dx/(1-x).

    This is retained as an independent cutoff-free control. Here
    ``p_nodes`` must be an integer, while ``p_max`` and ``p_cut`` are ignored.

    ``scheme="infinite_composite"`` is the resolved near-real variant.  It
    applies Gauss--Legendre rules on the six fixed intervals with boundaries
    ``(0,.18,.36,.56,.85,1.30,2.10)`` and maps a seventh Gaussian rule from
    ``[2.10,infinity)`` with the same logarithmic half-line map.  ``p_nodes``
    is either ``"segmented"`` or a sequence of seven node counts.  This keeps
    the full infinite domain while resolving cancellations that a single
    global map can miss near the physical-energy boundary.

    ``scheme="infinite_composite_extended"`` adds the resolved interval
    ``[2.10,3.20]`` before mapping ``[3.20,infinity)``.  It takes eight node
    counts and is intended for the enlarged real-energy scan, whose dominant
    structure support moves beyond ``P=2.10`` near ``E=1``.

    The legacy ``scheme="cutoff"`` uses the same seven bulk intervals as
    :func:`fast._p_quadrature`.  Its endpoint interval is evaluated with a
    four-node Gauss--Jacobi rule whose weight is proportional to ``P**2``.
    """

    if scheme == "threshold_weighted":
        rule = ref.threshold_weighted_rule(p_nodes, momentum_threshold_options)
        return rule.momenta, rule.weights
    if momentum_threshold_options is not None:
        raise ValueError("momentum_threshold_options require scheme='threshold_weighted'")
    if scheme == "infinite_gauss":
        if not isinstance(p_nodes, int) or isinstance(p_nodes, bool):
            raise ValueError("infinite_gauss momentum quadrature requires integer p_nodes")
        if p_nodes < 1:
            raise ValueError("p_nodes must be positive")
        if not math.isfinite(infinite_gauss_scale) or infinite_gauss_scale <= 0:
            raise ValueError("infinite_gauss_scale must be finite and positive")
        nodes, weights = np.polynomial.legendre.leggauss(p_nodes)
        momenta = -infinite_gauss_scale * np.log((1.0 - nodes) / 2.0)
        momentum_weights = infinite_gauss_scale * weights / (1.0 - nodes)
        return momenta, momentum_weights
    if scheme in ("infinite_composite", "infinite_composite_extended"):
        if not math.isfinite(infinite_gauss_scale) or infinite_gauss_scale <= 0:
            raise ValueError("infinite_gauss_scale must be finite and positive")
        if scheme == "infinite_composite":
            boundaries = (0.0, 0.18, 0.36, 0.56, 0.85, 1.30, 2.10)
            default_counts = [16, 18, 24, 18, 16, 14, 20]
        else:
            boundaries = (0.0, 0.18, 0.36, 0.56, 0.85, 1.30, 2.10, 3.20)
            default_counts = [16, 18, 24, 18, 16, 18, 24, 20]
        if isinstance(p_nodes, str):
            if p_nodes.lower() != "segmented":
                raise ValueError("p_nodes string must be 'segmented'")
            counts = default_counts
        elif isinstance(p_nodes, Sequence):
            counts = [int(value) for value in p_nodes]
            if len(counts) != len(boundaries):
                raise ValueError(
                    f"{scheme} p_nodes must have {len(boundaries)} counts"
                )
        else:
            raise ValueError(
                f"{scheme} momentum quadrature requires segmented counts"
            )
        if any(count < 1 for count in counts):
            raise ValueError(f"all {scheme} node counts must be positive")
        momentum_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for lower, upper, count in zip(boundaries[:-1], boundaries[1:], counts[:-1]):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            momentum_parts.append((lower + upper) / 2 + (upper - lower) * nodes / 2)
            weight_parts.append((upper - lower) * weights / 2)
        tail_nodes, tail_weights = np.polynomial.legendre.leggauss(counts[-1])
        tail_momenta = boundaries[-1] - infinite_gauss_scale * np.log(
            (1.0 - tail_nodes) / 2.0
        )
        tail_jacobian = infinite_gauss_scale / (1.0 - tail_nodes)
        momentum_parts.append(tail_momenta)
        weight_parts.append(tail_weights * tail_jacobian)
        return np.concatenate(momentum_parts), np.concatenate(weight_parts)
    if scheme != "cutoff":
        raise ValueError(f"unknown momentum quadrature scheme: {scheme!r}")

    if isinstance(p_nodes, int) and p_nodes < 5:
        nodes, weights = np.polynomial.legendre.leggauss(p_nodes)
        return (nodes + 1) * p_max / 2, weights * p_max / 2

    momenta, momentum_weights = fast._p_quadrature(p_nodes, p_max, p_cut)
    momenta = np.asarray(momenta, dtype=float)
    momentum_weights = np.asarray(momentum_weights, dtype=float)
    endpoint_count = 4
    jacobi_nodes, jacobi_weights = roots_jacobi(endpoint_count, 0.0, 2.0)
    endpoint_momenta = p_cut * (jacobi_nodes + 1.0) / 2.0
    endpoint_weights = (
        p_cut**3
        * jacobi_weights
        / 8.0
        / endpoint_momenta**2
    )
    momenta[:endpoint_count] = endpoint_momenta
    momentum_weights[:endpoint_count] = endpoint_weights
    return momenta, momentum_weights


@dataclass(frozen=True)
class _CoefficientTable:
    """Direct z-series coefficients for every requested external pattern."""

    momenta: np.ndarray
    weights: tuple[complex, complex, complex, complex]
    coefficients: Mapping[str, tuple[np.ndarray, np.ndarray]]
    high_precision_solve_count: int
    maximum_recursion_cancellation: float | None = None
    high_precision_recursion_node_count: int = 0


def _coefficient_table(
    weights: Sequence[complex],
    momenta: np.ndarray,
    solvers: Mapping[int, tuple[_GramSolver, ...]],
    q_order: int,
    pattern_names: Sequence[str],
    *,
    high_precision_condition: float | None = None,
    high_precision_digits: int = 60,
    high_precision_max_momentum: float | None = None,
) -> _CoefficientTable:
    """Inverse-Gram oracle; production uses _recursive_coefficient_table."""
    normalized_weights = tuple(complex(value) for value in weights)
    maximum_twice_level = 2 * q_order + 1
    coefficients: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (
            np.empty((momenta.size, q_order + 1), dtype=complex),
            np.empty((momenta.size, q_order + 1), dtype=complex),
        )
        for name in pattern_names
    }
    high_precision_solve_count = 0

    h1, h2, h3, h4 = normalized_weights
    internal_weights = np.asarray(
        [fast.h_of_p(float(momentum)) for momentum in momenta],
        dtype=complex,
    )
    for level in range(maximum_twice_level + 1):
        template = _level_template(level)
        parity = level % 2
        power = (level - parity) // 2
        raw_vectors = template.vectors(
            internal_weights,
            h1,
            h2,
            h3,
            h4,
            13.5,
        )
        # Lambdified constant expressions remain scalars while expressions
        # depending on h_internal return one value per momentum.  Broadcast
        # the former explicitly and store rows by momentum.
        vector_values = np.stack(
            [
                np.broadcast_to(
                    np.asarray(expression, dtype=complex),
                    internal_weights.shape,
                )
                for expression in raw_vectors
            ],
            axis=1,
        )
        for momentum_index, values in enumerate(vector_values):
            solver = solvers[level][momentum_index]
            left_vectors = []
            right_vectors = []
            for name in pattern_names:
                left_slice, right_slice = template.slices[name]
                left_vectors.append(values[left_slice])
                right_vectors.append(values[right_slice])
            right_hand_sides = np.stack(right_vectors, axis=1)
            level_coefficients = _gram_bilinear_coefficients(
                solver,
                np.stack(left_vectors, axis=0),
                right_hand_sides,
            )
            use_high_precision = (
                high_precision_condition is not None
                and solver.equilibrated_condition > high_precision_condition
                and (
                    high_precision_max_momentum is None
                    or float(momenta[momentum_index]) <= high_precision_max_momentum
                )
            )
            if use_high_precision:
                level_coefficients = _high_precision_level_coefficients(
                    twice_descendant_level=level,
                    h_internal=internal_weights[momentum_index],
                    weights=normalized_weights,  # type: ignore[arg-type]
                    pattern_names=pattern_names,
                    digits=high_precision_digits,
                )
                high_precision_solve_count += 1
            for name, value in zip(pattern_names, level_coefficients):
                coefficients[name][parity][momentum_index, power] = value

    return _CoefficientTable(
        momenta=np.asarray(momenta, dtype=float),
        weights=normalized_weights,  # type: ignore[arg-type]
        coefficients=coefficients,
        high_precision_solve_count=high_precision_solve_count,
    )


def _recursive_coefficient_table(
    weights: Sequence[complex],
    momenta: np.ndarray,
    q_order: int,
    pattern_names: Sequence[str],
    *,
    digits: int = 70,
    reference_p_max: float = 0.18,
    cancellation_limit: float = 1.0e6,
) -> _CoefficientTable:
    """All P/M/O/A/L/R coefficients from descendant-aware c-recursion."""
    normalized_weights = tuple(complex(value) for value in weights)
    patterns = {name: tuple(int(bool(w)) for w in WORD_PATTERNS[name])
                for name in pattern_names}
    coefficients = {name: (np.empty((momenta.size, q_order + 1), complex),
                           np.empty((momenta.size, q_order + 1), complex))
                    for name in pattern_names}
    max_cancellation = None
    precise_nodes = 0
    for index, momentum in enumerate(momenta):
        result = block_coefficients(
            c=13.5,
            h_internal=fast.h_of_p(float(momentum)),
            external_weights=normalized_weights,
            external_patterns=tuple(patterns.values()),
            maximum_twice_level=2 * q_order + 1,
            force_high_precision=float(momentum) <= reference_p_max,
            digits=digits,
            cancellation_limit=cancellation_limit,
        )
        precise_nodes += int(result.high_precision)
        if result.maximum_cancellation is not None:
            max_cancellation = max(max_cancellation or 0, result.maximum_cancellation)
        for name, alphas in patterns.items():
            values = result.coefficients[alphas]
            coefficients[name][0][index] = values[::2]
            coefficients[name][1][index] = values[1::2]
    return _CoefficientTable(
        momenta=np.asarray(momenta, dtype=float),
        weights=normalized_weights,  # type: ignore[arg-type]
        coefficients=coefficients,
        high_precision_solve_count=0,
        maximum_recursion_cancellation=max_cancellation,
        high_precision_recursion_node_count=precise_nodes,
    )


def _crossed_pair_coefficients_from_ward_identity(
    table: _CoefficientTable,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    r"""Construct the adjacent-pair ``L`` and ``R`` blocks from ``P`` and ``M``.

    These are exact global-superconformal Ward identities, applied after the
    component phases in :func:`component_phase`.  If

    ``P = <V4 V3 V2 V1>`` and ``M = <V4 GV3 GV2 V1>``,

    the two adjacent-pair components are

    ``L = <V4 V3 GV2 GV1>`` and ``R = <GV4 GV3 V2 V1>``.

    Constructing them from the two standard block families avoids two
    independent inverse-Gram contractions and supplies a coefficientwise
    identity that can fail if a descendant sign or ordering is wrong.
    """

    if "P" not in table.coefficients or "M" not in table.coefficients:
        raise ValueError("the Ward reduction requires both P and M coefficients")
    p_even_raw, p_odd_raw = table.coefficients["P"]
    m_even_raw, m_odd_raw = table.coefficients["M"]
    p_even = component_phase(P_WORDS, 0) * p_even_raw
    p_odd = component_phase(P_WORDS, 1) * p_odd_raw
    m_even = component_phase(M_WORDS, 0) * m_even_raw
    m_odd = component_phase(M_WORDS, 1) * m_odd_raw

    h1, h2, h3, h4 = table.weights
    h_internal = np.asarray(
        [fast.h_of_p(float(momentum)) for momentum in table.momenta],
        dtype=complex,
    )[:, np.newaxis]
    orders = np.arange(p_even.shape[1], dtype=float)[np.newaxis, :]
    left_exponent = h_internal - h1 - h2
    right_exponent = h3 + h4 - h_internal
    shifted_m_odd = np.column_stack(
        (np.zeros(table.momenta.size, dtype=complex), m_odd[:, :-1])
    )

    l_even = -(left_exponent + orders) * p_even + shifted_m_odd
    l_odd = m_even - (left_exponent + 0.5 + orders) * p_odd
    r_even = (right_exponent - orders) * p_even + shifted_m_odd
    r_odd = m_even + (right_exponent - 0.5 - orders) * p_odd

    # _generic_block applies component_phase, so return coefficients in the
    # fixed-parity convention expected by the rest of the module.
    l_raw = (
        l_even / component_phase(L_WORDS, 0),
        l_odd / component_phase(L_WORDS, 1),
    )
    r_raw = (
        r_even / component_phase(R_WORDS, 0),
        r_odd / component_phase(R_WORDS, 1),
    )
    return l_raw, r_raw


@dataclass(frozen=True)
class _GenericBlock:
    """Genuine component coefficients and their algebraic nome resummation."""

    h_internal: complex
    words: ExternalWords
    base_exponent: complex
    effective_weights: tuple[complex, complex, complex, complex]
    even_z: np.ndarray
    odd_z: np.ndarray
    q_order: int

    @cached_property
    def elliptic(self):
        return elliptic_conversion.EllipticComponentSeries(
            self.h_internal, self.effective_weights, self.even_z, self.odd_z, self.q_order,
        )

    def local_data(self, parity, order, total_order, series_parameter):
        if series_parameter == "elliptic_nome":
            return self.elliptic.local_data(parity, order, total_order)
        return self.direct_data(parity, order)

    def _component(self, parity: int) -> tuple[np.ndarray, complex]:
        if parity == 0:
            return self.even_z, self.base_exponent
        if parity == 1:
            return self.odd_z, self.base_exponent + 0.5
        raise ValueError("parity must be 0 or 1")

    def direct_data(self, parity: int, order: int) -> tuple[complex, np.ndarray]:
        z_coefficients, exponent = self._component(parity)
        return exponent, z_coefficients[: order + 1]

    def direct_value(self, parity: int, z: np.ndarray, order: int) -> np.ndarray:
        exponent, coefficients = self.direct_data(parity, order)
        return np.exp(exponent * np.log(z)) * np.polynomial.polynomial.polyval(
            z, coefficients
        )

def _generic_block(
    *,
    h_internal: complex,
    external_weights: Sequence[complex],
    words: ExternalWords,
    even_coefficients: np.ndarray,
    odd_coefficients: np.ndarray,
    q_order: int,
) -> _GenericBlock:
    effective = _effective_weights(external_weights, words)
    h1, h2, _, _ = effective
    h_internal = complex(h_internal)
    even_z = component_phase(words, 0) * np.asarray(even_coefficients, dtype=complex)
    odd_z = component_phase(words, 1) * np.asarray(odd_coefficients, dtype=complex)

    return _GenericBlock(
        h_internal=h_internal,
        words=words,
        base_exponent=h_internal - h1 - h2,
        effective_weights=effective,
        even_z=even_z,
        odd_z=odd_z,
        q_order=q_order,
    )


@dataclass(frozen=True)
class _SingletKernel:
    momentum: float
    quadrature_weight: float
    even_structure: complex
    odd_structure: complex
    blocks: Mapping[str, _GenericBlock]


def _build_kernels(
    ordered_energies: Sequence[complex],
    table: _CoefficientTable,
    pattern_aliases: Mapping[str, str],
    momentum_weights: np.ndarray,
    q_order: int,
    structure_cache: MutableMapping[tuple[Any, ...], tuple[complex, complex]],
) -> tuple[_SingletKernel, ...]:
    energies = tuple(complex(value) for value in ordered_energies)
    kernels: list[_SingletKernel] = []
    for index, (momentum, momentum_weight) in enumerate(
        zip(table.momenta, momentum_weights)
    ):
        h_internal = fast.h_of_p(float(momentum))
        blocks: dict[str, _GenericBlock] = {}
        for public_name, actual_name in pattern_aliases.items():
            even, odd = table.coefficients[actual_name]
            words = WORD_PATTERNS[actual_name]
            blocks[public_name] = _generic_block(
                h_internal=h_internal,
                external_weights=table.weights,
                words=words,
                even_coefficients=even[index],
                odd_coefficients=odd[index],
                q_order=q_order,
            )
        ce, co = fast._structure_products(
            (energies[0], energies[1], float(momentum)),
            (energies[2], energies[3], float(momentum)),
            structure_cache,
        )
        kernels.append(
            _SingletKernel(
                momentum=float(momentum),
                quadrature_weight=float(momentum_weight) / math.pi,
                even_structure=ce,
                odd_structure=co,
                blocks=blocks,
            )
        )
    return tuple(kernels)


@dataclass(frozen=True)
class _ChannelData:
    # Physical ordering is always (old 1,old 2,old 3,old 4).  The blocks in a
    # crossed channel have a different internal ordering, but TX and the PCO
    # coefficient must continue to use these physical energies.
    energies: tuple[complex, complex, complex, complex]
    kernels: tuple[_SingletKernel, ...]


@dataclass(frozen=True)
class _AmplitudeAtlas:
    original_s: _ChannelData
    original_t: _ChannelData
    swapped_s: _ChannelData
    swapped_t: _ChannelData
    maximum_gram_condition: float | None
    maximum_equilibrated_gram_condition: float | None
    high_precision_gram_solve_count: int
    build_seconds: float
    template_seconds: float
    block_backend: str
    maximum_recursion_cancellation: float | None
    high_precision_recursion_node_count: int
    momentum_quadrature: dict[str, Any]


def _ordered_weights(energies: Sequence[complex]) -> tuple[complex, ...]:
    return tuple(fast.h_of_p(value) for value in energies)


def _build_atlas(
    energies: Sequence[complex],
    *,
    q_order: int,
    p_nodes: int | str | Sequence[int],
    p_max: float,
    p_cut: float,
    gram_condition_limit: float,
    gram_high_precision_condition: float | None = None,
    gram_high_precision_digits: int = 60,
    gram_high_precision_max_momentum: float | None = None,
    momentum_scheme: str = "cutoff",
    infinite_gauss_scale: float = 1.0,
    momentum_threshold_options=None,
    block_backend: str = "c_recursion",
    recursion_digits: int = 70,
    recursion_reference_p_max: float = 0.18,
    recursion_cancellation_limit: float = 1.0e6,
) -> _AmplitudeAtlas:
    if block_backend not in ("c_recursion", "inverse_gram"):
        raise ValueError("block_backend must be 'c_recursion' or 'inverse_gram'")
    started = time.perf_counter()
    maximum_twice_level = 2 * q_order + 1
    template_seconds = 0.0
    if block_backend == "inverse_gram":
        template_started = time.perf_counter()
        for level in range(maximum_twice_level + 1):
            _level_template(level)
        template_seconds = time.perf_counter() - template_started

    momenta, momentum_weights = _momentum_quadrature(
        p_nodes,
        p_max,
        p_cut,
        scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
        momentum_threshold_options=momentum_threshold_options,
    )
    momenta = np.asarray(momenta, dtype=float)
    momentum_weights = np.asarray(momentum_weights, dtype=float)
    maximum_condition = maximum_equilibrated_condition = None
    if block_backend == "inverse_gram":
        solvers, maximum_condition = _gram_solvers(
            momenta, maximum_twice_level, condition_limit=gram_condition_limit,
        )
        maximum_equilibrated_condition = max(
            solver.equilibrated_condition
            for level_solvers in solvers.values() for solver in level_solvers
        )

    original = tuple(complex(value) for value in energies)
    swapped = (original[0], original[2], original[1], original[3])
    t_permutation = (2, 1, 0, 3)
    original_t = tuple(original[index] for index in t_permutation)
    swapped_t = tuple(swapped[index] for index in t_permutation)

    channel_specs = (
        ("original_s", original, original, ("P", "M", "O", "A"), {"P": "P", "M": "M", "O": "O", "A": "A"}),
        ("original_t", original_t, original, ("P", "M", "A"), {"P": "P", "M": "L", "O": "R", "A": "A"}),
        ("swapped_s", swapped, swapped, ("P", "M", "O", "A"), {"P": "P", "M": "M", "O": "O", "A": "A"}),
        ("swapped_t", swapped_t, swapped, ("P", "M", "A"), {"P": "P", "M": "L", "O": "R", "A": "A"}),
    )
    structure_cache: dict[tuple[Any, ...], tuple[complex, complex]] = {}
    built: dict[str, _ChannelData] = {}
    high_precision_solve_count = 0
    high_precision_recursion_node_count = 0
    maximum_recursion_cancellation = None
    # Equal-energy channels often have identical weight tuples but different
    # descendant aliases.  Recurse once on their union, not once per chart.
    recursive_tables: dict[tuple[complex, ...], _CoefficientTable] = {}
    required_patterns: dict[tuple[complex, ...], list[str]] = {}
    for _, ordered_energies, _, _, aliases in channel_specs:
        union = required_patterns.setdefault(_ordered_weights(ordered_energies), [])
        union.extend(name for name in aliases.values() if name not in union)
    kernel_cache: dict[tuple[Any, ...], tuple[_SingletKernel, ...]] = {}
    for label, ordered_energies, physical_energies, names, aliases in channel_specs:
        weights = _ordered_weights(ordered_energies)
        table_was_built = True
        if block_backend == "c_recursion":
            table_was_built = weights not in recursive_tables
            if table_was_built:
                recursive_tables[weights] = _recursive_coefficient_table(
                    weights, momenta, q_order, required_patterns[weights],
                    digits=recursion_digits,
                    reference_p_max=recursion_reference_p_max,
                    cancellation_limit=recursion_cancellation_limit,
                )
            table = recursive_tables[weights]
        else:
            table = _coefficient_table(
                weights, momenta, solvers, q_order, names,
                high_precision_condition=gram_high_precision_condition,
                high_precision_digits=gram_high_precision_digits,
                high_precision_max_momentum=gram_high_precision_max_momentum,
            )
        high_precision_solve_count += table.high_precision_solve_count
        if table_was_built:
            high_precision_recursion_node_count += table.high_precision_recursion_node_count
        if table.maximum_recursion_cancellation is not None:
            maximum_recursion_cancellation = max(
                maximum_recursion_cancellation or 0, table.maximum_recursion_cancellation,
            )
        if block_backend == "inverse_gram" and label.endswith("_t"):
            l_coefficients, r_coefficients = (
                _crossed_pair_coefficients_from_ward_identity(table)
            )
            table = _CoefficientTable(
                momenta=table.momenta,
                weights=table.weights,
                coefficients={
                    **table.coefficients,
                    "L": l_coefficients,
                    "R": r_coefficients,
                },
                high_precision_solve_count=table.high_precision_solve_count,
            )
        kernel_key = (ordered_energies, tuple(aliases.items()))
        if kernel_key not in kernel_cache:
            kernel_cache[kernel_key] = _build_kernels(
                ordered_energies,
                table,
                aliases,
                momentum_weights,
                q_order,
                structure_cache,
            )
        built[label] = _ChannelData(
            energies=physical_energies,  # type: ignore[arg-type]
            kernels=kernel_cache[kernel_key],
        )

    return _AmplitudeAtlas(
        original_s=built["original_s"],
        original_t=built["original_t"],
        swapped_s=built["swapped_s"],
        swapped_t=built["swapped_t"],
        maximum_gram_condition=maximum_condition,
        maximum_equilibrated_gram_condition=maximum_equilibrated_condition,
        high_precision_gram_solve_count=high_precision_solve_count,
        build_seconds=time.perf_counter() - started,
        template_seconds=template_seconds,
        block_backend=block_backend,
        maximum_recursion_cancellation=maximum_recursion_cancellation,
        high_precision_recursion_node_count=high_precision_recursion_node_count,
        momentum_quadrature=(ref.threshold_weighted_rule(p_nodes, momentum_threshold_options).metadata()
                             if momentum_scheme == "threshold_weighted" else
                             {"scheme": momentum_scheme, "node_count": len(momenta)}),
    )


def _contract_grid(
    kernel: _SingletKernel,
    holomorphic_name: str,
    antiholomorphic_name: str,
    coordinate: np.ndarray,
    conjugate_coordinate: np.ndarray,
    *,
    q: np.ndarray | None,
    qbar: np.ndarray | None,
    theta3: np.ndarray | None,
    theta3bar: np.ndarray | None,
    order: int,
    series_parameter: str = "sewing",
) -> np.ndarray:
    elliptic_conversion.validate_representation(series_parameter)
    if series_parameter == "sewing" and any(value is not None for value in (q, qbar, theta3, theta3bar)):
        raise ValueError("ordinary sewing blocks take the local coordinate directly; no elliptic q/theta data")
    if series_parameter == "elliptic_nome":
        if q is None or theta3 is None:
            q, theta3, _ = elliptic_conversion.nome_geometry(coordinate)
        if qbar is None or theta3bar is None:
            qbar, theta3bar, _ = elliptic_conversion.nome_geometry(conjugate_coordinate)
    holomorphic = kernel.blocks[holomorphic_name]
    antiholomorphic = kernel.blocks[antiholomorphic_name]
    total = np.zeros_like(coordinate, dtype=complex)
    for structure_parity, structure in (
        (0, kernel.even_structure),
        (1, kernel.odd_structure),
    ):
        holomorphic_parity = internal_parity(
            holomorphic.words, structure_parity
        )
        antiholomorphic_parity = internal_parity(
            antiholomorphic.words, structure_parity
        )
        if series_parameter == "elliptic_nome":
            holomorphic_value = holomorphic.elliptic.value(
                coordinate, holomorphic_parity, order, geometry=(q, theta3, None))
            antiholomorphic_value = antiholomorphic.elliptic.value(
                conjugate_coordinate, antiholomorphic_parity, order, geometry=(qbar, theta3bar, None))
        else:
            holomorphic_value = holomorphic.direct_value(holomorphic_parity, coordinate, order)
            antiholomorphic_value = antiholomorphic.direct_value(antiholomorphic_parity, conjugate_coordinate, order)
        total += structure * holomorphic_value * antiholomorphic_value
    return total


def _stable_complex_sum(values: Any) -> complex:
    """Sum complex values with extended-precision pairwise reduction.

    NumPy's ``clongdouble`` accumulator is used only for the reduction; the
    integrand itself remains binary64.  This targets cancellation between
    quadrature nodes without changing the special-function backend.
    """

    flattened = np.asarray(values, dtype=complex).reshape(-1)
    if flattened.size == 0:
        return 0.0j
    real = np.sum(flattened.real, dtype=np.longdouble)
    imaginary = np.sum(flattened.imag, dtype=np.longdouble)
    return complex(float(real), float(imaginary))


def _stable_complex_fsum(values: Any) -> complex:
    """Accumulate a short sequence with correctly rounded scalar sums."""

    normalized = tuple(complex(value) for value in values)
    return complex(
        math.fsum(value.real for value in normalized),
        math.fsum(value.imag for value in normalized),
    )


def _process_antiholomorphic_pattern(process: str) -> str:
    """Return the physical right-moving descendant pattern for a channel."""

    patterns = {
        "ssvv": "O",
        "ssss": "A",
        "v_to_vss": "M",
    }
    try:
        return patterns[str(process)]
    except KeyError as error:
        raise ValueError(
            f"unknown singlet four-point process {process!r}"
        ) from error


def _grid_integral(
    data: _ChannelData,
    coordinate: np.ndarray,
    area_weights: np.ndarray,
    *,
    original_z: np.ndarray,
    q: np.ndarray | None,
    theta3: np.ndarray | None,
    order: int,
    process: str,
    series_parameter: str = "sewing",
) -> complex:
    if series_parameter == "elliptic_nome" and (q is None or theta3 is None):
        q, theta3, _ = elliptic_conversion.nome_geometry(coordinate)
    zbar = np.conj(original_z)
    coordinate_bar = np.conj(coordinate)
    qbar = None if q is None else np.conj(q)
    theta3bar = None if theta3 is None else np.conj(theta3)
    w1, w2, w3, _ = data.energies
    antiholomorphic_name = _process_antiholomorphic_pattern(process)
    spectator = 1 / (1 - zbar) if process == "ssvv" else 1.0
    momentum_terms: list[complex] = []
    time_factor = np.exp(
        -2 * w1 * w2 * np.log(np.abs(original_z))
        -2 * w2 * w3 * np.log(np.abs(1 - original_z))
    )
    for kernel in data.kernels:
        descendant = _contract_grid(
            kernel,
            "M",
            antiholomorphic_name,
            coordinate,
            coordinate_bar,
            q=q,
            qbar=qbar,
            theta3=theta3,
            theta3bar=theta3bar,
            order=order,
            series_parameter=series_parameter,
        )
        primary = _contract_grid(
            kernel,
            "P",
            antiholomorphic_name,
            coordinate,
            coordinate_bar,
            q=q,
            qbar=qbar,
            theta3=theta3,
            theta3bar=theta3bar,
            order=order,
            series_parameter=series_parameter,
        )
        bracket = descendant + (w2 * w3) * primary / (1 - original_z)
        momentum_terms.append(
            kernel.quadrature_weight
            * _stable_complex_sum(area_weights * time_factor * spectator * bracket)
        )
    return _stable_complex_fsum(momentum_terms)


def _disk_kernel_integral(
    kernel: _SingletKernel,
    energies: Sequence[complex],
    epsilon: float,
    *,
    block_order: int,
    total_order: int,
    process: str,
    series_parameter: str = "sewing",
) -> complex:
    w1, w2, w3, _ = map(complex, energies)
    one_minus_exponent = w2 * w3
    zero_exponent = w1 * w2
    antiholomorphic_name = _process_antiholomorphic_pattern(process)
    anti_extra = 1 if process == "ssvv" else 0
    terms: list[complex] = []

    for structure_parity, structure in (
        (0, kernel.even_structure),
        (1, kernel.odd_structure),
    ):
        for holomorphic_name, holomorphic_extra, coefficient in (
            ("M", 0, 1.0),
            ("P", 1, w2 * w3),
        ):
            holomorphic = kernel.blocks[holomorphic_name]
            antiholomorphic = kernel.blocks[antiholomorphic_name]
            holomorphic_parity = internal_parity(
                holomorphic.words, structure_parity
            )
            antiholomorphic_parity = internal_parity(
                antiholomorphic.words, structure_parity
            )
            holomorphic_power, holomorphic_polynomial = holomorphic.local_data(
                holomorphic_parity, block_order, total_order, series_parameter
            )
            antiholomorphic_power, antiholomorphic_polynomial = (
                antiholomorphic.local_data(antiholomorphic_parity, block_order, total_order, series_parameter)
            )
            holomorphic_binomial = fast._binomial_minus_power(
                one_minus_exponent + holomorphic_extra,
                total_order,
            )
            antiholomorphic_binomial = fast._binomial_minus_power(
                one_minus_exponent + anti_extra,
                total_order,
            )
            holomorphic_series = fast._conv_trunc(
                holomorphic_polynomial,
                holomorphic_binomial,
                total_order,
            )
            antiholomorphic_series = fast._conv_trunc(
                antiholomorphic_polynomial,
                antiholomorphic_binomial,
                total_order,
            )
            terms.append(
                structure
                * coefficient
                * fast._integrate_series_disk(
                    holomorphic_series,
                    antiholomorphic_series,
                    holomorphic_power - zero_exponent,
                    antiholomorphic_power - zero_exponent,
                    epsilon,
                )
            )
    return _stable_complex_fsum(terms)


def _disk_integral(
    data: _ChannelData,
    epsilon: float,
    *,
    block_order: int,
    total_order: int,
    process: str,
    series_parameter: str = "sewing",
) -> complex:
    return _stable_complex_fsum(
        kernel.quadrature_weight
        * _disk_kernel_integral(
            kernel,
            data.energies,
            epsilon,
            block_order=block_order,
            total_order=total_order,
            process=process,
            series_parameter=series_parameter,
        )
        for kernel in data.kernels
    )


@lru_cache(maxsize=None)
def _lens_angular_table(maximum_spin: int, order: int) -> np.ndarray:
    r"""Return Taylor coefficients of the lens angular integrals.

    For integer spin ``m`` the angular factor at fixed radius is

    .. math::

       A_m(r)=\int_{-\arccos(r/2)}^{\arccos(r/2)}e^{im\phi}\,d\phi.

    It is even in ``m``.  For ``m=0`` it is ``2 arccos(r/2)``; for positive
    integer ``m`` it is

    .. math::

       A_m(r)=\frac{2}{m}\sqrt{1-r^2/4}\,U_{m-1}(r/2),

    where ``U`` is a Chebyshev polynomial of the second kind.  The returned
    row ``table[m]`` contains the coefficients through ``r**order``.
    """

    maximum_spin = int(maximum_spin)
    order = int(order)
    if maximum_spin < 0 or order < 0:
        raise ValueError("maximum_spin and order must be nonnegative")

    table = np.zeros((maximum_spin + 1, order + 1), dtype=float)
    table[0, 0] = math.pi
    for k in range((order - 1) // 2 + 1):
        power = 2 * k + 1
        table[0, power] = -math.comb(2 * k, k) / (16**k * (2 * k + 1))

    square_root = np.zeros(order + 1, dtype=float)
    square_root[0] = 1.0
    coefficient = 1.0
    for k in range(1, order // 2 + 1):
        coefficient *= (0.5 - (k - 1)) / k * (-0.25)
        square_root[2 * k] = coefficient

    # Q_n(r)=U_n(r/2) obeys Q_0=1, Q_1=r and
    # Q_n=r Q_{n-1}-Q_{n-2}.
    q_previous = np.asarray([1.0])
    q_current = np.asarray([0.0, 1.0])
    for spin in range(1, maximum_spin + 1):
        table[spin] = (2.0 / spin) * np.convolve(
            q_previous,
            square_root,
        )[: order + 1]
        if spin < maximum_spin:
            shifted = np.pad(q_current, (1, 0))
            previous = np.pad(q_previous, (0, shifted.size - q_previous.size))
            q_next = shifted - previous
            q_previous, q_current = q_current, q_next
    return table


def _integrate_series_lens(
    holomorphic: np.ndarray,
    antiholomorphic: np.ndarray,
    holomorphic_power: complex,
    antiholomorphic_power: complex,
    epsilon: float,
    *,
    angular_order: int | None = None,
) -> complex:
    r"""Meromorphically integrate two local series over the folded lens.

    The powers must differ by an integer, as required by single-valuedness.
    Expanding :func:`_lens_angular_table` gives

    .. math::

       \sum_k A_{m,k}\,
       \frac{\epsilon^{a+b+n+\bar n+2+k}}
            {a+b+n+\bar n+2+k}

    for each pair of series coefficients.  At an exactly logarithmic term,
    the finite part ``log(epsilon)`` is used.
    """

    holomorphic = np.asarray(holomorphic, dtype=complex)
    antiholomorphic = np.asarray(antiholomorphic, dtype=complex)
    if holomorphic.ndim != 1 or antiholomorphic.ndim != 1:
        raise ValueError("local series coefficients must be one-dimensional")
    if not 0 < epsilon < 2:
        raise ValueError("epsilon must lie in (0, 2)")

    spin_difference = complex(holomorphic_power - antiholomorphic_power)
    integer_difference = int(round(spin_difference.real))
    if abs(spin_difference - integer_difference) > 2.0e-8:
        raise ValueError(
            "lens continuation requires integer holomorphic spin; "
            f"received {spin_difference}"
        )

    if angular_order is None:
        angular_order = max(20, len(holomorphic), len(antiholomorphic)) + 8
    angular_order = int(angular_order)
    if angular_order < 0:
        raise ValueError("angular_order must be nonnegative")

    holomorphic_indices = np.arange(holomorphic.size)[:, None]
    antiholomorphic_indices = np.arange(antiholomorphic.size)[None, :]
    spins = np.abs(
        integer_difference + holomorphic_indices - antiholomorphic_indices
    )
    angular_table = _lens_angular_table(int(np.max(spins)), angular_order)
    coefficient_products = holomorphic[:, None] * antiholomorphic[None, :]
    radial_powers = (
        holomorphic_power
        + antiholomorphic_power
        + holomorphic_indices
        + antiholomorphic_indices
        + 2
    ).astype(complex)
    log_epsilon = math.log(epsilon)

    terms: list[complex] = []
    for power in range(angular_order + 1):
        angular_coefficients = angular_table[spins, power]
        if not np.any(angular_coefficients):
            continue
        exponents = radial_powers + power
        radial_integrals = np.empty_like(exponents)
        logarithmic = np.abs(exponents) <= 1.0e-13
        radial_integrals[logarithmic] = log_epsilon
        radial_integrals[~logarithmic] = np.exp(
            exponents[~logarithmic] * log_epsilon
        ) / exponents[~logarithmic]
        terms.append(
            _stable_complex_sum(
                coefficient_products * angular_coefficients * radial_integrals
            )
        )
    return _stable_complex_fsum(terms)


def _crossed_disk_kernel_integral(
    kernel: _SingletKernel,
    energies: Sequence[complex],
    epsilon: float,
    *,
    block_order: int,
    total_order: int,
    process: str,
    series_parameter: str = "sewing",
) -> complex:
    r"""Integrate the local ``y=1-z`` series over the unit-disk lens.

    In crossed coordinates the time factor is

    .. math::

       (1-y)^{-\omega_1\omega_2}
       (1-\bar y)^{-\omega_1\omega_2}
       y^{-\omega_2\omega_3}\bar y^{-\omega_2\omega_3}.

    The PCO primary term supplies one additional ``y**-1`` and the SSVV
    spectator contraction supplies one additional ``ybar**-1``.  Applying
    The folded moduli region is not the full disk ``|y| < epsilon``.  It is

    .. math::

       |y|<\epsilon,\qquad |1-y|<1,

    or, in polar coordinates, ``|arg(y)| < arccos(|y|/2)``.  The angular
    integral is expanded at the origin and the radial monomials are then
    integrated term by term.  This gives the meromorphic continuation even
    when the corresponding radial integral is divergent.
    """

    w1, w2, w3, _ = map(complex, energies)
    zero_exponent = w2 * w3
    one_minus_exponent = w1 * w2
    antiholomorphic_name = _process_antiholomorphic_pattern(process)
    antiholomorphic_zero_extra = 1 if process == "ssvv" else 0
    binomial = fast._binomial_minus_power(one_minus_exponent, total_order)
    terms: list[complex] = []

    for structure_parity, structure in (
        (0, kernel.even_structure),
        (1, kernel.odd_structure),
    ):
        for holomorphic_name, holomorphic_zero_extra, coefficient in (
            ("M", 0, 1.0),
            ("P", 1, w2 * w3),
        ):
            holomorphic = kernel.blocks[holomorphic_name]
            antiholomorphic = kernel.blocks[antiholomorphic_name]
            holomorphic_parity = internal_parity(
                holomorphic.words,
                structure_parity,
            )
            antiholomorphic_parity = internal_parity(
                antiholomorphic.words,
                structure_parity,
            )
            holomorphic_power, holomorphic_polynomial = holomorphic.local_data(
                holomorphic_parity,
                block_order,
                total_order,
                series_parameter,
            )
            antiholomorphic_power, antiholomorphic_polynomial = (
                antiholomorphic.local_data(
                    antiholomorphic_parity,
                    block_order,
                    total_order,
                    series_parameter,
                )
            )
            holomorphic_series = fast._conv_trunc(
                holomorphic_polynomial,
                binomial,
                total_order,
            )
            antiholomorphic_series = fast._conv_trunc(
                antiholomorphic_polynomial,
                binomial,
                total_order,
            )
            terms.append(
                structure
                * coefficient
                * _integrate_series_lens(
                    holomorphic_series,
                    antiholomorphic_series,
                    holomorphic_power
                    - zero_exponent
                    - holomorphic_zero_extra,
                    antiholomorphic_power
                    - zero_exponent
                    - antiholomorphic_zero_extra,
                    epsilon,
                )
            )
    return _stable_complex_fsum(terms)


def _crossed_disk_integral(
    data: _ChannelData,
    epsilon: float,
    *,
    block_order: int,
    total_order: int,
    process: str,
    series_parameter: str = "sewing",
) -> complex:
    return _stable_complex_fsum(
        kernel.quadrature_weight
        * _crossed_disk_kernel_integral(
            kernel,
            data.energies,
            epsilon,
            block_order=block_order,
            total_order=total_order,
            process=process,
            series_parameter=series_parameter,
        )
        for kernel in data.kernels
    )


@dataclass(frozen=True)
class SingletAmplitudeValues:
    """Reduced amplitudes at one numerical truncation."""

    ssvv_raw: complex
    ssss_raw: complex
    ssvv_unit_descendants: complex
    ssss_unit_descendants: complex
    v_to_vss_raw: complex | None
    v_to_vss_unit_descendants: complex | None
    pieces: Mapping[str, Mapping[str, complex]]


@dataclass(frozen=True)
class SingletAmplitudeEvaluation:
    """Production values and optional adjacent-q-order diagnostics."""

    values: SingletAmplitudeValues
    lower_order_values: SingletAmplitudeValues | None
    q_order: int
    lower_q_order: int | None
    maximum_gram_condition: float | None
    maximum_equilibrated_gram_condition: float | None
    high_precision_gram_solve_count: int
    momentum_nodes: int
    build_seconds: float
    integration_seconds: float
    template_seconds: float
    block_backend: str
    maximum_recursion_cancellation: float | None
    high_precision_recursion_node_count: int
    momentum_quadrature: dict[str, Any]
    series_parameter: str = "elliptic_nome"
    numerical_algorithm: str = elliptic_conversion.ALGORITHM_VERSION


def _evaluate_at_order(
    atlas: _AmplitudeAtlas,
    energies: Sequence[complex],
    *,
    order: int,
    epsilon0: float,
    epsilon1: float,
    theta_orders: Sequence[int],
    radial_order: int,
    disk_total_order: int,
    crossed_disk_total_order: int,
    include_v_to_vss: bool = False,
    series_parameter: str = "sewing",
) -> SingletAmplitudeValues:
    elliptic_conversion.validate_representation(series_parameter)
    z, annulus_weights = fast.sewing_annulus_grid(
        epsilon0,
        epsilon1,
        theta_orders,
        radial_order,
    )
    if series_parameter == "elliptic_nome":
        geometry_s = elliptic_conversion.nome_geometry(z)
        geometry_t = elliptic_conversion.nome_geometry(1-z)
        use_t = np.abs(geometry_t[0]) < np.abs(geometry_s[0])
    else:
        geometry_s = geometry_t = (None, None, None)
        use_t = np.abs(1-z) < np.abs(z)
    process_values: dict[str, complex] = {}
    process_pieces: dict[str, dict[str, complex]] = {}
    processes = (
        ("ssvv", "ssss", "v_to_vss")
        if include_v_to_vss
        else ("ssvv", "ssss")
    )
    for process in processes:
        def bulk(s_channel, t_channel):
            # Use the smaller expansion coordinate, while all physical
            # prefactors and the area measure remain in the original plane z.
            pieces = []
            for data, mask, coord, geometry in ((s_channel, ~use_t, z, geometry_s), (t_channel, use_t, 1-z, geometry_t)):
                if np.any(mask):
                    pieces.append(_grid_integral(
                        data, coord[mask], annulus_weights[mask], original_z=z[mask],
                        q=None if geometry[0] is None else geometry[0][mask],
                        theta3=None if geometry[1] is None else geometry[1][mask],
                        order=order, process=process, series_parameter=series_parameter,
                    ))
            return _stable_complex_fsum(pieces)

        original_annulus = bulk(atlas.original_s, atlas.original_t)
        swapped_annulus = bulk(atlas.swapped_s, atlas.swapped_t)
        original_disk = _disk_integral(
            atlas.original_s,
            epsilon0,
            block_order=order,
            total_order=disk_total_order,
            process=process,
            series_parameter=series_parameter,
        )
        swapped_disk = _disk_integral(
            atlas.swapped_s,
            epsilon0,
            block_order=order,
            total_order=disk_total_order,
            process=process,
            series_parameter=series_parameter,
        )
        original_crossed_disk = _crossed_disk_integral(
            atlas.original_t,
            epsilon1,
            block_order=order,
            total_order=crossed_disk_total_order,
            process=process,
            series_parameter=series_parameter,
        )
        swapped_crossed_disk = _crossed_disk_integral(
            atlas.swapped_t,
            epsilon1,
            block_order=order,
            total_order=crossed_disk_total_order,
            process=process,
            series_parameter=series_parameter,
        )
        process_pieces[process] = {
            "original_annulus": original_annulus,
            "swapped_annulus": swapped_annulus,
            "original_disk": original_disk,
            "swapped_disk": swapped_disk,
            "original_crossed_disk": original_crossed_disk,
            "swapped_crossed_disk": swapped_crossed_disk,
        }
        process_values[process] = _stable_complex_fsum(
            process_pieces[process].values()
        )

    w1, w2, w3, w0 = map(complex, energies)
    ssvv_norm = singlet_descendant_norm(w0) * singlet_descendant_norm(w1)
    ssss_norm = ssvv_norm * singlet_descendant_norm(w2) * singlet_descendant_norm(w3)
    v_to_vss_norm = singlet_descendant_norm(w2) * singlet_descendant_norm(
        w3
    )
    v_to_vss_raw = process_values.get("v_to_vss")
    return SingletAmplitudeValues(
        ssvv_raw=process_values["ssvv"],
        ssss_raw=process_values["ssss"],
        ssvv_unit_descendants=process_values["ssvv"] / ssvv_norm,
        ssss_unit_descendants=process_values["ssss"] / ssss_norm,
        v_to_vss_raw=v_to_vss_raw,
        v_to_vss_unit_descendants=(
            None if v_to_vss_raw is None else v_to_vss_raw / v_to_vss_norm
        ),
        pieces=process_pieces,
    )


def evaluate_singlet_amplitudes(
    energies: Sequence[complex],
    *,
    q_order: int = 6,
    lower_q_order: int | None = 5,
    p_nodes: int | str | Sequence[int] = "segmented",
    p_max: float = 4.0,
    p_cut: float = 0.03,
    momentum_scheme: str = "threshold_weighted",
    infinite_gauss_scale: float = 1.0,
    momentum_threshold_options=None,
    epsilon0: float = 0.08,
    epsilon1: float = 0.06,
    theta_orders: Sequence[int] = (15, 15, 60),
    radial_order: int = 24,
    disk_total_order: int = 17,
    crossed_disk_total_order: int = 17,
    gram_condition_limit: float = 1.0e13,
    gram_high_precision_condition: float | None = None,
    gram_high_precision_digits: int = 60,
    gram_high_precision_max_momentum: float | None = None,
    include_v_to_vss: bool = False,
    block_backend: str = "c_recursion",
    recursion_digits: int = 70,
    recursion_reference_p_max: float = 0.18,
    recursion_cancellation_limit: float = 1.0e6,
    series_parameter: str = "elliptic_nome",
) -> SingletAmplitudeEvaluation:
    """Evaluate genuine singlet amplitudes at one kinematic point.

    Set ``include_v_to_vss=True`` to additionally evaluate the ordering with
    vectors on the incoming and first outgoing legs.  It reuses the same
    block atlas but adds one moduli contraction per chart.

    The default block backend is descendant-aware c-recursion.  Select
    ``block_backend='inverse_gram'`` only for independent validation.  The
    ``gram_*`` options affect that oracle only; ``recursion_*`` options control
    the production precision policy.  Unused Gram condition numbers are None.

    ``q_order=N`` retains H_even through qhat**N and H_odd through
    qhat**(N+1/2), obtained algebraically from c-recursion sewing coefficients.
    Plane prefactors and d**2z are unchanged. Analytic OPE patches expand the
    SAME truncated elliptic block to the independent disk_total_order.
    ``series_parameter='sewing'`` is an explicit unresummed validation mode.

    The default momentum rule splits endpoint/bulk/infinite tail, adapted to
    P**beta exp(-a*P**2+s*P). ``momentum_threshold_options`` selects endpoint,
    tail, beta, a, s and optional bulk_breakpoints. An integer p_nodes is the
    TOTAL count; a triple allocates endpoint/bulk/tail separately. p_cut and
    p_max are legacy controls and do not truncate this rule.
    """

    energies = tuple(complex(value) for value in energies)
    elliptic_conversion.validate_representation(series_parameter)
    if len(energies) != 4:
        raise ValueError("energies must contain (omega1,omega2,omega3,omega0)")
    if abs(energies[3] - sum(energies[:3])) > 1.0e-12 * max(
        1.0, *(abs(value) for value in energies)
    ):
        raise ValueError("omega0 must equal omega1+omega2+omega3")
    if q_order < 1:
        raise ValueError("q_order must be positive")
    if recursion_digits < 30 or recursion_reference_p_max < 0:
        raise ValueError("recursion_digits must be >=30 and reference_p_max nonnegative")
    if lower_q_order is not None and not (1 <= lower_q_order < q_order):
        raise ValueError("lower_q_order must be positive and below q_order")
    if (
        gram_high_precision_condition is not None
        and gram_high_precision_condition <= 0
    ):
        raise ValueError("gram_high_precision_condition must be positive")
    if gram_high_precision_digits < 30:
        raise ValueError("gram_high_precision_digits must be at least 30")
    if (
        gram_high_precision_max_momentum is not None
        and gram_high_precision_max_momentum < 0
    ):
        raise ValueError("gram_high_precision_max_momentum must be nonnegative")

    atlas = _build_atlas(
        energies,
        q_order=q_order,
        p_nodes=p_nodes,
        p_max=p_max,
        p_cut=p_cut,
        momentum_scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
        momentum_threshold_options=momentum_threshold_options,
        gram_condition_limit=gram_condition_limit,
        gram_high_precision_condition=gram_high_precision_condition,
        gram_high_precision_digits=gram_high_precision_digits,
        gram_high_precision_max_momentum=gram_high_precision_max_momentum,
        block_backend=block_backend,
        recursion_digits=recursion_digits,
        recursion_reference_p_max=recursion_reference_p_max,
        recursion_cancellation_limit=recursion_cancellation_limit,
    )
    integration_started = time.perf_counter()
    values = _evaluate_at_order(
        atlas,
        energies,
        order=q_order,
        epsilon0=epsilon0,
        epsilon1=epsilon1,
        theta_orders=theta_orders,
        radial_order=radial_order,
        disk_total_order=disk_total_order,
        crossed_disk_total_order=crossed_disk_total_order,
        include_v_to_vss=include_v_to_vss,
        series_parameter=series_parameter,
    )
    lower_values = None
    if lower_q_order is not None:
        lower_values = _evaluate_at_order(
            atlas,
            energies,
            order=lower_q_order,
            epsilon0=epsilon0,
            epsilon1=epsilon1,
            theta_orders=theta_orders,
            radial_order=radial_order,
            disk_total_order=disk_total_order,
            crossed_disk_total_order=crossed_disk_total_order,
            include_v_to_vss=include_v_to_vss,
            series_parameter=series_parameter,
        )
    integration_seconds = time.perf_counter() - integration_started
    return SingletAmplitudeEvaluation(
        values=values,
        lower_order_values=lower_values,
        q_order=q_order,
        lower_q_order=lower_q_order,
        maximum_gram_condition=atlas.maximum_gram_condition,
        maximum_equilibrated_gram_condition=(
            atlas.maximum_equilibrated_gram_condition
        ),
        high_precision_gram_solve_count=atlas.high_precision_gram_solve_count,
        momentum_nodes=len(atlas.original_s.kernels),
        build_seconds=atlas.build_seconds,
        integration_seconds=integration_seconds,
        template_seconds=atlas.template_seconds,
        block_backend=atlas.block_backend,
        maximum_recursion_cancellation=atlas.maximum_recursion_cancellation,
        high_precision_recursion_node_count=atlas.high_precision_recursion_node_count,
        momentum_quadrature=atlas.momentum_quadrature,
        series_parameter=series_parameter,
        numerical_algorithm=(elliptic_conversion.ALGORITHM_VERSION if series_parameter == "elliptic_nome" else "ns_c_recursion_sewing_v1"),
    )


__all__ = [
    "A_WORDS",
    "M_WORDS",
    "O_WORDS",
    "P_WORDS",
    "SingletAmplitudeEvaluation",
    "SingletAmplitudeValues",
    "component_phase",
    "evaluate_singlet_amplitudes",
    "internal_parity",
]
