#!/usr/bin/env python3
r"""Neveu--Schwarz sphere four-point blocks for the Spin(23) scan.

The direct evaluator is the finite-level definition of the chiral block.  In
the ``12 -> 34`` channel, with insertions at ``(0,z,1,infinity)``, it computes

.. math::

   \mathcal F_a(z)=z^{h_p-h_1-h_2}
   \sum_{N\in a/2+\mathbb Z_{\geq0}}z^N
   \rho(\nu_4,\nu_3,\xi_A)
   (B_N^{-1})^{AB}
   \rho(\xi_B,\nu_2,\nu_1).

Here ``a=0`` and ``a=1`` label the integer- and half-integer-level NS
components.  External descendant words may be supplied independently.  The
Spin(23) two-star block uses ``G_-1/2`` on legs 2 and 3.

The accelerated evaluator implements fixed-internal-weight central-charge
recursion.  Its regular term is the component-specific global ``osp(1|2)``
block, including external ``G_-1/2`` insertions.  Residues use the corresponding
NS fusion parities and null-transport signs in the direct Ward convention.

The crossed channel is a separate expansion in ``x=1-z`` with standardized
local coordinates and external ordering ``(3,2,1,4)``.  A single conformal
block is not crossing invariant; only the complete spectral integral may be
compared between channels.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.linalg import lu_factor, lu_solve

from ns_algebra.ns_fusion import A_rs, h_rs, sigma
from ns_algebra.ns_sca import G, L, Word, gram_matrix, twice_level
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_general_kinematics import BlockValues
from spin23_ns_c_recursion import RecursionPoleCollision, block_coefficients


Component = Literal["even", "odd"]
Method = Literal["direct", "global_beta", "recursion"]
Channel = Literal["s", "t"]

EMPTY_WORD: Word = ()
G_MINUS_HALF: Word = (G(sp.Rational(-1, 2)),)


def _component_parity(component: Component | str) -> int:
    aliases = {"even": 0, "F^1": 0, "odd": 1, "F^{1/2}": 1}
    try:
        return aliases[str(component)]
    except KeyError as error:
        raise ValueError("component must be 'even' or 'odd'") from error


def _validate_cutoff(maximum_twice_level: int) -> int:
    if not isinstance(maximum_twice_level, int):
        raise TypeError("maximum_twice_level must be an integer")
    if maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be nonnegative")
    return maximum_twice_level


def _validate_external_data(
    weights: Sequence[complex],
    words: Sequence[Word],
) -> tuple[tuple[complex, ...], tuple[Word, ...]]:
    weights_tuple = tuple(complex(value) for value in weights)
    words_tuple = tuple(tuple(word) for word in words)
    if len(weights_tuple) != 4 or len(words_tuple) != 4:
        raise ValueError("four external weights and four descendant words are required")
    return weights_tuple, words_tuple


def primary_external_words() -> tuple[Word, Word, Word, Word]:
    """Return primary words for external legs ``(1,2,3,4)``."""

    return (EMPTY_WORD,) * 4


def two_star_external_words() -> tuple[Word, Word, Word, Word]:
    r"""Return words with ``G_-1/2`` on external legs 2 and 3."""

    return (EMPTY_WORD, G_MINUS_HALF, G_MINUS_HALF, EMPTY_WORD)


def channel_coordinates(
    z: complex,
    external_weights: Sequence[complex],
    external_words: Sequence[Word],
    *,
    channel: Channel,
) -> tuple[complex, tuple[complex, ...], tuple[Word, ...]]:
    r"""Return the standard coordinate and ordering for one OPE channel.

    ``s`` is the ``12 -> 34`` expansion at ``z=0``.  For ``t`` we apply
    ``y=1-z`` and use the standard ordered positions
    ``(old 3, old 2, old 1, old 4)=(0,y,1,infinity)``.

    The returned words are expressed in independently standardized local
    coordinates.  Any phases associated with a separate choice of square
    root for the supercoordinate map must be applied by the caller.
    """

    weights, words = _validate_external_data(external_weights, external_words)
    if channel == "s":
        return complex(z), weights, words
    if channel == "t":
        permutation = (2, 1, 0, 3)
        return (
            1.0 - complex(z),
            tuple(weights[index] for index in permutation),
            tuple(words[index] for index in permutation),
        )
    raise ValueError("channel must be 's' or 't'")


def _as_numeric(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite NS block value {value!r}")
    return value


def _three_point_vectors(
    basis: Sequence[Word],
    *,
    c: complex,
    h_internal: complex,
    external_weights: tuple[complex, ...],
    external_words: tuple[Word, ...],
    digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    h1, h2, h3, h4 = external_weights
    word1, word2, word3, word4 = external_words
    left = []
    right = []
    for internal_word in basis:
        left.append(
            _as_numeric(
                ns_three_point(
                    word4,
                    word3,
                    internal_word,
                    h_infinity=h4,
                    h_middle=h3,
                    h_zero=h_internal,
                    c=c,
                ),
                digits,
            )
        )
        right.append(
            _as_numeric(
                ns_three_point(
                    internal_word,
                    word2,
                    word1,
                    h_infinity=h_internal,
                    h_middle=h2,
                    h_zero=h1,
                    c=c,
                ),
                digits,
            )
        )
    return np.asarray(left, dtype=np.complex128), np.asarray(right, dtype=np.complex128)


@lru_cache(maxsize=None)
def _factored_gram_data(
    twice_descendant_level: int,
    c: complex,
    h_internal: complex,
    digits: int,
) -> tuple[tuple[Word, ...], np.ndarray, np.ndarray, float]:
    """Return a cached PBW basis, LU-factored Gram matrix, and condition number.

    At fixed internal momentum the same Gram matrix is used by every choice
    of external components.  Caching one pivoted LU factorization removes the
    dominant duplicated linear-algebra work in the genuine singlet
    correlators without explicitly forming the numerically less stable matrix
    inverse.
    """

    basis, gram = gram_matrix(
        twice_descendant_level,
        h=sp.sympify(h_internal),
        c=sp.sympify(c),
    )
    matrix = np.asarray(
        [
            [_as_numeric(gram[row, column], digits) for column in range(gram.cols)]
            for row in range(gram.rows)
        ],
        dtype=np.complex128,
    )
    condition = float(np.linalg.cond(matrix))
    lu, pivots = lu_factor(matrix, check_finite=False)
    return tuple(basis), lu, pivots, condition


def _direct_level_coefficient(
    twice_level: int,
    *,
    c: complex,
    h_internal: complex,
    external_weights: tuple[complex, ...],
    external_words: tuple[Word, ...],
    digits: int,
    condition_limit: float,
    basis_override: Sequence[Word] | None = None,
) -> tuple[complex, float]:
    if basis_override is None:
        basis, lu, pivots, condition = _factored_gram_data(
            twice_level,
            complex(c),
            complex(h_internal),
            int(digits),
        )
        matrix = None
    else:
        basis, gram = gram_matrix(
            twice_level,
            h=sp.sympify(h_internal),
            c=sp.sympify(c),
        )
        indices = [basis.index(tuple(word)) for word in basis_override]
        basis = [basis[index] for index in indices]
        gram = gram.extract(indices, indices)
        matrix = np.asarray(
            [
                [_as_numeric(gram[row, column], digits) for column in range(gram.cols)]
                for row in range(gram.rows)
            ],
            dtype=np.complex128,
        )
        lu = pivots = None
        condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            f"NS Gram matrix at level {twice_level}/2 has condition number "
            f"{condition:.3e}, above the limit {condition_limit:.3e}"
        )
    left, right = _three_point_vectors(
        basis,
        c=c,
        h_internal=h_internal,
        external_weights=external_weights,
        external_words=external_words,
        digits=digits,
    )
    if lu is None or pivots is None:
        if matrix is None:  # pragma: no cover - guarded by the branches above
            raise AssertionError("missing Gram matrix")
        coefficient = left @ np.linalg.solve(matrix, right)
    else:
        coefficient = left @ lu_solve((lu, pivots), right, check_finite=False)
    return complex(coefficient), condition


@dataclass(frozen=True)
class SphereBlockSeries:
    """A component of a normalized NS sphere block as a finite series."""

    coefficients: Mapping[int, complex]
    component: Component
    method: Method
    h_internal: complex
    external_weights: tuple[complex, complex, complex, complex]
    external_words: tuple[Word, Word, Word, Word]
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def descendant_value(self, z: complex) -> complex:
        """Evaluate only the descendant series, without its primary power."""

        z = complex(z)
        if z == 0:
            if any(level < 0 for level in self.coefficients):
                raise ValueError("negative levels cannot be evaluated at z=0")
            return complex(self.coefficients.get(0, 0.0j))
        log_z = cmath.log(z)
        return sum(
            coefficient * cmath.exp(0.5 * twice_level * log_z)
            for twice_level, coefficient in self.coefficients.items()
        )

    def value(self, z: complex, *, include_primary_power: bool = True) -> complex:
        """Evaluate the block with a fixed principal branch of ``log(z)``."""

        descendant = self.descendant_value(z)
        if not include_primary_power:
            return descendant
        h1, h2, _, _ = self.external_weights
        word1, word2, _, _ = self.external_words
        effective_h1 = h1 + 0.5 * twice_level(word1) if word1 else h1
        effective_h2 = h2 + 0.5 * twice_level(word2) if word2 else h2
        if z == 0:
            raise ValueError("the primary block power is singular/ambiguous at z=0")
        return cmath.exp(
            (self.h_internal - effective_h1 - effective_h2) * cmath.log(z)
        ) * descendant


def direct_sphere_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    maximum_twice_level: int,
    component: Component,
    external_words: Sequence[Word] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> SphereBlockSeries:
    """Construct the finite-level inverse-Gram definition of one NS block."""

    maximum_twice_level = _validate_cutoff(maximum_twice_level)
    parity = _component_parity(component)
    words = primary_external_words() if external_words is None else external_words
    weights, words_tuple = _validate_external_data(external_weights, words)
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}
    for twice_level in range(parity, maximum_twice_level + 1, 2):
        coefficient, condition = _direct_level_coefficient(
            twice_level,
            c=complex(c),
            h_internal=complex(h_internal),
            external_weights=weights,
            external_words=words_tuple,
            digits=int(digits),
            condition_limit=float(condition_limit),
        )
        coefficients[twice_level] = coefficient
        conditions[twice_level] = condition
    return SphereBlockSeries(
        coefficients=coefficients,
        component="even" if parity == 0 else "odd",
        method="direct",
        h_internal=complex(h_internal),
        external_weights=weights,  # type: ignore[arg-type]
        external_words=words_tuple,  # type: ignore[arg-type]
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def _global_word(twice_level: int) -> Word:
    n, beta = divmod(twice_level, 2)
    return (L(-1),) * n + (G_MINUS_HALF if beta else EMPTY_WORD)


def _global_seed_coefficients(
    *,
    h_internal: complex,
    external_weights: tuple[complex, ...],
    external_words: tuple[Word, ...],
    maximum_twice_level: int,
    component_parity: int,
    digits: int,
) -> dict[int, complex]:
    coefficients: dict[int, complex] = {}
    for twice_level in range(component_parity, maximum_twice_level + 1, 2):
        coefficient, _ = _direct_level_coefficient(
            twice_level,
            c=0.0,
            h_internal=h_internal,
            external_weights=external_weights,
            external_words=external_words,
            digits=digits,
            condition_limit=float("inf"),
            basis_override=(_global_word(twice_level),),
        )
        coefficients[twice_level] = coefficient
    return coefficients


def global_sphere_block_beta_series(
    *,
    h_internal: complex,
    external_weights: Sequence[complex],
    maximum_twice_level: int,
    beta: int,
    external_words: Sequence[Word] | None = None,
    digits: int = 50,
) -> SphereBlockSeries:
    r"""Return one parity-resolved ``osp(1|2)`` four-point light block.

    ``beta=0`` selects integer levels and ``beta=1`` half-integer levels.
    These are the two terms in the global sum of Belavin--Geiko eq. (3.8),
    not a separately named recursion in the published literature.
    """

    maximum_twice_level = _validate_cutoff(maximum_twice_level)
    if beta not in (0, 1):
        raise ValueError("beta must be 0 or 1")
    parity = int(beta)
    words = primary_external_words() if external_words is None else external_words
    weights, words_tuple = _validate_external_data(external_weights, words)
    coefficients = _global_seed_coefficients(
        h_internal=complex(h_internal),
        external_weights=weights,
        external_words=words_tuple,
        maximum_twice_level=maximum_twice_level,
        component_parity=parity,
        digits=int(digits),
    )
    return SphereBlockSeries(
        coefficients=coefficients,
        component="even" if parity == 0 else "odd",
        method="global_beta",
        h_internal=complex(h_internal),
        external_weights=weights,  # type: ignore[arg-type]
        external_words=words_tuple,  # type: ignore[arg-type]
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers={},
    )


@dataclass(frozen=True)
class SphereHPoleData:
    """Fixed-central-charge singular data for one four-point Kac pole."""

    r: int
    s: int
    h_pole: complex
    residue: complex
    null_twice_level: int
    shifted_component_parity: int


def sphere_h_pole_data(
    *,
    b: complex,
    r: int,
    s: int,
    external_weights: Sequence[complex],
    component: Component,
    digits: int = 50,
) -> SphereHPoleData:
    """Return the published primary-external singular part of h-recursion.

    This supplies the Kac pole, null residue, and parity transport.  It does
    not supply the large-h regular term needed to close an elliptic
    recursion.
    """

    parity = _component_parity(component)
    weights, _ = _validate_external_data(external_weights, primary_external_words())
    b_symbol = sp.sympify(b)
    null_twice_level = r * s
    shifted_parity = parity ^ (null_twice_level % 2)
    h_pole = _as_numeric(h_rs(r, s, b_symbol), digits)
    h1, h2, h3, h4 = weights
    residue = _as_numeric(
        sp.cancel(
            A_rs(r, s, b_symbol)
            * sigma(r, s, shifted_parity, h2, h1, b_symbol)
            * sigma(r, s, shifted_parity, h3, h4, b_symbol)
        ),
        digits,
    )
    return SphereHPoleData(
        r=r,
        s=s,
        h_pole=h_pole,
        residue=residue,
        null_twice_level=null_twice_level,
        shifted_component_parity=shifted_parity,
    )


def recursive_sphere_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    maximum_twice_level: int,
    component: Component,
    external_words: Sequence[Word] | None = None,
    digits: int = 50,
    pole_tolerance: float = 1.0e-11,
) -> SphereBlockSeries:
    r"""Construct a sphere block with fixed-weight NS ``c``-recursion.

    The result is a power-series truncation through ``maximum_twice_level``.
    Near coincident recursion poles are rejected rather than silently
    regulated; collision-aware continuation should be added only with an
    independent confluent-limit test.
    """

    maximum_twice_level = _validate_cutoff(maximum_twice_level)
    initial_parity = _component_parity(component)
    words = primary_external_words() if external_words is None else external_words
    weights, words_tuple = _validate_external_data(external_weights, words)
    if any(word not in (EMPTY_WORD, G_MINUS_HALF) for word in words_tuple):
        raise NotImplementedError(
            "c-recursion supports primary and G_-1/2 external states; "
            "use method='direct' for higher external descendants"
        )
    alphas = tuple(int(bool(word)) for word in words_tuple)
    result = block_coefficients(
        c=c,
        h_internal=h_internal,
        external_weights=weights,
        external_patterns=(alphas,),
        maximum_twice_level=maximum_twice_level,
        force_high_precision=True,
        digits=digits,
        pole_tolerance=pole_tolerance,
    )
    coefficients = {n: value for n, value in enumerate(result.coefficients[alphas])
                    if n % 2 == initial_parity}
    return SphereBlockSeries(
        coefficients=coefficients,
        component="even" if initial_parity == 0 else "odd",
        method="recursion",
        h_internal=complex(h_internal),
        external_weights=weights,  # type: ignore[arg-type]
        external_words=words_tuple,  # type: ignore[arg-type]
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers={},
    )


def sphere_block_series(
    *,
    method: Method,
    **kwargs: object,
) -> SphereBlockSeries:
    """Dispatch to the direct oracle, global beta seed, or c-recursion."""

    if method == "direct":
        return direct_sphere_block_series(**kwargs)  # type: ignore[arg-type]
    if method == "global_beta":
        return global_sphere_block_beta_series(**kwargs)  # type: ignore[arg-type]
    if method == "recursion":
        return recursive_sphere_block_series(**kwargs)  # type: ignore[arg-type]
    raise ValueError("method must be 'direct', 'global_beta', or 'recursion'")


@dataclass(frozen=True)
class Spin23NSBlockBundle:
    """The five chiral series required by ``assemble_channel_densities``."""

    even: SphereBlockSeries
    odd: SphereBlockSeries
    even_star: SphereBlockSeries
    odd_star: SphereBlockSeries
    even_bar: SphereBlockSeries
    odd_bar: SphereBlockSeries
    channel: Channel

    def values(self, coordinate: complex) -> BlockValues:
        """Evaluate all series at one standardized channel coordinate."""

        coordinate = complex(coordinate)
        return BlockValues(
            even=self.even.value(coordinate),
            odd=self.odd.value(coordinate),
            even_bar=self.even_bar.value(coordinate.conjugate()),
            odd_bar=self.odd_bar.value(coordinate.conjugate()),
            even_star=self.even_star.value(coordinate),
            odd_star=self.odd_star.value(coordinate),
        )

    def values_at_z(self, z: complex) -> BlockValues:
        """Evaluate at the original cross ratio, routing to this channel."""

        coordinate = complex(z) if self.channel == "s" else 1.0 - complex(z)
        return self.values(coordinate)


def build_spin23_block_bundle(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    maximum_twice_level: int,
    method: Method = "recursion",
    starred_method: Method | None = None,
    channel: Channel = "s",
    anti_c: complex | None = None,
    anti_h_internal: complex | None = None,
    anti_external_weights: Sequence[complex] | None = None,
    digits: int = 50,
) -> Spin23NSBlockBundle:
    """Build the primary, two-star, and anti-holomorphic block components.

    Analytically continued calculations should pass anti-holomorphic weights
    explicitly if they are not identified with the holomorphic weights.  The
    default evaluates the same weight data at the conjugate coordinate; it
    does not complex-conjugate the weights.

    Both primary and external-G_-1/2 blocks default to ``c``-recursion.  An
    explicit ``starred_method`` override is useful for inverse-Gram tests.
    """

    if starred_method is None:
        starred_method = method

    weights, primary_words = _validate_external_data(
        external_weights,
        primary_external_words(),
    )
    _, star_words = _validate_external_data(external_weights, two_star_external_words())
    _, channel_weights, channel_primary_words = channel_coordinates(
        0.25,
        weights,
        primary_words,
        channel=channel,
    )
    _, _, channel_star_words = channel_coordinates(
        0.25,
        weights,
        star_words,
        channel=channel,
    )

    anti_weights_input = weights if anti_external_weights is None else anti_external_weights
    anti_weights, anti_words = _validate_external_data(
        anti_weights_input,
        primary_external_words(),
    )
    _, channel_anti_weights, channel_anti_words = channel_coordinates(
        0.25,
        anti_weights,
        anti_words,
        channel=channel,
    )

    common = {
        "c": c,
        "h_internal": h_internal,
        "external_weights": channel_weights,
        "maximum_twice_level": maximum_twice_level,
        "method": method,
        "digits": digits,
    }
    anti_common = {
        "c": c if anti_c is None else anti_c,
        "h_internal": h_internal if anti_h_internal is None else anti_h_internal,
        "external_weights": channel_anti_weights,
        "maximum_twice_level": maximum_twice_level,
        "method": method,
        "digits": digits,
    }
    return Spin23NSBlockBundle(
        even=sphere_block_series(
            **common,
            component="even",
            external_words=channel_primary_words,
        ),
        odd=sphere_block_series(
            **common,
            component="odd",
            external_words=channel_primary_words,
        ),
        even_star=sphere_block_series(
            **{**common, "method": starred_method},
            component="even",
            external_words=channel_star_words,
        ),
        odd_star=sphere_block_series(
            **{**common, "method": starred_method},
            component="odd",
            external_words=channel_star_words,
        ),
        even_bar=sphere_block_series(
            **anti_common,
            component="even",
            external_words=channel_anti_words,
        ),
        odd_bar=sphere_block_series(
            **anti_common,
            component="odd",
            external_words=channel_anti_words,
        ),
        channel=channel,
    )


def relative_series_difference(
    left: SphereBlockSeries,
    right: SphereBlockSeries,
    *,
    floor: float = 1.0e-15,
) -> float:
    """Return the maximum coefficientwise relative difference."""

    levels = set(left.coefficients) | set(right.coefficients)
    if not levels:
        return 0.0
    return max(
        abs(left.coefficients.get(level, 0.0j) - right.coefficients.get(level, 0.0j))
        / max(
            floor,
            abs(left.coefficients.get(level, 0.0j)),
            abs(right.coefficients.get(level, 0.0j)),
        )
        for level in levels
    )


__all__ = [
    "Channel",
    "Component",
    "EMPTY_WORD",
    "G_MINUS_HALF",
    "Method",
    "RecursionPoleCollision",
    "SphereBlockSeries",
    "SphereHPoleData",
    "Spin23NSBlockBundle",
    "build_spin23_block_bundle",
    "channel_coordinates",
    "direct_sphere_block_series",
    "global_sphere_block_beta_series",
    "primary_external_words",
    "recursive_sphere_block_series",
    "relative_series_difference",
    "sphere_h_pole_data",
    "sphere_block_series",
    "two_star_external_words",
]
