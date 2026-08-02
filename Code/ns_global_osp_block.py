#!/usr/bin/env python3
"""Global osp(1|2) building blocks in the CCY plumbing frame.

The module implements the explicit Neveu--Schwarz analogue of the global
SL(2) kernel used by Cho--Collier--Yin.  A global state is

    L_-1^n G_-1/2^epsilon |h>,    n >= 0, epsilon in {0, 1}.

The main routine :func:`osp_three_point` evaluates the plane three-point
form with arbitrary global descendants on the three slots (infinity, one,
zero).  :func:`trifundamental_coefficient` then assembles the direct NS
analogue of CCY's sphere six-point trifundamental coefficient.

Conventions start from Appendix A of Belavin--Geiko, arXiv:1806.09563, but
the component tensors are converted to a *fixed-parity trilinear-form*
convention before the parity projector is applied.  This distinction changes
the reflected ``(1,b,0)`` kernels and replaces ``S+1/2`` by ``S-1/2`` in the
``(1,1,1)`` entry.  The converted tensors obey odd-null factorization; using
the unconverted component-map table with the projector does not.  Rising and
falling Pochhammer symbols are kept distinct throughout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence, Union


Number = Union[complex, float, int]


def rising(value: Number, order: int) -> complex:
    """Return the rising Pochhammer symbol ``(value)_order``."""

    if order < 0:
        raise ValueError("order must be non-negative")
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= complex(value) + offset
    return result


def falling(value: Number, order: int) -> complex:
    """Return the falling Pochhammer symbol ``value^(underline order)``."""

    if order < 0:
        raise ValueError("order must be non-negative")
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= complex(value) - offset
    return result


def _validate_state(n: int, epsilon: int) -> None:
    if n < 0:
        raise ValueError("the L_-1 occupation must be non-negative")
    if epsilon not in (0, 1):
        raise ValueError("the G_-1/2 occupation must be 0 or 1")


def osp_norm(weight: Number, n: int, epsilon: int) -> complex:
    r"""Norm of ``L_-1^n G_-1/2^epsilon |h>``.

    It equals ``n! (2h)_(n+epsilon)``.  The odd state is an SL(2)
    primary of weight ``h+1/2`` and has primary norm ``2h``.
    """

    _validate_state(n, epsilon)
    return math.factorial(n) * rising(2.0 * complex(weight), n + epsilon)


def osp_raising_coefficients(
    weight: Number, n: int, epsilon: int
) -> tuple[complex, complex]:
    r"""Coefficients of L_1 and G_1/2 acting on a global basis state.

    The returned pair ``(l_coefficient, g_coefficient)`` means

    * ``L_1 |n,eps> = l_coefficient |n-1,eps>``;
    * for eps=0, ``G_1/2 |n,0> = g_coefficient |n-1,1>``;
    * for eps=1, ``G_1/2 |n,1> = g_coefficient |n,0>``.

    A state with a negative occupation is understood to vanish.
    """

    _validate_state(n, epsilon)
    h = complex(weight)
    if epsilon == 0:
        return n * (2.0 * h + n - 1.0), complex(n)
    return n * (2.0 * h + n), 2.0 * h + n


def osp_two_chain_kernel(
    *,
    k: int,
    m: int,
    epsilon1: int,
    epsilon2: int,
    epsilon3: int,
    d1: Number,
    d2: Number,
    d3: Number,
) -> complex:
    r"""Return the reduced global three-point kernel tau_(k,m).

    This evaluates

        rho(G_-1/2^e1 L_-1^k nu_1,
            G_-1/2^e2 nu_2,
            G_-1/2^e3 L_-1^m nu_3)

    at (infinity, 1, 0).  Both independent primary three-point structures
    are normalized to one.  A physical vertex sector is selected separately
    by :func:`osp_sector_vertex`.
    """

    _validate_state(k, epsilon1)
    _validate_state(0, epsilon2)
    _validate_state(m, epsilon3)
    h1 = complex(d1)
    h2 = complex(d2)
    h3 = complex(d3)
    a_value = h2 + h3 - h1
    b_value = h1 + h2 - h3
    c_value = h1 - h2 + h3
    s_value = h1 + h2 + h3
    bits = (epsilon1, epsilon2, epsilon3)

    # In the fixed-parity trilinear-form convention, reflection of the two
    # endpoints gives
    #
    #   T^{a b c}_{k,0,m}(d1,d2,d3)
    #     = (-1)^{b(a+c)}
    #       T^{c b a}_{m,0,k}(d3,d2,d1).
    #
    # Applying this when (a,c)=(1,0) reduces the kernel to one of the four
    # a=0 formulas below.  It is precisely the term missed if the published
    # component-map kernels are combined with a parity projector unchanged.
    if bits in ((1, 0, 0), (1, 1, 0)):
        reflection_sign = -1.0 if epsilon2 else 1.0
        return reflection_sign * osp_two_chain_kernel(
            k=m,
            m=k,
            epsilon1=0,
            epsilon2=epsilon2,
            epsilon3=1,
            d1=d3,
            d2=d2,
            d3=d1,
        )

    result = 0.0 + 0.0j
    for p in range(min(k, m) + 1):
        common0 = (
            math.comb(k, p)
            * falling(m, p)
            * falling(2.0 * h3 + m - 1.0, p)
        )
        common1 = (
            math.comb(k, p)
            * falling(m, p)
            * falling(2.0 * h3 + m, p)
        )

        if bits == (0, 0, 0):
            term = common0 * rising(a_value, m - p) * rising(
                b_value + p - m, k - p
            )
        elif bits == (1, 0, 0):  # handled by reflection above
            raise AssertionError("unreachable reflected kernel")
        elif bits == (0, 1, 0):
            term = common0 * rising(a_value + 0.5, m - p) * rising(
                b_value + 0.5 + p - m, k - p
            )
        elif bits == (0, 0, 1):
            term = common1 * rising(a_value + 0.5, m - p) * rising(
                b_value - 0.5 + p - m, k - p
            )
        elif bits == (1, 1, 0):  # handled by reflection above
            raise AssertionError("unreachable reflected kernel")
        elif bits == (1, 0, 1):
            term = c_value * common1 * rising(a_value, m - p) * rising(
                b_value + p - m, k - p
            )
        elif bits == (0, 1, 1):
            term = -common1 * rising(a_value, m - p + 1) * rising(
                b_value + p - m, k - p
            )
        elif bits == (1, 1, 1):
            term = (s_value - 0.5) * common1 * rising(
                a_value + 0.5, m - p
            ) * rising(b_value + 0.5 + p - m, k - p)
        else:  # pragma: no cover - validation makes this unreachable
            raise AssertionError("unreachable fermion-label branch")
        result += term
    return result


def osp_three_point(
    *,
    n1: int,
    n2: int,
    n3: int,
    epsilon1: int,
    epsilon2: int,
    epsilon3: int,
    d1: Number,
    d2: Number,
    d3: Number,
) -> complex:
    r"""Return the global three-point form with three arbitrary chains.

    Slot 1 is the BPZ-conjugate state at infinity, slot 2 is inserted at
    one, and slot 3 is the ket at zero.  Translation covariance supplies
    the falling Pochhammer multiplying the two-chain kernel.
    """

    _validate_state(n1, epsilon1)
    _validate_state(n2, epsilon2)
    _validate_state(n3, epsilon3)
    exponent = (
        complex(d1)
        - complex(d2)
        - complex(d3)
        + 0.5 * (epsilon1 - epsilon2 - epsilon3)
        + n1
        - n3
    )
    return falling(exponent, n2) * osp_two_chain_kernel(
        k=n1,
        m=n3,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
        d1=d1,
        d2=d2,
        d3=d3,
    )


def osp_sector_vertex(*, sector: int, **three_point_arguments: Number) -> complex:
    """Evaluate one of the two parity-homogeneous NS trinion tensors."""

    if sector not in (0, 1):
        raise ValueError("sector must be 0 or 1")
    fermion_parity = sum(
        int(three_point_arguments[key])
        for key in ("epsilon1", "epsilon2", "epsilon3")
    ) % 2
    if fermion_parity != sector:
        return 0.0 + 0.0j
    return osp_three_point(**three_point_arguments)


def bottom_endpoint_factor(
    *, n: int, epsilon: int, internal_weight: Number, middle_weight: Number, ket_weight: Number
) -> complex:
    """Three-point factor for an internal descendant and two bottom fields."""

    _validate_state(n, epsilon)
    return rising(
        complex(internal_weight)
        + complex(middle_weight)
        - complex(ket_weight)
        + 0.5 * epsilon,
        n,
    )


def trifundamental_coefficient(
    *,
    occupations: Sequence[int],
    fermions: Sequence[int],
    internal_weights: Sequence[Number],
    external_pairs: Sequence[tuple[Number, Number]],
    outer_sectors: Sequence[int],
    central_sector: int,
) -> complex:
    r"""One coefficient of the NS sphere six-point trifundamental block.

    ``external_pairs[i]`` is ``(d_(2i+1), d_(2i+2))``: the first weight is
    at the ket slot and the second at the middle slot of the corresponding
    outer trinion.  All six external fields are bottom components.
    """

    if not all(len(values) == 3 for values in (occupations, fermions, internal_weights)):
        raise ValueError("occupations, fermions, and internal_weights must have length 3")
    if len(external_pairs) != 3 or len(outer_sectors) != 3:
        raise ValueError("external_pairs and outer_sectors must have length 3")
    if central_sector not in (0, 1) or any(value not in (0, 1) for value in outer_sectors):
        raise ValueError("all sector labels must be 0 or 1")

    n1, n2, n3 = (int(value) for value in occupations)
    e1, e2, e3 = (int(value) for value in fermions)
    h1, h2, h3 = (complex(value) for value in internal_weights)
    for n, epsilon in zip((n1, n2, n3), (e1, e2, e3)):
        _validate_state(n, epsilon)

    # Two bottom external states force the internal fermion label at each
    # outer trinion to equal that trinion's parity sector.
    if tuple((e1, e2, e3)) != tuple(int(value) for value in outer_sectors):
        return 0.0 + 0.0j
    if (e1 + e2 + e3) % 2 != central_sector:
        return 0.0 + 0.0j

    endpoint_product = 1.0 + 0.0j
    for n, epsilon, h, (ket_weight, middle_weight) in zip(
        (n1, n2, n3),
        (e1, e2, e3),
        (h1, h2, h3),
        external_pairs,
    ):
        endpoint_product *= bottom_endpoint_factor(
            n=n,
            epsilon=epsilon,
            internal_weight=h,
            middle_weight=middle_weight,
            ket_weight=ket_weight,
        )

    central = osp_three_point(
        n1=n1,
        n2=n2,
        n3=n3,
        epsilon1=e1,
        epsilon2=e2,
        epsilon3=e3,
        d1=h1,
        d2=h2,
        d3=h3,
    )
    denominator = (
        osp_norm(h1, n1, e1)
        * osp_norm(h2, n2, e2)
        * osp_norm(h3, n3, e3)
    )
    return endpoint_product * central / denominator


def trifundamental_component_coefficient(
    *,
    occupations: Sequence[int],
    fermions: Sequence[int],
    internal_weights: Sequence[Number],
    external_pairs: Sequence[tuple[Number, Number]],
) -> complex:
    """Return a term in the component-normalized direct-sum convention.

    Belavin--Geiko set both parity-homogeneous primary three-forms to one
    before sewing.  At a fixed fermion assignment this is equivalent to
    choosing the unique compatible sector at each trinion; summing the
    returned terms over all three fermion labels gives their convention.
    """

    if len(fermions) != 3:
        raise ValueError("fermions must have length 3")
    normalized_fermions = tuple(int(value) for value in fermions)
    return trifundamental_coefficient(
        occupations=occupations,
        fermions=normalized_fermions,
        internal_weights=internal_weights,
        external_pairs=external_pairs,
        outer_sectors=normalized_fermions,
        central_sector=sum(normalized_fermions) % 2,
    )


@dataclass(frozen=True)
class GlobalCheckSummary:
    norm_recursion_error: float
    endpoint_formula_error: float
    primary_ward_table_error: float
    ccy_middle_chain_error: float
    torus_half_level_error: float
    torus_level_one_error: float
    parity_selection_passed: bool
    sample_even_trifundamental_coefficient: complex
    sample_odd_trifundamental_coefficient: complex


def run_checks() -> GlobalCheckSummary:
    """Check the global formulas against algebraic and published identities."""

    h = 0.83
    norm_errors: list[float] = []
    for epsilon in (0, 1):
        for n in range(1, 6):
            ratio = osp_norm(h, n, epsilon) / osp_norm(h, n - 1, epsilon)
            expected = n * (2.0 * h + n - 1.0 + epsilon)
            norm_errors.append(abs(ratio - expected))

    d1, d2, d3 = 0.83, 0.47, 0.29
    endpoint_errors: list[float] = []
    for epsilon in (0, 1):
        for n in range(6):
            table_value = osp_two_chain_kernel(
                k=n,
                m=0,
                epsilon1=epsilon,
                epsilon2=0,
                epsilon3=0,
                d1=d1,
                d2=d2,
                d3=d3,
            )
            endpoint_errors.append(
                abs(
                    table_value
                    - bottom_endpoint_factor(
                        n=n,
                        epsilon=epsilon,
                        internal_weight=d1,
                        middle_weight=d2,
                        ket_weight=d3,
                    )
                )
            )

    a_value = d2 + d3 - d1
    b_value = d1 + d2 - d3
    c_value = d1 - d2 + d3
    s_value = d1 + d2 + d3
    primary_expected = {
        (0, 0, 0): 1.0,
        (1, 0, 0): 1.0,
        (0, 1, 0): 1.0,
        (0, 0, 1): 1.0,
        (1, 1, 0): b_value,
        (1, 0, 1): c_value,
        (0, 1, 1): -a_value,
        (1, 1, 1): s_value - 0.5,
    }
    primary_errors = []
    for bits, expected in primary_expected.items():
        primary_errors.append(
            abs(
                osp_two_chain_kernel(
                    k=0,
                    m=0,
                    epsilon1=bits[0],
                    epsilon2=bits[1],
                    epsilon3=bits[2],
                    d1=d1,
                    d2=d2,
                    d3=d3,
                )
                - expected
            )
        )

    # CCY Eq. (4.20): the middle L_-1 chain is an ordinary derivative.
    # This check catches the off-by-one shift printed in Eq. (A.6) of
    # Belavin--Geiko: the falling factorial must start at the scaling
    # exponent before the derivatives are applied.
    middle_errors: list[float] = []
    for k in range(4):
        for middle in range(4):
            for m in range(4):
                reduced = osp_two_chain_kernel(
                    k=k,
                    m=m,
                    epsilon1=0,
                    epsilon2=0,
                    epsilon3=0,
                    d1=d1,
                    d2=d2,
                    d3=d3,
                )
                ccy_factor = rising(
                    d1 + k - d2 - middle + 1.0 - d3 - m,
                    middle,
                )
                middle_errors.append(
                    abs(
                        osp_three_point(
                            n1=k,
                            n2=middle,
                            n3=m,
                            epsilon1=0,
                            epsilon2=0,
                            epsilon3=0,
                            d1=d1,
                            d2=d2,
                            d3=d3,
                        )
                        - ccy_factor * reduced
                    )
                )

    torus_internal = 0.83
    insertion = 0.47
    torus_half = osp_three_point(
        n1=0,
        n2=0,
        n3=0,
        epsilon1=1,
        epsilon2=0,
        epsilon3=1,
        d1=torus_internal,
        d2=insertion,
        d3=torus_internal,
    ) / osp_norm(torus_internal, 0, 1)
    torus_half_expected = (2.0 * torus_internal - insertion) / (2.0 * torus_internal)

    torus_one = osp_three_point(
        n1=1,
        n2=0,
        n3=1,
        epsilon1=0,
        epsilon2=0,
        epsilon3=0,
        d1=torus_internal,
        d2=insertion,
        d3=torus_internal,
    ) / osp_norm(torus_internal, 1, 0)
    torus_one_expected = (
        2.0 * torus_internal + insertion * (insertion - 1.0)
    ) / (2.0 * torus_internal)

    parity_arguments = dict(
        n1=1,
        n2=0,
        n3=0,
        epsilon1=1,
        epsilon2=0,
        epsilon3=0,
        d1=d1,
        d2=d2,
        d3=d3,
    )
    parity_passed = (
        osp_sector_vertex(sector=0, **parity_arguments) == 0
        and osp_sector_vertex(sector=1, **parity_arguments) != 0
    )

    common_trifundamental = dict(
        occupations=(1, 1, 0),
        internal_weights=(0.83, 0.71, 0.92),
        external_pairs=((0.21, 0.38), (0.27, 0.43), (0.31, 0.49)),
    )
    even_sample = trifundamental_coefficient(
        fermions=(0, 0, 0),
        outer_sectors=(0, 0, 0),
        central_sector=0,
        **common_trifundamental,
    )
    odd_sample = trifundamental_coefficient(
        fermions=(1, 0, 0),
        outer_sectors=(1, 0, 0),
        central_sector=1,
        **common_trifundamental,
    )

    summary = GlobalCheckSummary(
        norm_recursion_error=float(max(norm_errors)),
        endpoint_formula_error=float(max(endpoint_errors)),
        primary_ward_table_error=float(max(primary_errors)),
        ccy_middle_chain_error=float(max(middle_errors)),
        torus_half_level_error=float(abs(torus_half - torus_half_expected)),
        torus_level_one_error=float(abs(torus_one - torus_one_expected)),
        parity_selection_passed=parity_passed,
        sample_even_trifundamental_coefficient=even_sample,
        sample_odd_trifundamental_coefficient=odd_sample,
    )
    for name in (
        "norm_recursion_error",
        "endpoint_formula_error",
        "primary_ward_table_error",
        "ccy_middle_chain_error",
        "torus_half_level_error",
        "torus_level_one_error",
    ):
        if getattr(summary, name) > 1.0e-12:
            raise AssertionError(f"global check failed: {name}")
    if not summary.parity_selection_passed:
        raise AssertionError("the trinion parity projector failed")
    return summary


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    summary = run_checks()
    if args.json:
        print(json.dumps(asdict(summary), indent=2, default=_json_default))
        return
    print("global osp(1|2) checks: PASS")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
