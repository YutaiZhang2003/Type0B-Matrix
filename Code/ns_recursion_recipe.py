"""Principal-sheet scalar kernels for all-NS graph c-recursion.

The functions in this module are the numerical form of the local rules in
``Machine Notes/c-Recursion/ns_genus_c_recursion.tex``.  They deliberately do
not know about a particular plumbing graph.  A graph implementation supplies
the ordered endpoint weights, vertex sectors, canonical endpoint signs,
plumbing powers, and Koszul transport; this module supplies only the common
weight-dependent Kac pole, inverse null slope, fusion factors, and the scalar
ordinary-edge or incidence-ordered self-loop kernel.

This is not an analytic-continuation engine.  Every square root is evaluated
on mpmath's principal branch independently on each call.  This convention is
faithful for the real standard-frame checks in this repository.  A caller
following complex weights around a continuation path or detuning an
exceptional locus must carry explicit continuous branch data outside this
module; it must not infer such continuation from these principal-sheet
helpers.

All arithmetic uses ``mpmath`` so the same implementation can be used by the
sphere, torus, and collision-aware genus-two recursions without a binary64
round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class NSPoleMP:
    """Fixed-weight NS Kac pole in the ordinary-central-charge convention."""

    r: int
    s: int
    weight: object
    b: object
    b_squared: object
    c: object
    dc_dh: object
    jacobian: object


def ns_c_pole_mp(r: int, s: int, weight) -> NSPoleMP:
    """Return the principal-sheet ``c_(r,s)(h)`` and ``J=-dc/dh``.

    Here ``c`` is the ordinary super-Virasoro central charge.  Belavin--Geiko
    denote ``hat_c = 2*c/3`` by ``c``; both their pole value and Jacobian are
    multiplied by ``3/2`` before entering this API.

    The discriminant and ``b`` square roots are fresh principal roots.  This
    function does not preserve a sheet along a complex continuation path.
    """

    if r < 2 or s < 1 or (r + s) % 2:
        raise ValueError("NS c-poles require r>=2, s>=1, and r+s even")
    h = mpmath.mpc(weight)
    rs = int(r) * int(s)
    discriminant = mpmath.sqrt(
        16 * h * h + 8 * (rs - 1) * h + (r - s) ** 2
    )
    x = -(4 * h + rs - 1 + discriminant) / (r * r - 1)
    b = mpmath.sqrt(x)
    c_value = mpmath.mpf("7.5") + 3 * x + 3 / x
    dx_dh = -(
        4 + (16 * h + 4 * (rs - 1)) / discriminant
    ) / (r * r - 1)
    dc_dh = 3 * (1 - 1 / (x * x)) * dx_dh
    return NSPoleMP(
        r=int(r),
        s=int(s),
        weight=h,
        b=b,
        b_squared=x,
        c=c_value,
        dc_dh=dc_dh,
        jacobian=-dc_dh,
    )


def ns_inverse_null_slope_mp(r: int, s: int, b):
    """Return the NS inverse null-norm slope ``A_(r,s)``."""

    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS labels require positive r,s with r+s even")
    b = mpmath.mpc(b)
    result = mpmath.mpf("0.5")
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            result *= mpmath.sqrt(2) / (p * b + q / b)
    return result


def ns_fusion_polynomial_mp(
    *,
    r: int,
    s: int,
    alpha: int,
    first_weight,
    second_weight,
    b,
):
    """Return the principal-sheet weight-only factor ``P_(r,s)^alpha``.

    The two momentum square roots are evaluated independently on their
    principal branches.  Canonical slot-permutation and component signs are
    not part of this scalar function.
    """

    if alpha not in (0, 1):
        raise ValueError("alpha must be zero or one")
    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS labels require positive r,s with r+s even")
    b = mpmath.mpc(b)
    background = b + 1 / b

    def momentum(weight):
        return mpmath.sqrt(background * background - 8 * mpmath.mpc(weight))

    lambda_i = momentum(first_weight)
    lambda_j = momentum(second_weight)
    congruence = 2 if alpha == 0 else 0
    denominator = 2 * mpmath.sqrt(2)
    result = mpmath.mpc(1)
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - r - s) % 4 != congruence:
                continue
            shift = p * b + q / b
            result *= (lambda_i - lambda_j + shift) / denominator
            result *= (lambda_i + lambda_j + shift) / denominator
    return result


def ns_ordinary_edge_scalar_kernel_mp(
    *,
    r: int,
    s: int,
    internal_weight,
    left_weights: tuple[object, object],
    right_weights: tuple[object, object],
    left_sector: int,
    right_sector: int,
) -> tuple[NSPoleMP, object, tuple[int, int]]:
    """Return the principal-sheet weight-only kernel for an ordinary edge.

    The returned scalar is ``J A P_left P_right``.  It is not the manuscript's
    full hatted endpoint operator: canonical slot-permutation signs, resolved
    component-parity phases, plumbing lifts, and Koszul transport remain the
    responsibility of the graph caller.
    """

    if left_sector not in (0, 1) or right_sector not in (0, 1):
        raise ValueError("endpoint sectors must be zero or one")
    pole = ns_c_pole_mp(r, s, internal_weight)
    left = ns_fusion_polynomial_mp(
        r=r,
        s=s,
        alpha=int(left_sector),
        first_weight=left_weights[0],
        second_weight=left_weights[1],
        b=pole.b,
    )
    right = ns_fusion_polynomial_mp(
        r=r,
        s=s,
        alpha=int(right_sector),
        first_weight=right_weights[0],
        second_weight=right_weights[1],
        b=pole.b,
    )
    residue = (
        pole.jacobian
        * ns_inverse_null_slope_mp(r, s, pole.b)
        * left
        * right
    )
    parity = (r * s) % 2
    return (
        pole,
        residue,
        (int(left_sector) ^ parity, int(right_sector) ^ parity),
    )


def ns_self_loop_scalar_kernel_mp(
    *,
    r: int,
    s: int,
    handle_weight,
    external_weight,
    sector: int,
) -> tuple[NSPoleMP, object, int]:
    """Return the principal-sheet incidence-ordered self-loop scalar kernel.

    The first incidence sees ``(external, h)`` in sector ``alpha``.  The
    second sees ``(external, h+rs/2)`` in the intermediate sector
    ``alpha xor (rs mod 2)``.  The final sector is unchanged because the loop
    meets the same vertex twice.  This includes the intrinsic toric sign but
    not graph-level plumbing-lift or Koszul transport.
    """

    if sector not in (0, 1):
        raise ValueError("sector must be zero or one")
    rs = int(r) * int(s)
    pole = ns_c_pole_mp(r, s, handle_weight)
    first = ns_fusion_polynomial_mp(
        r=r,
        s=s,
        alpha=int(sector),
        first_weight=external_weight,
        second_weight=handle_weight,
        b=pole.b,
    )
    second = ns_fusion_polynomial_mp(
        r=r,
        s=s,
        alpha=int(sector) ^ (rs % 2),
        first_weight=external_weight,
        second_weight=mpmath.mpc(handle_weight) + mpmath.mpf(rs) / 2,
        b=pole.b,
    )
    residue = (
        (-1) ** (int(sector) * rs)
        * pole.jacobian
        * ns_inverse_null_slope_mp(r, s, pole.b)
        * first
        * second
    )
    return pole, residue, int(sector)


__all__ = [
    "NSPoleMP",
    "ns_c_pole_mp",
    "ns_fusion_polynomial_mp",
    "ns_inverse_null_slope_mp",
    "ns_ordinary_edge_scalar_kernel_mp",
    "ns_self_loop_scalar_kernel_mp",
]
